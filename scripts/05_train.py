import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import FundusDataset, build_transforms, load_split  # noqa: E402
from src.metrics import grading_metrics  # noqa: E402
from src.models import NUM_GRADES, build_model, build_optimizer  # noqa: E402


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        target = cfg
        for p in parents:
            target = target[p]
        target[last] = yaml.safe_load(value)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels):
    # rare grades count more, so the model can't just predict No DR
    counts = np.bincount(labels, minlength=NUM_GRADES)
    return torch.tensor(len(labels) / (NUM_GRADES * np.maximum(counts, 1)), dtype=torch.float32)


def amp_dtype_for(cfg, device):
    if device.type != "cuda" or cfg["amp"] == "none":
        return None
    if cfg["amp"] == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_epoch(model, loader, device, amp_dtype, criterion, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probs, labels, names = 0.0, [], [], []
    for x, y, n in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = criterion(logits.float(), y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
        total_loss += loss.item() * len(y)
        probs.append(torch.softmax(logits.float(), dim=1).detach().cpu())
        labels.append(y.cpu())
        names.extend(n)
    labels = torch.cat(labels).numpy()
    return total_loss / len(labels), torch.cat(probs).numpy(), labels, names


def save_predictions(path, names, labels, probs):
    df = pd.DataFrame({"name": names, "grade": labels, "pred": probs.argmax(1)})
    for k in range(NUM_GRADES):
        df[f"p{k}"] = probs[:, k].round(6)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="change settings, e.g. epochs=5 augment.blur_p=0")
    ap.add_argument("--smoke", action="store_true", help="tiny one-epoch run to check everything works")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.smoke:
        cfg["run_name"] += "_smoke"
        cfg["epochs"] = 1
    out = Path(cfg["output_dir"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_df = load_split(cfg["splits"], "train")
    val_df = load_split(cfg["splits"], "val")
    if args.smoke:
        train_df = train_df.groupby("grade").head(8).reset_index(drop=True)
        val_df = val_df.groupby("grade").head(8).reset_index(drop=True)

    gen = torch.Generator().manual_seed(cfg["seed"])
    loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                     persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(FundusDataset(train_df, cfg["image_dir"], build_transforms(cfg, True)),
                              batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=gen, **loader_kw)
    val_loader = DataLoader(FundusDataset(val_df, cfg["image_dir"], build_transforms(cfg, False)),
                            batch_size=cfg["batch_size"] * 2, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    weights = class_weights(train_df["grade"].to_numpy()) if cfg["class_weights"] == "inverse_frequency" else None
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    optimizer = build_optimizer(model, cfg, fused=device.type == "cuda")

    total_steps = cfg["epochs"] * len(train_loader)
    warm_steps = cfg["warmup_epochs"] * len(train_loader)

    def lr_factor(step):
        # short linear warm-up, then cosine decay to zero
        if step < warm_steps:
            return (step + 1) / warm_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))

    scheduler = LambdaLR(optimizer, lr_factor)
    amp_dtype = amp_dtype_for(cfg, device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}  precision: {amp_dtype or 'float32'}")
    print(f"train: {len(train_df)} images  val: {len(val_df)} images  steps per epoch: {len(train_loader)}")
    if weights is not None:
        print("class weights:", [round(w, 2) for w in weights.tolist()])

    metric = cfg["select_metric"]
    best, best_epoch, history = -math.inf, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr_loss, tr_probs, tr_y, _ = run_epoch(model, train_loader, device, amp_dtype, criterion, optimizer, scheduler, scaler)
        if device.type == "cuda":
            torch.cuda.empty_cache()  # stops reserved memory spilling into normal RAM on Windows
        va_loss, va_probs, va_y, va_names = run_epoch(model, val_loader, device, amp_dtype, criterion)
        tm, vm = grading_metrics(tr_y, tr_probs.argmax(1)), grading_metrics(va_y, va_probs.argmax(1))
        gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        gpu_reserved = torch.cuda.max_memory_reserved() / 1e9 if device.type == "cuda" else 0.0
        if device.type == "cuda":
            torch.cuda.empty_cache()

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_macro_f1": tm["macro_f1"], "train_qwk": tm["qwk"],
                        "val_loss": va_loss, "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
                        "val_qwk": vm["qwk"], "lr": max(scheduler.get_last_lr()),
                        "minutes": (time.time() - start) / 60, "gpu_gb": gpu_gb, "gpu_reserved_gb": gpu_reserved})
        pd.DataFrame(history).round(5).to_csv(out / "history.csv", index=False)

        improved = vm[metric] > best
        if improved:
            best, best_epoch = vm[metric], epoch
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": vm}, out / "best.pt")
            save_predictions(out / "val_predictions.csv", va_names, va_y, va_probs)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f}  acc {vm['accuracy']:.3f}  "
              f"macro-F1 {vm['macro_f1']:.3f}  QWK {vm['qwk']:.3f} | {history[-1]['minutes']:.1f} min  "
              f"{gpu_gb:.1f}/{gpu_reserved:.1f} GB{'  <- best so far' if improved else ''}")
        # patience: null trains the full budget, so every model gets the same number of epochs
        if cfg.get("patience") and epoch - best_epoch >= cfg["patience"]:
            print(f"no improvement in validation {metric} for {cfg['patience']} epochs, stopping")
            break

    best_row = history[best_epoch - 1]
    print(f"\nbest epoch: {best_epoch}  val accuracy {best_row['val_accuracy']:.3f}  "
          f"macro-F1 {best_row['val_macro_f1']:.3f}  QWK {best_row['val_qwk']:.3f}")


if __name__ == "__main__":
    main()
