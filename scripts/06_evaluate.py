import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import FundusDataset, build_transforms, load_split  # noqa: E402
from src.metrics import LABELS, grading_metrics  # noqa: E402
from src.models import NUM_GRADES, build_model  # noqa: E402
from src.plots import GRADE_LABELS, confusion_matrix_figure  # noqa: E402

REFERABLE = 2  # moderate NPDR or worse


def predict(run_dir, device, batch_size):
    # our own checkpoint, written by 05_train.py
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    test_df = load_split(cfg["splits"], "test")
    loader = DataLoader(FundusDataset(test_df, cfg["image_dir"], build_transforms(cfg, False)),
                        batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"])
    use_amp = device.type == "cuda" and cfg["amp"] != "none"
    probs, names, labels = [], [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for x, y, n in loader:
            probs.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
            labels.append(y)
            names.extend(n)
    probs = torch.cat(probs).numpy()
    df = pd.DataFrame({"name": names, "grade": torch.cat(labels).numpy(), "pred": probs.argmax(1)})
    for k in range(NUM_GRADES):
        df[f"p{k}"] = probs[:, k].round(6)
    return df, cfg, ckpt["epoch"]


def referable(y, p):
    t, q = y >= REFERABLE, p >= REFERABLE
    return {"sensitivity": (t & q).sum() / max(t.sum(), 1), "specificity": (~t & ~q).sum() / max((~t).sum(), 1)}


def all_metrics(y, p):
    m = grading_metrics(y, p)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    for k in LABELS:
        m[f"precision_{k}"], m[f"recall_{k}"], m[f"f1_{k}"] = prec[k], rec[k], f1[k]
    m.update({f"referable_{k}": v for k, v in referable(y, p).items()})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/05_train_v2/selection.json")
    ap.add_argument("--run", default=None, help="evaluate another finished run, only after the choice is locked")
    ap.add_argument("--out", default="outputs/06_test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    selection = Path(args.selection)
    chosen = json.loads(selection.read_text())["chosen_run"]
    run_name = args.run or chosen
    run_dir = selection.parent / run_name
    out = Path(args.out) / run_name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, cfg, epoch = predict(run_dir, device, args.batch_size)
    df.to_csv(out / "test_predictions.csv", index=False)
    y, p = df["grade"].to_numpy(), df["pred"].to_numpy()

    point = all_metrics(y, p)
    rng = np.random.default_rng(args.seed)
    boot = pd.DataFrame([all_metrics(y[i], p[i]) for i in rng.integers(0, len(y), size=(args.n_boot, len(y)))])
    ci = {k: (float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))) for k in point}

    def show(key):
        lo, hi = ci[key]
        return f"{point[key]:.3f} [{lo:.3f}, {hi:.3f}]"

    print(f"run: {run_name} ({'chosen model' if run_name == chosen else 'for context only, not the chosen model'})")
    print(f"checkpoint: epoch {epoch}  test images: {len(y)}  95% intervals from {args.n_boot} bootstrap resamples")
    print("\n== Overall ==")
    for key, label in [("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("qwk", "QWK")]:
        print(f"{label:<10} {show(key)}")

    sizes = np.bincount(y, minlength=NUM_GRADES)
    rows = [{"grade": GRADE_LABELS[k], "images": int(sizes[k]), "precision": round(point[f"precision_{k}"], 3),
             "recall": show(f"recall_{k}"), "F1": round(point[f"f1_{k}"], 3)} for k in LABELS]
    print("\n== Per grade ==")
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n== Referable DR (grade {REFERABLE} or above) ==")
    print(f"sensitivity {show('referable_sensitivity')}")
    print(f"specificity {show('referable_specificity')}")

    cm = confusion_matrix(y, p, labels=LABELS)
    pd.DataFrame(cm, index=[f"true {g}" for g in GRADE_LABELS], columns=[f"pred {g}" for g in GRADE_LABELS]).to_csv(out / "confusion_matrix.csv")
    confusion_matrix_figure(cm, out / "confusion_matrix.png")
    print("\n== Confusion matrix (rows true, columns predicted) ==")
    print(pd.DataFrame(cm, index=GRADE_LABELS, columns=[g.split()[0] for g in GRADE_LABELS]).to_string())

    under, over = int((p < y).sum()), int((p > y).sum())
    print(f"\nunderestimated: {under} ({under / len(y):.1%})  overestimated: {over} ({over / len(y):.1%})  "
          f"off by 2 or more grades: {int((abs(p - y) >= 2).sum())}")

    results = {"run": run_name, "chosen": run_name == chosen, "epoch": epoch, "n_test": int(len(y)),
               "metrics": {k: round(float(v), 4) for k, v in point.items()},
               "ci95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
               "confusion_matrix": cm.tolist(), "underestimated": under, "overestimated": over}
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
