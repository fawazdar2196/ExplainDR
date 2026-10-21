import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import grading_metrics  # noqa: E402


def param_count(model_name):
    with torch.device("meta"):
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
    return sum(p.numel() for p in model.parameters())


def load_runs(runs_dir):
    runs = {}
    for run in sorted(Path(runs_dir).iterdir()):
        preds = run / "val_predictions.csv"
        if not run.is_dir() or run.name.endswith("_smoke") or not preds.exists():
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        hist = pd.read_csv(run / "history.csv")
        runs[run.name] = {"preds": pd.read_csv(preds).sort_values("name").reset_index(drop=True),
                          "model": cfg["model"], "image_size": cfg["image_size"],
                          "epochs_done": len(hist), "epochs_planned": cfg["epochs"],
                          "best_epoch": int(hist.loc[hist["val_qwk"].idxmax(), "epoch"])}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/05_train_v2")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if len(runs) < 2:
        raise SystemExit("need at least two finished runs to compare")
    unfinished = [n for n, r in runs.items() if r["epochs_done"] < r["epochs_planned"]]
    if unfinished:
        raise SystemExit(f"still training: {unfinished}")

    names = list(runs)
    y = runs[names[0]]["preds"]["grade"].to_numpy()
    for n in names[1:]:
        if not runs[n]["preds"]["name"].equals(runs[names[0]]["preds"]["name"]):
            raise SystemExit(f"{n} was validated on different images")

    rows = []
    for n in names:
        m = grading_metrics(y, runs[n]["preds"]["pred"].to_numpy())
        runs[n]["metrics"] = m
        runs[n]["params"] = param_count(runs[n]["model"])
        rows.append({"run": n, "best epoch": runs[n]["best_epoch"], "accuracy": round(m["accuracy"], 3),
                     "macro-F1": round(m["macro_f1"], 3), "QWK": round(m["qwk"], 3),
                     "parameters (M)": round(runs[n]["params"] / 1e6, 1)})
    print(f"== Validation results ({len(y)} images, best epoch of each run) ==")
    print(pd.DataFrame(rows).to_string(index=False))

    # paired bootstrap: every model is scored on the same resampled images
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(y), size=(args.n_boot, len(y)))
    boot = {n: np.array([grading_metrics(y[i], runs[n]["preds"]["pred"].to_numpy()[i])["qwk"] for i in idx]) for n in names}

    leader = max(names, key=lambda n: runs[n]["metrics"]["qwk"])
    tied = [leader]
    print(f"\n== Paired bootstrap against the leader, {leader} ({args.n_boot} resamples) ==")
    for n in names:
        if n == leader:
            continue
        gap = boot[leader] - boot[n]
        lo, hi = np.percentile(gap, [2.5, 97.5])
        clear = lo > 0
        if not clear:
            tied.append(n)
        print(f"{leader} minus {n}: QWK gap {runs[leader]['metrics']['qwk'] - runs[n]['metrics']['qwk']:+.3f}, "
              f"95% interval [{lo:+.3f}, {hi:+.3f}], {'leader clearly better' if clear else 'could be chance'}")

    # tie rule: the simplest model, fewest parameters first, then smallest images
    chosen = min(tied, key=lambda n: (runs[n]["params"], runs[n]["image_size"]))
    if len(tied) == 1:
        reason = "highest validation QWK, and the gap to every other model holds up under the bootstrap"
    else:
        reason = f"tied with {[t for t in tied if t != chosen]}, so the simpler model is chosen"
    print(f"\n== Decision ==\nchosen: {chosen} ({reason})")

    lo, hi = np.percentile(boot[chosen], [2.5, 97.5])
    out = {"chosen_run": chosen, "reason": reason, "rule": "highest val QWK; if the gap could be chance, the simpler model",
           "val_qwk": round(runs[chosen]["metrics"]["qwk"], 4), "val_qwk_95ci": [round(lo, 4), round(hi, 4)],
           "candidates": rows}
    Path(args.runs, "selection.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
