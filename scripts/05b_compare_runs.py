import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/05_train_v2")
    args = ap.parse_args()

    rows = []
    for run in sorted(Path(args.runs).iterdir()):
        hist = run / "history.csv"
        if not run.is_dir() or not hist.exists() or run.name.endswith("_smoke"):
            continue
        h = pd.read_csv(hist)
        best = h.loc[h["val_qwk"].idxmax()]  # first epoch with the top score, as in training
        rows.append({
            "run": run.name,
            "epochs": len(h),
            "best epoch": int(best["epoch"]),
            "val accuracy": round(best["val_accuracy"], 3),
            "val macro-F1": round(best["val_macro_f1"], 3),
            "val QWK": round(best["val_qwk"], 3),
            "min per epoch": round(h["minutes"].median(), 1),
            "total min": round(h["minutes"].sum()),
            "peak GPU GB": round(h["gpu_gb"].max(), 1),
            "peak reserved GB": round(h["gpu_reserved_gb"].max(), 1) if "gpu_reserved_gb" in h else None,
        })
    if not rows:
        raise SystemExit(f"no finished runs in {args.runs}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
