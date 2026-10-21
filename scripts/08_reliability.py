import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import apply_temperature, calibration_bins, expected_calibration_error, fit_temperature  # noqa: E402
from src.plots import GRADE_LABELS, reliability_diagram, review_curve  # noqa: E402

REFERABLE = 2
FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def load(path):
    df = pd.read_csv(path)
    return df, df[[f"p{k}" for k in range(5)]].to_numpy(), df["grade"].to_numpy(), df["pred"].to_numpy()


def certainty_scores(probs):
    # higher means more certain; the least certain photos get flagged first
    p_ref = probs[:, REFERABLE:].sum(axis=1)
    return {"grade confidence": probs.max(axis=1), "referral certainty": np.abs(p_ref - 0.5)}


def balanced_accuracy(y, p):
    present = [g for g in range(5) if (y == g).any()]
    return float(np.mean([(p[y == g] == g).mean() for g in present]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/05_train_v2/selection.json")
    ap.add_argument("--test-dir", default="outputs/06_test")
    ap.add_argument("--quality", default="outputs/04_preprocess/quality.csv")
    ap.add_argument("--out", default="outputs/08_reliability")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    selection = Path(args.selection)
    run = json.loads(selection.read_text())["chosen_run"]
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    _, val_probs, val_y, val_pred = load(selection.parent / run / "val_predictions.csv")
    test_df, test_probs, y, pred = load(Path(args.test_dir) / run / "test_predictions.csv")

    # 1. calibration: the temperature is fitted on validation only, then applied to test
    t = fit_temperature(val_probs, val_y)
    val_cal, test_cal = apply_temperature(val_probs, t), apply_temperature(test_probs, t)
    if not (test_cal.argmax(1) == test_probs.argmax(1)).all():
        raise SystemExit("temperature scaling changed a predicted grade, which should never happen")
    correct = pred == y

    def ece(probs, yy, pp):
        return expected_calibration_error(probs.max(1), pp == yy)

    idx = rng.integers(0, len(y), size=(args.n_boot, len(y)))
    ece_raw = [ece(test_probs[i], y[i], pred[i]) for i in idx]
    ece_cal = [ece(test_cal[i], y[i], pred[i]) for i in idx]
    calib = {
        "temperature": round(t, 3),
        "val_ece_before": round(ece(val_probs, val_y, val_pred), 4), "val_ece_after": round(ece(val_cal, val_y, val_pred), 4),
        "test_ece_before": round(ece(test_probs, y, pred), 4), "test_ece_before_ci": [round(float(np.percentile(ece_raw, q)), 4) for q in (2.5, 97.5)],
        "test_ece_after": round(ece(test_cal, y, pred), 4), "test_ece_after_ci": [round(float(np.percentile(ece_cal, q)), 4) for q in (2.5, 97.5)],
        "test_accuracy": round(float(correct.mean()), 4),
        "test_mean_confidence_before": round(float(test_probs.max(1).mean()), 4),
        "test_mean_confidence_after": round(float(test_cal.max(1).mean()), 4),
    }
    print(f"run: {run}  validation photos: {len(val_y)}  test photos: {len(y)}")
    print("\n== 1. Calibration ==")
    print(f"temperature fitted on validation: {t:.3f}")
    print(f"test accuracy {calib['test_accuracy']:.3f}  mean confidence before {calib['test_mean_confidence_before']:.3f}, after {calib['test_mean_confidence_after']:.3f}")
    print(f"calibration error (ECE), validation: before {calib['val_ece_before']:.3f}, after {calib['val_ece_after']:.3f}")
    lo_b, hi_b = calib["test_ece_before_ci"]
    lo_a, hi_a = calib["test_ece_after_ci"]
    print(f"calibration error (ECE), test: before {calib['test_ece_before']:.3f} [{lo_b:.3f}, {hi_b:.3f}], "
          f"after {calib['test_ece_after']:.3f} [{lo_a:.3f}, {hi_a:.3f}]")
    reliability_diagram([("Before scaling", calibration_bins(test_probs.max(1), correct)),
                         ("After scaling", calibration_bins(test_cal.max(1), correct))], out / "reliability_diagram.png")

    # 2. flagging the least certain photos for review; cut-offs come from validation
    missed_ref = (y >= REFERABLE) & (pred < REFERABLE)
    serious = (y >= 3) & (pred < REFERABLE)
    errors = ~correct
    val_scores, test_scores = certainty_scores(val_cal), certainty_scores(test_cal)
    rows = []
    for name in test_scores:
        for f in FRACTIONS:
            cut = np.quantile(val_scores[name], f)
            flag = test_scores[name] <= cut
            rows.append({"certainty score": name, "target share": f"{f:.0%}", "test photos flagged": f"{flag.mean():.1%}",
                         "accuracy of the rest": f"{correct[~flag].mean():.1%}",
                         "errors caught": f"{(flag & errors).sum()} of {errors.sum()}",
                         "missed referrals caught": f"{(flag & missed_ref).sum()} of {missed_ref.sum()}",
                         "serious underestimates caught": f"{(flag & serious).sum()} of {serious.sum()}"})
    review = pd.DataFrame(rows)
    review.to_csv(out / "review_table.csv", index=False)
    print("\n== 2. Review rule (cut-offs set on validation, results on test) ==")
    print(f"accuracy with no review: {correct.mean():.1%}")
    print(review.to_string(index=False))

    grid = np.arange(0, 0.51, 0.01)
    curves, flagged_pct = [], None
    for name, label in [("grade confidence", "Lowest grade confidence"), ("referral certainty", "Closest to the referral line")]:
        cuts = np.quantile(val_scores[name], grid)
        flags = [test_scores[name] <= c if f > 0 else np.zeros(len(y), bool) for c, f in zip(cuts, grid)]
        flagged_pct = [100 * fl.mean() for fl in flags]
        curves.append((label, [100 * (fl & missed_ref).sum() / max(missed_ref.sum(), 1) for fl in flags]))
    review_curve(flagged_pct, curves, out / "review_curve.png")

    # 3. image quality: compare within each grade, because healthy photos are sharper on average
    quality = pd.read_csv(args.quality)[["name", "sharpness"]]
    q = test_df[["name"]].merge(quality, on="name", how="left")["sharpness"].to_numpy()
    if np.isnan(q).any():
        raise SystemExit("some test photos have no sharpness value")
    cuts = np.quantile(q, [1 / 3, 2 / 3])
    level = np.digitize(q, cuts)
    names = ["blurry third", "middle third", "sharp third"]
    rows = []
    for g in range(5):
        row = {"true grade": GRADE_LABELS[g]}
        for k, lv in enumerate(names):
            m = (y == g) & (level == k)
            row[lv] = f"{(pred[m] == g).mean():.1%} (n={m.sum()})" if m.any() else "-"
        rows.append(row)
    for label in ["balanced accuracy", "mean confidence"]:
        row = {"true grade": label}
        for k, lv in enumerate(names):
            m = level == k
            if label == "balanced accuracy":
                boots = [balanced_accuracy(y[m][i], pred[m][i]) for i in rng.integers(0, m.sum(), size=(args.n_boot, m.sum()))]
                row[lv] = f"{balanced_accuracy(y[m], pred[m]):.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
            else:
                row[lv] = f"{test_cal.max(1)[m].mean():.3f}"
        rows.append(row)
    quality_table = pd.DataFrame(rows)
    quality_table.to_csv(out / "quality_table.csv", index=False)
    print(f"\n== 3. Accuracy by image sharpness (thirds of the test set, cut at {cuts[0]:.1f} and {cuts[1]:.1f}) ==")
    print("share of each true grade graded correctly, within each third:")
    print(quality_table.to_string(index=False))

    cal = test_df[["name", "grade", "pred"]].copy()
    for k in range(5):
        cal[f"pc{k}"] = test_cal[:, k].round(6)
    cal["confidence"] = test_cal.max(1).round(4)
    cal["p_referable"] = test_cal[:, REFERABLE:].sum(1).round(4)
    cal["sharpness"] = q.round(2)
    cal.to_csv(out / "test_predictions_calibrated.csv", index=False)
    (out / "reliability.json").write_text(json.dumps({"calibration": calib, "sharpness_cuts": [round(c, 2) for c in cuts]}, indent=2))


if __name__ == "__main__":
    main()
