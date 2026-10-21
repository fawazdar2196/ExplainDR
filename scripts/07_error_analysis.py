import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import GRADE_LABELS  # noqa: E402

LESIONS = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
REFERABLE = 2


def lesion_minimum_grades(splits):
    # the lowest grade each lesion ever appears with in the training labels
    train = splits[splits["split"] == "train"]
    return {les: int(train.loc[train[les] == 1, "grade"].min()) for les in LESIONS if train[les].any()}


def gallery(df, image_dir, out_file, tile=320, per_row=4, gap=6):
    # plain grid of photos, row by row in the same order as the matching csv
    rows = (len(df) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * tile + (per_row - 1) * gap, rows * tile + (rows - 1) * gap), "white")
    for k, name in enumerate(df["name"]):
        with Image.open(Path(image_dir) / f"{name}.jpg") as im:
            r, c = divmod(k, per_row)
            sheet.paste(im.convert("RGB").resize((tile, tile)), (c * (tile + gap), r * (tile + gap)))
    sheet.save(out_file, quality=92)


def pct(a, b):
    return f"{a} of {b} ({a / b:.1%})" if b else f"{a} of 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/05_train_v2/selection.json")
    ap.add_argument("--test-dir", default="outputs/06_test")
    ap.add_argument("--splits", default="outputs/03_splits/splits.csv")
    ap.add_argument("--quality", default="outputs/04_preprocess/quality.csv")
    ap.add_argument("--image-dir", default="data/processed_768")
    ap.add_argument("--out", default="outputs/07_errors")
    ap.add_argument("--n-moderate", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = json.loads(Path(args.selection).read_text())["chosen_run"]
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(Path(args.test_dir) / run / "test_predictions.csv")
    splits = pd.read_csv(args.splits)
    df = preds.merge(splits[["name"] + LESIONS], on="name", how="left")
    quality = pd.read_csv(args.quality)
    df = df.merge(quality[["name", "sharpness"]], on="name", how="left")
    if df[LESIONS].isna().any().any():
        raise SystemExit("some test images have no lesion labels in the splits file")

    probs = df[[f"p{k}" for k in range(5)]].to_numpy()
    df["confidence"] = probs.max(axis=1).round(3)
    df["p_true"] = probs[np.arange(len(df)), df["grade"]].round(3)
    df["error"] = df["pred"] - df["grade"]
    y, p = df["grade"], df["pred"]

    print(f"run: {run}  test images: {len(df)}")
    print("\n== Errors by true grade ==")
    rows = []
    for g in range(5):
        d = df[y == g]
        rows.append({"true grade": GRADE_LABELS[g], "images": len(d), "correct": f"{(d['error'] == 0).mean():.1%}",
                     "too low": f"{(d['error'] < 0).mean():.1%}", "too high": f"{(d['error'] > 0).mean():.1%}",
                     "2+ grades too low": int((d["error"] <= -2).sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== Errors that change the clinical decision ==")
    ref_t, ref_p = y >= REFERABLE, p >= REFERABLE
    print(f"referable (grade 2 to 4) called not referable (0 or 1): {pct(int((ref_t & ~ref_p).sum()), int(ref_t.sum()))}")
    print(f"not referable called referable: {pct(int((~ref_t & ref_p).sum()), int((~ref_t).sum()))}")
    sight = y >= 3
    print(f"severe or PDR called mild or No DR: {pct(int((sight & ~ref_p).sum()), int(sight.sum()))}")
    print(f"PDR called a lower grade: {pct(int(((y == 4) & (p < 4)).sum()), int((y == 4).sum()))}")
    missed = df[ref_t & ~ref_p]
    print("missed referrals by true grade: " + ", ".join(f"{GRADE_LABELS[g]} {int((missed['grade'] == g).sum())}" for g in range(2, 5)))

    minimum = lesion_minimum_grades(splits)
    df["lesion_min_grade"] = [max([minimum[les] for les in LESIONS if r[les] == 1 and les in minimum], default=0)
                              for _, r in df.iterrows()]
    print("\n== Predictions below what the lesion labels imply ==")
    print("lowest grade each lesion has in the training labels: " + ", ".join(f"{k} {v}" for k, v in minimum.items()))
    rows = []
    for les in LESIONS:
        d = df[df[les] == 1]
        if les not in minimum or not len(d):
            continue
        below = int((d["pred"] < minimum[les]).sum())
        rows.append({"lesion": les, "test images with it": len(d), "implies grade at least": minimum[les],
                     "predicted lower": below, "share": f"{below / len(d):.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))
    contradicted = df[df["pred"] < df["lesion_min_grade"]]
    print(f"any prediction below the lesion-implied grade: {pct(len(contradicted), len(df))}")

    print("\n== Moderate photos, with and without a haemorrhage label ==")
    mod = df[y == 2]
    rows = []
    for label, d in [("with haemorrhage (IH)", mod[mod["IH"] == 1]), ("without haemorrhage", mod[mod["IH"] == 0])]:
        rows.append({"group": label, "images": len(d), "correct": f"{(d['pred'] == 2).mean():.1%}",
                     "called mild": f"{(d['pred'] == 1).mean():.1%}", "called severe": f"{(d['pred'] == 3).mean():.1%}",
                     "median lesion types": float(d[LESIONS].sum(axis=1).median()) if len(d) else 0.0})
    print(pd.DataFrame(rows).to_string(index=False))

    cols = ["name", "grade", "pred", "confidence", "p_true", "lesion_min_grade", "sharpness"] + LESIONS
    df[df["error"] != 0][cols].sort_values(["grade", "pred"]).to_csv(out / "errors.csv", index=False)

    serious = df[sight & ~ref_p].sort_values(["grade", "pred", "name"], ascending=[False, True, True])
    serious[cols].to_csv(out / "serious_underestimates.csv", index=False)
    if len(serious):
        gallery(serious, args.image_dir, out / "serious_underestimates.jpg")

    # equal numbers with and without a haemorrhage label, so both kinds are seen
    as_mild = mod[mod["pred"] == 1]
    half = args.n_moderate // 2
    picks = [d.sample(min(half, len(d)), random_state=args.seed) for d in (as_mild[as_mild["IH"] == 1], as_mild[as_mild["IH"] == 0])]
    sample = pd.concat(picks)
    sample[cols].to_csv(out / "moderate_called_mild.csv", index=False)
    if len(sample):
        gallery(sample, args.image_dir, out / "moderate_called_mild.jpg")

    print("\n== Serious underestimates (pictured in serious_underestimates.jpg, row by row) ==")
    print(serious[["name", "grade", "pred", "confidence", "p_true", "sharpness"] + LESIONS].to_string(index=False) if len(serious) else "none")
    # diseased photos are blurrier in general, so compare with severe and PDR photos graded correctly
    ref = df[sight & (df["error"] == 0)]
    print(f"\nmedian sharpness: correctly graded severe or PDR photos {ref['sharpness'].median():.1f}, "
          f"serious underestimates {serious['sharpness'].median():.1f}")


if __name__ == "__main__":
    main()
