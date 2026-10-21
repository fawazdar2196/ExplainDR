import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

LESIONS = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}
SPLITS = ["train", "test", "total"]


def find_name_column(df, stems):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        hit = df[col].dropna().astype(str).map(lambda s: Path(s).stem).isin(stems).mean()
        if hit > 0.5:
            return col
    return None


def parse_lesions(value):
    digits = re.findall(r"\d", str(value))
    return [int(d) for d in digits] if len(digits) == 7 else None


def by_split(df, col):
    t = pd.crosstab(df[col], df["split"], margins=True, margins_name="total")
    return t.reindex(columns=[c for c in SPLITS if c in t.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MMRDR-CFP")
    args = ap.parse_args()
    root = Path(args.root)

    df = pd.read_csv(root / "FP.csv")
    print("== FP.csv ==")
    print(f"rows: {len(df)}  columns: {list(df.columns)}")
    print(df.head(3).to_string())
    missing = df.isna().sum()
    print("missing values:", missing[missing > 0].to_dict() or "none")

    images = {p.stem: p for p in (root / "img").iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}}
    col = find_name_column(df, set(images))
    if col is None:
        sys.exit("no column in FP.csv matches the image file names")

    df["stem"] = df[col].astype(str).map(lambda s: Path(s).stem)
    df["split"] = df["stem"].str[:2].map({"tr": "train", "ts": "test"})
    df["grade_name"] = df["grade"].map(lambda g: f"{g} {GRADES.get(g, '?')}")

    print(f"\n== Files ==\nimage name column: '{col}'")
    print("images on disk:", len(images))
    print("rows with unknown tr/ts prefix:", int(df["split"].isna().sum()))
    print("duplicate names in csv:", int(df["stem"].duplicated().sum()))
    print("csv rows without an image:", int((~df["stem"].isin(images)).sum()))
    print("images without a csv row:", len(set(images) - set(df["stem"])))

    print("\n== Grade counts ==")
    print(by_split(df, "grade_name").to_string())
    pct = pd.crosstab(df["grade_name"], df["split"], normalize="columns") * 100
    print("\n% within each split")
    print(pct.reindex(columns=[c for c in SPLITS if c in pct.columns]).round(1).to_string())

    print("\n== Laterality (0 left, 1 right) ==")
    print(by_split(df, "lr").to_string())

    lesions = df["lesion"].map(parse_lesions)
    ok = lesions.notna()
    print(f"\n== Lesions ==\nrows that failed to parse: {int((~ok).sum())}")
    les = pd.DataFrame(lesions[ok].tolist(), columns=LESIONS, index=df.index[ok])
    grade = df.loc[ok, "grade"]
    print("\n% of images with each lesion, by grade")
    print((les.groupby(df.loc[ok, "grade_name"]).mean() * 100).round(1).to_string())

    any_lesion = les.sum(axis=1) > 0
    print("\n== Label consistency ==")
    print("grade 0 with any lesion:", int(((grade == 0) & any_lesion).sum()))
    print("grade 1 with a lesion other than MA:", int(((grade == 1) & (les[LESIONS[1:]].sum(axis=1) > 0)).sum()))
    print("grade 1 to 4 with no lesion:", int(((grade > 0) & ~any_lesion).sum()))
    print("grade 0 to 3 with NV or VH:", int(((grade < 4) & ((les["NV"] + les["VH"]) > 0)).sum()))

    sizes, bad = Counter(), []
    for p in images.values():
        try:
            with Image.open(p) as im:
                sizes[(im.size, im.mode)] += 1
        except Exception:
            bad.append(p.name)
    print(f"\n== Images ==\nunreadable: {len(bad)} {bad[:10]}")
    print("distinct size and mode combinations:", len(sizes))
    for (size, mode), n in sizes.most_common(10):
        print(f"  {size[0]}x{size[1]} {mode}  {n}")


if __name__ == "__main__":
    main()
