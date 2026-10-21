import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LESIONS = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}
ORDER = ["train", "val", "test", "excluded"]


def root_of(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def copy_clusters(vecs, threshold, block=1024):
    # images joined by any chain of copies end up in the same cluster
    parent = np.arange(len(vecs))
    for s in range(0, len(vecs), block):
        sims = vecs[s:s + block] @ vecs.T
        for a, b in zip(*np.nonzero(sims >= threshold)):
            a += s
            if b > a:
                ra, rb = root_of(parent, a), root_of(parent, b)
                if ra != rb:
                    parent[rb] = ra
    return np.array([root_of(parent, k) for k in range(len(vecs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MMRDR-CFP")
    ap.add_argument("--fingerprints", default="outputs/02_duplicates/fingerprints.npy")
    ap.add_argument("--out", default="outputs/03_splits")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--folds", type=int, default=7, help="validation gets one fold, about 14 percent")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "FP.csv")
    df["name"] = df["image"].map(lambda s: Path(s).stem)
    df["split"] = df["name"].str[:2].map({"tr": "train", "ts": "test"})
    flags = df["lesion"].map(lambda s: [int(d) for d in re.findall(r"\d", str(s))])
    df[LESIONS] = pd.DataFrame(flags.tolist(), index=df.index)

    vecs = np.load(args.fingerprints)
    if len(vecs) != len(df):
        raise SystemExit("fingerprints don't match FP.csv, rerun step 2")
    df["cluster"] = copy_clusters(vecs, args.threshold)

    # training copies of test images would leak, so they leave training
    test_clusters = set(df.loc[df["split"] == "test", "cluster"])
    leak = (df["split"] == "train") & df["cluster"].isin(test_clusters)
    df.loc[leak, "split"] = "excluded"

    pool = df.index[df["split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    _, val_pos = next(sgkf.split(pool, df.loc[pool, "grade"], df.loc[pool, "cluster"]))
    df.loc[pool[val_pos], "split"] = "val"

    # no cluster may sit in more than one of train, val and test
    used = df[df["split"] != "excluded"]
    spans = used.groupby("cluster")["split"].nunique()
    if (spans > 1).any():
        raise SystemExit(f"{int((spans > 1).sum())} clusters cross splits")

    sizes = df["cluster"].value_counts()
    multi = sizes[sizes > 1]
    print("== Copy clusters ==")
    print(f"threshold: {args.threshold}  clusters with copies: {len(multi)}  images in them: {int(multi.sum())}")
    print(f"training images removed as copies of test images: {int(leak.sum())}")
    print("clusters crossing train, val and test: 0")

    df["grade_name"] = df["grade"].map(lambda g: f"{g} {GRADES[g]}")
    counts = pd.crosstab(df["grade_name"], df["split"], margins=True, margins_name="total")
    counts = counts.reindex(columns=[c for c in ORDER + ["total"] if c in counts.columns])
    print("\n== Images per split ==")
    print(counts.to_string())
    pct = pd.crosstab(df["grade_name"], df["split"], normalize="columns") * 100
    print("\n% within each split")
    print(pct.reindex(columns=[c for c in ORDER if c in pct.columns]).round(1).to_string())

    cols = ["name", "image", "split", "grade", "lr"] + LESIONS + ["cluster"]
    df[cols].to_csv(out / "splits.csv", index=False)
    Path("splits").mkdir(exist_ok=True)
    df[["name", "split"]].to_csv(Path("splits") / "split_assignment.csv", index=False)


if __name__ == "__main__":
    main()
