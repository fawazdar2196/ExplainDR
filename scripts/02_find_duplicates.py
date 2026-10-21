import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

BANDS = [(0.99, 1.01), (0.95, 0.99), (0.90, 0.95), (0.80, 0.90), (0.60, 0.80), (-1.01, 0.60)]


def centre_crop(im, frac=0.7):
    # keep the middle of the retina so the black border doesn't drive the match
    w, h = im.size
    s = int(min(w, h) * frac)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def fingerprint(path, size=64):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.draft("RGB", (256, 256))  # decode at reduced size, much faster
        green = centre_crop(im.convert("RGB")).getchannel("G").resize((size, size), Image.BILINEAR)
    # subtract a blurred copy so vessels and lesions drive the match, not overall brightness
    v = np.asarray(green, np.float32) - np.asarray(green.filter(ImageFilter.GaussianBlur(3)), np.float32)
    v = v.ravel()
    v -= v.mean()
    v /= np.linalg.norm(v) + 1e-8
    return hashlib.md5(data).hexdigest(), v


def best_match(query, ref, same_set=False, block=1024):
    best = np.empty(len(query), np.float32)
    arg = np.empty(len(query), np.int64)
    for s in range(0, len(query), block):
        sims = query[s:s + block] @ ref.T
        if same_set:
            k = np.arange(s, s + len(sims))
            sims[k - s, k] = -2.0  # ignore each image matching itself
        arg[s:s + block] = sims.argmax(1)
        best[s:s + block] = sims.max(1)
    return best, arg


def pairs_above(vecs, threshold, block=1024):
    found = []
    for s in range(0, len(vecs), block):
        sims = vecs[s:s + block] @ vecs.T
        i, j = np.nonzero(sims >= threshold)
        keep = j > i + s
        found.append(pd.DataFrame({"i": i[keep] + s, "j": j[keep], "corr": sims[i[keep], j[keep]]}))
    return pd.concat(found, ignore_index=True)


def band_label(lo, hi):
    if hi > 1:
        return f">= {lo:.2f}"
    if lo < -1:
        return f"< {hi:.2f}"
    return f"{lo:.2f} to {hi:.2f}"


def band_counts(values):
    rows = []
    for lo, hi in BANDS:
        n = int(((values >= lo) & (values < hi)).sum())
        rows.append((band_label(lo, hi), n, f"{n / len(values):.1%}"))
    return pd.DataFrame(rows, columns=["max correlation", "images", "share"])


def contact_sheet(rows, out_file, size=180, per_band=4):
    # one row per band, each cell is a test image next to its closest train image
    sheet = Image.new("RGB", (per_band * 2 * size, len(BANDS) * size), "white")
    for _, r in rows.iterrows():
        for k, p in enumerate((r["test_path"], r["train_path"])):
            with Image.open(p) as im:
                im.draft("RGB", (size, size))
                tile = im.convert("RGB").resize((size, size))
            sheet.paste(tile, ((r["slot"] * 2 + k) * size, r["band"] * size))
    sheet.save(out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MMRDR-CFP")
    ap.add_argument("--out", default="outputs/02_duplicates")
    ap.add_argument("--pair-threshold", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "FP.csv")
    df["name"] = df["image"].map(lambda s: Path(s).stem)
    df["split"] = df["name"].str[:2].map({"tr": "train", "ts": "test"})

    md5s, vecs = [], []
    for k, rel in enumerate(df["image"], 1):
        m, v = fingerprint(root / rel)
        md5s.append(m)
        vecs.append(v)
        if k % 1000 == 0 or k == len(df):
            print(f"  fingerprinted {k}/{len(df)}")
    df["md5"] = md5s
    vecs = np.stack(vecs)
    np.save(out / "fingerprints.npy", vecs)

    tr = np.nonzero(df["split"].eq("train").to_numpy())[0]
    ts = np.nonzero(df["split"].eq("test").to_numpy())[0]

    best, arg = best_match(vecs[ts], vecs[tr])
    nn = pd.DataFrame({"i": ts, "j": tr[arg], "corr": best})
    nn["same_file"] = df["md5"].to_numpy()[nn["i"]] == df["md5"].to_numpy()[nn["j"]]
    nn["same_grade"] = df["grade"].to_numpy()[nn["i"]] == df["grade"].to_numpy()[nn["j"]]

    print("\n== Test images by correlation with their closest train image ==")
    table = band_counts(nn["corr"].to_numpy())
    table["exact file copies"] = [int(nn.loc[(nn["corr"] >= lo) & (nn["corr"] < hi), "same_file"].sum()) for lo, hi in BANDS]
    table["same grade"] = [f"{nn.loc[(nn['corr'] >= lo) & (nn['corr'] < hi), 'same_grade'].mean():.1%}"
                           if ((nn["corr"] >= lo) & (nn["corr"] < hi)).any() else "-" for lo, hi in BANDS]
    print(table.to_string(index=False))

    best_tr, _ = best_match(vecs[tr], vecs[tr], same_set=True)
    print("\n== Train images by correlation with their closest other train image ==")
    print(band_counts(best_tr).to_string(index=False))

    rng = np.random.default_rng(args.seed)
    picks = []
    for b, (lo, hi) in enumerate(BANDS):
        pool = nn.index[(nn["corr"] >= lo) & (nn["corr"] < hi)].to_numpy()
        chosen = np.sort(rng.choice(pool, size=min(4, len(pool)), replace=False)) if len(pool) else []
        for slot, k in enumerate(chosen):
            picks.append((b, slot, k))
    rows = pd.DataFrame(picks, columns=["band", "slot", "k"])
    rows = rows.join(nn, on="k")
    rows["test"] = df["name"].to_numpy()[rows["i"]]
    rows["train"] = df["name"].to_numpy()[rows["j"]]
    rows["test_grade"] = df["grade"].to_numpy()[rows["i"]]
    rows["train_grade"] = df["grade"].to_numpy()[rows["j"]]
    rows["test_path"] = [root / df["image"][i] for i in rows["i"]]
    rows["train_path"] = [root / df["image"][j] for j in rows["j"]]
    contact_sheet(rows, out / "pairs_by_band.png")
    shown = rows[["band", "slot", "test", "train", "test_grade", "train_grade", "corr", "same_file"]].copy()
    shown["band"] = [band_label(*BANDS[b]) for b in shown["band"]]
    shown["corr"] = shown["corr"].round(3)
    shown.to_csv(out / "pairs_by_band.csv", index=False)
    print("\n== Sample pairs in pairs_by_band.png (one row per band) ==")
    print(shown.to_string(index=False))

    pairs = pairs_above(vecs, args.pair_threshold)
    for col in ("name", "split", "grade", "md5"):
        pairs[f"{col}_a"] = df[col].to_numpy()[pairs["i"]]
        pairs[f"{col}_b"] = df[col].to_numpy()[pairs["j"]]
    pairs.drop(columns=["i", "j"]).to_csv(out / "candidate_pairs.csv", index=False)
    print(f"\n== All pairs with correlation >= {args.pair_threshold} ==")
    if len(pairs):
        kind = pairs[["split_a", "split_b"]].apply(lambda r: "-".join(sorted(r)), axis=1)
        summary = pd.DataFrame({
            "pairs": kind.value_counts(),
            "different_grade": kind[pairs["grade_a"] != pairs["grade_b"]].value_counts(),
        }).fillna(0).astype(int)
        print(summary.to_string())
    else:
        print("  none")
    df[["name", "split", "grade", "md5"]].to_csv(out / "fingerprints.csv", index=False)


if __name__ == "__main__":
    main()
