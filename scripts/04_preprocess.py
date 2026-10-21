import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

DARK = 15  # pixels at or below this are treated as black border


def retina_box(im):
    # find the retina on a small copy, then scale the box back up
    small = im.copy()
    small.thumbnail((512, 512))
    sx, sy = im.width / small.width, im.height / small.height
    gray = np.asarray(small.convert("L").filter(ImageFilter.GaussianBlur(2)), np.float32)
    mask = gray > DARK
    # ignore stray bright specks or text in the border
    rows = np.nonzero(mask.mean(axis=1) > 0.02)[0]
    cols = np.nonzero(mask.mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (int(cols[0] * sx), int(rows[0] * sy), int(np.ceil((cols[-1] + 1) * sx)), int(np.ceil((rows[-1] + 1) * sy)))


def square_pad(im):
    s = max(im.size)
    canvas = Image.new("RGB", (s, s))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return canvas


def quality(im):
    gray = np.asarray(im.convert("L"), np.float32)
    green = np.asarray(im.getchannel("G"), np.float32)
    mask = gray > DARK
    # shrink the mask so the retina's own edge doesn't count as sharpness
    inner = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(15))) > 0
    lap = green[1:-1, :-2] + green[1:-1, 2:] + green[:-2, 1:-1] + green[2:, 1:-1] - 4 * green[1:-1, 1:-1]
    inner = inner[1:-1, 1:-1]
    if not inner.any():
        return float(mask.mean()), np.nan, np.nan, np.nan
    return float(mask.mean()), float(gray[mask].mean()), float(gray[mask].std()), float(lap[inner].var())


def process(task):
    src, dst, size = task
    with Image.open(src) as im:
        im = im.convert("RGB")
    box = retina_box(im)
    if box is not None:
        im = im.crop(box)
    crop_w, crop_h = im.size
    im = square_pad(im).resize((size, size), Image.LANCZOS)
    im.save(dst, quality=95)
    return (Path(dst).stem, box is None, crop_w, crop_h) + quality(im)


def contact_sheet(paths, out_file, size=160, per_row=8):
    rows = (len(paths) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * size, rows * size), "white")
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            sheet.paste(im.resize((size, size)), ((k % per_row) * size, (k // per_row) * size))
    sheet.save(out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MMRDR-CFP")
    ap.add_argument("--splits", default="outputs/03_splits/splits.csv")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--images-out", default="data/processed_768")
    ap.add_argument("--out", default="outputs/04_preprocess")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, images_out, out = Path(args.root), Path(args.images_out), Path(args.out)
    images_out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(args.splits)
    use = splits[splits["split"] != "excluded"].reset_index(drop=True)
    tasks = [(root / img, images_out / f"{name}.jpg", args.size) for name, img in zip(use["name"], use["image"])]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(process, tasks, chunksize=16), 1):
            rows.append(row)
            if k % 1000 == 0 or k == len(tasks):
                print(f"  processed {k}/{len(tasks)}")

    cols = ["name", "no_retina_found", "crop_w", "crop_h", "fov_fraction", "brightness", "contrast", "sharpness"]
    q = pd.DataFrame(rows, columns=cols).merge(use[["name", "split", "grade"]], on="name")
    q.to_csv(out / "quality.csv", index=False)

    ratio = q["crop_h"] / q["crop_w"]
    print("\n== Cropping ==")
    print(f"images: {len(q)}  no retina found (kept whole): {int(q['no_retina_found'].sum())}")
    print(f"retina cut off at top or bottom (height under 95% of width): {int((ratio < 0.95).sum())}")
    print(f"retina share of the final square, median: {q['fov_fraction'].median():.1%}")

    print("\n== Image quality, all images ==")
    print(q[["brightness", "contrast", "sharpness"]].describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_string())
    print("\n== Median quality by grade ==")
    print(q.groupby("grade")[["brightness", "contrast", "sharpness"]].median().round(1).to_string())

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(q), size=min(32, len(q)), replace=False)
    contact_sheet([images_out / f"{q['name'][k]}.jpg" for k in picks], out / "processed_samples.png")
    q.iloc[picks][["name", "split", "grade"]].to_csv(out / "processed_samples.csv", index=False)


if __name__ == "__main__":
    main()
