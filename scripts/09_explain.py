import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import apply_temperature  # noqa: E402
from src.data import FundusDataset, build_transforms, load_split  # noqa: E402
from src.explain import (GradCAM, blur_cells, freeze, occlusion_cells, overlay_image, plain_image,  # noqa: E402
                         randomise_last_block, retina_mask, tile_sheet, upsample)
from src.models import build_model  # noqa: E402

LESIONS = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
DEFINING_LESION = {1: "MA", 2: "IH", 3: "VB_IRMA", 4: "NV"}  # the label that on its own implies each grade
REFERABLE = 2


def mean_ci(values, rng, n_boot):
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/05_train_v2/selection.json")
    ap.add_argument("--test-dir", default="outputs/06_test")
    ap.add_argument("--reliability-dir", default="outputs/08_reliability")
    ap.add_argument("--out", default="outputs/09_explain")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-sanity", type=int, default=100)
    ap.add_argument("--n-faithful", type=int, default=200)
    ap.add_argument("--hide-share", type=float, default=0.10)
    ap.add_argument("--review-share", type=float, default=0.20, help="same review workload as in step 8")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    selection = Path(args.selection)
    run = json.loads(selection.read_text())["chosen_run"]
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # our own checkpoint, written by 05_train.py
    ckpt = torch.load(selection.parent / run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, load_pretrained=False)
    if not hasattr(model, "layer4"):
        raise SystemExit("Grad-CAM here expects a ResNet with a layer4 block")
    model.load_state_dict(ckpt["model"])
    model = freeze(model.to(device))
    cam_model = GradCAM(model, model.layer4)

    test = load_split(cfg["splits"], "test")
    data = FundusDataset(test, cfg["image_dir"], build_transforms(cfg, False))
    preds = pd.read_csv(Path(args.test_dir) / run / "test_predictions.csv").set_index("name").loc[test["name"]]
    target = preds["pred"].to_numpy()

    # review flag from step 8: probability of referable DR closest to 50%, cut-off set on validation
    t = json.loads((Path(args.reliability_dir) / run / "reliability.json").read_text())["calibration"]["temperature"]
    val = pd.read_csv(selection.parent / run / "val_predictions.csv")
    val_ref = apply_temperature(val[[f"p{k}" for k in range(5)]].to_numpy(), t)[:, REFERABLE:].sum(1)
    test_cal = apply_temperature(preds[[f"p{k}" for k in range(5)]].to_numpy(), t)
    cut = np.quantile(np.abs(val_ref - 0.5), args.review_share)
    info = test[["name", "grade"] + LESIONS].copy()
    info["pred"] = target
    info["confidence"] = test_cal.max(1).round(3)
    info["p_referable"] = test_cal[:, REFERABLE:].sum(1).round(3)
    info["sent_for_review"] = np.abs(test_cal[:, REFERABLE:].sum(1) - 0.5) <= cut

    # 1. Grad-CAM for every test photo, explaining the grade reported in step 6
    cams, outside, agree = [], [], 0
    for s in range(0, len(data), args.batch_size):
        idx = range(s, min(s + args.batch_size, len(data)))
        x = torch.stack([data[i][0] for i in idx]).to(device)
        tgt = torch.as_tensor(target[list(idx)], device=device)
        cam, logits = cam_model(x, tgt)
        agree += int((logits.argmax(1) == tgt).sum())
        for j in range(len(idx)):
            heat = upsample(cam[j], x.shape[-2:])
            outside.append(float(heat[~retina_mask(x[j])].sum() / heat.sum().clamp_min(1e-8)))
        cams.append(cam.cpu().numpy().astype(np.float16))
        if (s // args.batch_size) % 50 == 0:
            print(f"  heatmaps {min(s + args.batch_size, len(data))}/{len(data)}")
    cams = np.concatenate(cams)
    np.save(out / "test_cams.npy", cams)
    info["heat_outside_retina"] = np.round(outside, 4)
    outside = np.array(outside)
    print(f"\n== Heatmaps for {len(data)} test photos (Grad-CAM on the last block, {cams.shape[1]}x{cams.shape[2]} grid) ==")
    print(f"same grade as step 6 when recomputed in full precision: {agree} of {len(data)}")
    print(f"share of heatmap outside the retina: median {np.median(outside):.1%}, 95th percentile {np.percentile(outside, 95):.1%}, "
          f"photos above 10%: {(outside > 0.10).sum()}")

    # 2. sanity check: do the maps change when the last block is scrambled?
    rand_model = build_model(cfg, load_pretrained=False)
    rand_model.load_state_dict(ckpt["model"])
    rand_model = randomise_last_block(rand_model.to(device), args.seed)
    rand_cam = GradCAM(rand_model, rand_model.layer4)
    picks = rng.choice(len(data), size=min(args.n_sanity, len(data)), replace=False)
    corrs, sanity_imgs = [], []
    for k, i in enumerate(picks):
        x = data[i][0][None].to(device)
        tgt = torch.as_tensor([target[i]], device=device)
        r_cam, _ = rand_cam(x, tgt)
        a, b = cams[i].astype(np.float32).ravel(), r_cam[0].cpu().numpy().ravel()
        if a.std() > 0 and b.std() > 0:
            corrs.append(spearmanr(a, b).statistic)
        if k < 4:
            sanity_imgs += [plain_image(x[0]), overlay_image(x[0], torch.from_numpy(cams[i].astype(np.float32)).to(device)),
                            overlay_image(x[0], r_cam[0])]
    tile_sheet(sanity_imgs, out / "sanity_check.jpg", per_row=3)
    info.iloc[picks[:4]].to_csv(out / "sanity_check.csv", index=False)
    corrs = np.array(corrs)
    print("\n== Sanity check: heatmaps from the trained model against a model with its last block scrambled ==")
    print(f"photos compared: {len(corrs)} of {len(picks)} (the rest gave a blank map)")
    print(f"rank correlation: median {np.median(corrs):.3f}, middle half {np.percentile(corrs, 25):.3f} to {np.percentile(corrs, 75):.3f}")
    print("near 0 means the maps depend on what the model learned; near 1 would mean they do not")

    # 3. faithfulness: hide the most highlighted 10% of the retina, or a random 10%, and watch the confidence
    pool = np.nonzero(target >= REFERABLE)[0]
    picks = rng.choice(pool, size=min(args.n_faithful, len(pool)), replace=False)
    drop_top, drop_rand, flip_top, flip_rand = [], [], [], []
    with torch.no_grad():
        for i in picks:
            x = data[i][0].to(device)
            cam = torch.from_numpy(cams[i].astype(np.float32)).to(device)
            mask = retina_mask(x)
            c = int(target[i])
            hidden = [blur_cells(x, occlusion_cells(cam, mask, args.hide_share))]
            hidden += [blur_cells(x, occlusion_cells(cam, mask, args.hide_share, rng)) for _ in range(3)]
            probs = torch.softmax(model(torch.stack([x] + hidden)), 1).cpu().numpy()
            drop_top.append(probs[0, c] - probs[1, c])
            drop_rand.append(np.mean(probs[0, c] - probs[2:, c]))
            flip_top.append(probs[1].argmax() != c)
            flip_rand.append(np.mean(probs[2:].argmax(1) != c))
    top, rnd = mean_ci(drop_top, rng, args.n_boot), mean_ci(drop_rand, rng, args.n_boot)
    diff = mean_ci(np.array(drop_top) - np.array(drop_rand), rng, args.n_boot)
    print(f"\n== Faithfulness: {len(picks)} photos predicted referable, {args.hide_share:.0%} of the retina hidden by blurring ==")
    print(f"drop in confidence for the predicted grade, hiding the most highlighted area: {top[0]:.3f} [{top[1]:.3f}, {top[2]:.3f}]")
    print(f"drop in confidence for the predicted grade, hiding a random area (3 tries each): {rnd[0]:.3f} [{rnd[1]:.3f}, {rnd[2]:.3f}]")
    print(f"difference: {diff[0]:.3f} [{diff[1]:.3f}, {diff[2]:.3f}]")
    print(f"predicted grade changed: {np.mean(flip_top):.1%} when hiding the highlighted area, {np.mean(flip_rand):.1%} when hiding a random area")

    # 4. example figures, each photo next to its heatmap, with a csv giving the order
    def pairs(rows, name):
        if rows.empty:
            return
        images = []
        for i in rows.index:
            x = data[i][0].to(device)
            images += [plain_image(x), overlay_image(x, torch.from_numpy(cams[i].astype(np.float32)).to(device))]
        tile_sheet(images, out / f"{name}.jpg", per_row=4)
        rows.to_csv(out / f"{name}.csv", index=False)

    correct = []
    for g in range(5):
        ok = info[(info["grade"] == g) & (info["pred"] == g)]
        if g in DEFINING_LESION:
            ok = ok[ok[DEFINING_LESION[g]] == 1]
        correct.append(ok.sample(min(2, len(ok)), random_state=args.seed))
    pairs(pd.concat(correct), "examples_correct")
    serious = info[(info["grade"] >= 3) & (info["pred"] < REFERABLE)].sort_values(["grade", "pred"], ascending=[False, True])
    if len(serious):
        pairs(serious, "examples_serious_errors")
    print("\n== The serious underestimates, with what a clinician would see ==")
    print(serious[["name", "grade", "pred", "confidence", "p_referable", "sent_for_review", "heat_outside_retina"]].to_string(index=False)
          if len(serious) else "none")

    info.to_csv(out / "test_explanations.csv", index=False)
    summary = {"photos": len(data), "agree_with_step6": agree,
               "outside_retina_median": round(float(np.median(outside)), 4), "outside_retina_over_10pct": int((outside > 0.10).sum()),
               "sanity_spearman_median": round(float(np.median(corrs)), 4), "sanity_n": int(len(corrs)),
               "faithful_drop_top": [round(v, 4) for v in top], "faithful_drop_random": [round(v, 4) for v in rnd],
               "faithful_difference": [round(v, 4) for v in diff],
               "flip_top": round(float(np.mean(flip_top)), 4), "flip_random": round(float(np.mean(flip_rand)), 4)}
    (out / "explain_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
