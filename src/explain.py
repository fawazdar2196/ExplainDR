import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from src.data import MEAN, STD

OVERLAY = np.array([0, 200, 255], dtype=np.float32)  # cyan stands out against the orange retina


class GradCAM:
    # Grad-CAM (Selvaraju et al., 2017) on one layer of a CNN
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        # only when gradients are being tracked; plain predictions pass straight through
        if output.requires_grad:
            self.acts = output
            output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        x = x.clone().requires_grad_(True)  # gradients flow to the activations without touching the weights
        with torch.enable_grad():
            logits = self.model(x)
            if target is None:
                target = logits.argmax(1)
            logits.gather(1, target[:, None]).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(1)).detach()
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam, logits.detach()


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def randomise_last_block(model, seed):
    # the sanity check of Adebayo et al. (2018): reset the classifier and the last block to random weights
    torch.manual_seed(seed)
    for module in [model.fc, *model.layer4.modules()]:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return freeze(model)


def unnormalise(x):
    mean = torch.tensor(MEAN, device=x.device)[:, None, None]
    std = torch.tensor(STD, device=x.device)[:, None, None]
    return (x * std + mean).clamp(0, 1)


def retina_mask(x, threshold=15 / 255):
    # same rule as preprocessing: anything brighter than near black is retina
    return unnormalise(x).mean(0) > threshold


def occlusion_cells(cam, mask, share, rng=None):
    # cells are the heatmap's own grid; a cell counts as retina if at least half of it is retina
    h, w = cam.shape
    cell = mask.shape[0] // h
    retina = mask[: h * cell, : w * cell].reshape(h, cell, w, cell).float().mean(dim=(1, 3)) >= 0.5
    candidates = torch.nonzero(retina.flatten()).squeeze(1)
    k = max(1, int(round(share * len(candidates))))
    if rng is None:
        order = torch.argsort(cam.flatten()[candidates], descending=True)
        chosen = candidates[order[:k]]
    else:
        chosen = candidates[torch.from_numpy(rng.choice(len(candidates), size=k, replace=False))]
    grid = torch.zeros(h * w, dtype=torch.bool, device=cam.device)
    grid[chosen] = True
    return grid.reshape(h, w).repeat_interleave(cell, 0).repeat_interleave(cell, 1)


def blur_cells(x, cells, sigma=16):
    # hidden areas are replaced by a heavily blurred copy, which keeps the colour but removes lesions
    kernel = int(6 * sigma) | 1
    blurred = gaussian_blur(x, kernel_size=kernel, sigma=sigma)
    full = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
    full[: cells.shape[0], : cells.shape[1]] = cells
    return torch.where(full[None], blurred, x)


def upsample(cam, size):
    return F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]


def overlay_image(x, cam, max_alpha=0.6):
    img = unnormalise(x).permute(1, 2, 0).cpu().numpy()
    heat = upsample(cam, x.shape[-2:]).cpu().numpy()[..., None]
    heat = heat * retina_mask(x).cpu().numpy()[..., None]
    alpha = max_alpha * heat
    out = (1 - alpha) * img * 255 + alpha * OVERLAY
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def plain_image(x):
    return Image.fromarray((unnormalise(x).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def tile_sheet(images, out_file, per_row, tile=320, gap=6):
    rows = (len(images) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * tile + (per_row - 1) * gap, rows * tile + (rows - 1) * gap), "white")
    for k, im in enumerate(images):
        r, c = divmod(k, per_row)
        sheet.paste(im.resize((tile, tile)), (c * (tile + gap), r * (tile + gap)))
    sheet.save(out_file, quality=92)
