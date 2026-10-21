import argparse

import timm
import torch
from timm.layers import resample_abs_pos_embed

NUM_GRADES = 5
HEAD_KEYS = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}


def _resolve(path):
    # hf://owner/repo/file.pth points at a file on the Hugging Face hub
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        repo, filename = path[len("hf://"):].rsplit("/", 1)
        return hf_hub_download(repo, filename)
    return path


def _read_checkpoint(path):
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"safe loading failed ({type(e).__name__}), loading the official file with full unpickling")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_mae_encoder(model, path):
    ckpt = _read_checkpoint(_resolve(path))
    state = ckpt.get("model", ckpt)
    clean = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("backbone.", "")
        k = k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")
        # the MAE decoder and old head are not part of the encoder
        if k.startswith("decoder") or k == "mask_token" or k.startswith("head."):
            continue
        clean[k] = v
    del ckpt, state

    if "pos_embed" in clean and clean["pos_embed"].shape != model.pos_embed.shape:
        grid = model.patch_embed.grid_size
        clean["pos_embed"] = resample_abs_pos_embed(clean["pos_embed"], new_size=grid, num_prefix_tokens=1)

    msg = model.load_state_dict(clean, strict=False)
    missing, unexpected = set(msg.missing_keys), set(msg.unexpected_keys)
    # only the new head may be missing, and only the unused final norm may be left over
    if not missing <= HEAD_KEYS or not unexpected <= {"norm.weight", "norm.bias"}:
        raise RuntimeError(f"checkpoint does not match the model\nmissing: {sorted(missing)}\nunexpected: {sorted(unexpected)}")
    torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.zeros_(model.head.bias)
    print(f"loaded {len(clean) - len(unexpected)} pretrained tensors, new: {sorted(missing)}")


def build_model(cfg, load_pretrained=True):
    # load_pretrained=False builds the same network empty, for loading our own trained weights
    if cfg.get("pretrained_weights"):
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=NUM_GRADES, global_pool="avg",
                                  img_size=cfg["image_size"], drop_path_rate=cfg.get("drop_path", 0.0))
        if load_pretrained:
            load_mae_encoder(model, cfg["pretrained_weights"])
    else:
        model = timm.create_model(cfg["model"], pretrained=cfg["pretrained"] and load_pretrained,
                                  num_classes=NUM_GRADES, drop_path_rate=cfg.get("drop_path", 0.0))
    # optional, saves GPU memory; the lowest layers barely move under layer decay anyway
    n = cfg.get("freeze_blocks", 0)
    if n:
        frozen = [model.cls_token, model.pos_embed, *model.patch_embed.parameters(), *model.blocks[:n].parameters()]
        for p in frozen:
            p.requires_grad = False
    if cfg.get("grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model


def _vit_layer_id(name, num_layers):
    if name in ("cls_token", "pos_embed") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1


def build_optimizer(model, cfg, fused):
    if not cfg.get("layer_decay"):
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=fused)

    # layer decay: the head learns at the full rate, each layer below it at 0.65 times the one above
    num_layers = len(model.blocks)
    skip_decay = set(model.no_weight_decay())
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer = _vit_layer_id(name, num_layers)
        no_decay = p.ndim == 1 or name in skip_decay
        key = (layer, no_decay)
        if key not in groups:
            scale = cfg["layer_decay"] ** (num_layers + 1 - layer)
            groups[key] = {"params": [], "lr": cfg["lr"] * scale, "weight_decay": 0.0 if no_decay else cfg["weight_decay"]}
        groups[key]["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), lr=cfg["lr"], fused=fused)
