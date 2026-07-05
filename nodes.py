"""ComfyUI nodes: distilled 1-step ResShift super-resolution (IMAGE → IMAGE), ×4 or ×2.

Two nodes:
  * ResShiftLoader   — load the RSD student + vq-f4 VQGAN -> RESSHIFT_MODEL socket
  * ResShiftUpscale  — RESSHIFT_MODEL + IMAGE -> IMAGE, super-resolved in ONE step

Unlike a latent/VAE-decode node, this is a pixel-space upscaler: it takes any RGB
image, bicubic-upsamples ×scale, runs one stochastic ResShift student step in vq-f4
latent space, and decodes scale× the input edge. The Loader's `scale` dropdown picks
×4 or ×2 (same arch + VQGAN — only the diffusion sf differs). Drop it after VAEDecode /
SaveImage — it is model-agnostic (works on any image, not just Anima output):

    ... -> VAEDecode -> IMAGE ─┐
                               ├─► ResShiftUpscale ─► IMAGE (×scale) -> SaveImage
    ResShiftLoader ───────┘

The student (~478 MB, ema-only safetensors) auto-downloads from the public
sorryhyun/Distilled-ResShift-4x (×4) or -2x (×2) HF repo per the selected scale; the
vq-f4 VQGAN (~211 MB) auto-downloads from the upstream ResShift v1.0 GitHub release.
Both land in ComfyUI/models/resshift/.
"""
import os
import shutil

import torch
import torch.nn as nn

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
from comfy.utils import ProgressBar

from . import resshift_infer as R

# Register a ComfyUI models/resshift folder for the student / VQGAN checkpoints.
_RSD_DIR = os.path.join(folder_paths.models_dir, "resshift")
os.makedirs(_RSD_DIR, exist_ok=True)
folder_paths.add_model_folder_path("resshift", _RSD_DIR)

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_CKPT_EXTS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")

# The distilled students (ema-only safetensors, ~478 MB each) on the public HF repos,
# one per scale. Both share the vq-f4 VQGAN and the StochasticUNet arch — the only
# difference is the diffusion sf (see resshift_infer.CONFIGS).
_STUDENTS = {
    "x4": {"repo": "sorryhyun/Distilled-ResShift-4x", "file": "rsd_student_12k.safetensors"},
    "x2": {"repo": "sorryhyun/Distilled-ResShift-2x", "file": "rsd_student_2k.safetensors"},
}
_STUDENT_FILES = {spec["file"] for spec in _STUDENTS.values()}
_SCALES = list(_STUDENTS)  # dropdown order: ["x4", "x2"]

# The vq-f4 autoencoder is a ResShift v1.0 GitHub release asset (NOT on HF Hub).
_VQGAN_URL = "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth"
_VQGAN_FILE = "autoencoder_vq_f4.pth"

# Stable dropdown sentinel for the student auto-download — ALWAYS present so a saved
# workflow that selected it stays valid across restarts. The `scale` input picks WHICH
# student the sentinel fetches (x4 or x2).
_AUTODL_ENTRY = "(auto-download)"


def _download_student(scale) -> str:
    """Fetch the ema-only student safetensors for `scale` into models/resshift/ (one-time)."""
    spec = _STUDENTS[scale]
    dest = os.path.join(_RSD_DIR, spec["file"])
    if os.path.exists(dest):
        return dest
    from huggingface_hub import hf_hub_download

    print(f"[ResShift] fetching {spec['repo']}/{spec['file']} -> {dest} (one-time, ~478 MB).")
    downloaded = hf_hub_download(repo_id=spec["repo"], filename=spec["file"])
    shutil.copyfile(downloaded, dest)
    return dest


def _ensure_vqgan() -> str:
    """Fetch the vq-f4 VQGAN from the ResShift v1.0 GitHub release (one-time)."""
    dest = os.path.join(_RSD_DIR, _VQGAN_FILE)
    if os.path.exists(dest):
        return dest
    print(f"[ResShift] fetching vq-f4 VQGAN {_VQGAN_URL} -> {dest} (one-time, ~211 MB).")
    try:
        torch.hub.download_url_to_file(_VQGAN_URL, dest, progress=True)
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not download the vq-f4 VQGAN ({e}). Download it manually from\n  {_VQGAN_URL}\n"
            f"and place it at {dest}."
        ) from e
    return dest


def _read_student(path):
    """Return (state_dict, noise_mode, noise_channels) from a student .safetensors or .pth.

    safetensors carries the arch metadata in its string header; a raw training .pth
    carries it as top-level keys alongside the 'ema' branch."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        sd = load_file(path)
        noise_mode = meta.get("noise_mode", "concat")
        nc = meta.get("noise_channels")
        return sd, noise_mode, (int(nc) if nc is not None else None)
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "ema" in blob:
        return blob["ema"], blob.get("noise_mode", "concat"), blob.get("noise_channels")
    return blob, "concat", None


class _RSDBundle(nn.Module):
    """Container holding the student + VQGAN as one module so a single ComfyUI
    ModelPatcher owns both nets' load/offload residency (per-param dtypes preserved)."""

    def __init__(self, student, vqgan):
        super().__init__()
        self.student = student
        self.vqgan = vqgan


class ResShiftModel:
    """RESSHIFT_MODEL socket: ModelPatcher-wrapped student+VQGAN bundle + inference config."""

    def __init__(self, patcher, cfg, diff, scale, align, amp):
        self.patcher = patcher
        self.cfg = cfg
        self.diff = diff
        self.scale = scale
        self.align = align
        self.amp = amp

    @property
    def student(self):
        return self.patcher.model.student

    @property
    def vqgan(self):
        return self.patcher.model.vqgan


class ResShiftLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = [f for f in folder_paths.get_filename_list("resshift")
                 if f.lower().endswith(_CKPT_EXTS) and os.path.basename(f) != _VQGAN_FILE]
        files.insert(0, _AUTODL_ENTRY)
        return {
            "required": {
                "scale": (_SCALES, {"default": "x4",
                          "tooltip": "Super-resolution factor. x4 (default) = the released ×4 student; "
                                     "x2 = the ×2 student (finer input, gentler jump). Picks both the "
                                     "auto-download repo AND the diffusion schedule (sf), so a locally "
                                     "selected student must match the scale it was distilled at."}),
                "student_name": (files,),
                "dtype": (["bf16", "fp32"], {"default": "bf16",
                          "tooltip": "bf16 (default) matches training/eval: ~2x faster, ~half VRAM. "
                                     "fp32 is slower + 2x VRAM, use only to rule out a precision issue."}),
            }
        }

    RETURN_TYPES = ("RESSHIFT_MODEL",)
    RETURN_NAMES = ("rsd_model",)
    FUNCTION = "load"
    CATEGORY = "ResShift"

    def load(self, scale, student_name, dtype):
        if student_name == _AUTODL_ENTRY or "(auto-download)" in student_name:
            student_path = _download_student(scale)
        else:
            student_path = folder_paths.get_full_path("resshift", student_name)
            # A saved workflow may store a known student's filename that isn't local yet
            # (e.g. after a cache wipe) — fetch the scale's student to keep it valid.
            if (student_path is None or not os.path.exists(student_path)) and student_name in _STUDENT_FILES:
                student_path = _download_student(scale)
        if student_path is None or not os.path.exists(student_path):
            raise FileNotFoundError(
                f"student checkpoint {student_name!r} not found under {_RSD_DIR}. "
                f"Select the auto-download entry, or place a student .safetensors/.pth there."
            )
        vqgan_path = _ensure_vqgan()

        dt = _DTYPES[dtype]
        amp = dt == torch.bfloat16
        offload = mm.unet_offload_device()
        load_device = mm.get_torch_device()

        cfg = R.load_configs(vqgan_path, config_path=R.CONFIGS[scale])
        diff = R.build_diffusion(cfg)
        scale = int(cfg.diffusion.params.sf)
        align = R.swin_align(cfg)

        sd, noise_mode, noise_channels = _read_student(student_path)
        # Student weights stay fp32 and run under bf16 autocast (matches infer.py); the
        # VQGAN goes native-bf16 so its full-res GroupNorm activations (the VRAM peak)
        # aren't forced back to fp32 by autocast. In fp32 mode both stay fp32.
        student = R.build_student(cfg, offload, dtype=torch.float32,
                                  noise_mode=noise_mode, noise_channels=noise_channels)
        R.load_student_weights(student, sd)
        vqgan = R.build_autoencoder(cfg, offload, dtype=dt)
        bundle = _RSDBundle(student, vqgan)

        patcher = comfy.model_patcher.ModelPatcher(bundle, load_device=load_device, offload_device=offload)
        print(f"[ResShift] loaded student ({noise_mode}/{student.noise_channels}ch) + vq-f4 "
              f"VQGAN as {dtype} (×{scale}, align={align}; ComfyUI-managed)")
        return (ResShiftModel(patcher, cfg, diff, scale, align, amp),)


class ResShiftUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rsd_model": ("RESSHIFT_MODEL",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Seeds the stochastic step (z_T = z_y + κ·randn and the "
                                            "injected ε). Same seed + input = reproducible output."}),
                "chop": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 256,
                                 "tooltip": "Tile size in px for large images (snapped to a multiple of the "
                                            "Swin align stride — 256 at ×4, 128 at ×2). Lower it if VRAM-bound; "
                                            "a whole image ≤ chop runs single-tile."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16,
                                    "tooltip": "Tile seam overlap in px. Tile stride = chop - overlap (must be "
                                               ">0). Larger = fewer seams, more redundant compute."}),
                "tile_batch": ("INT", {"default": 4, "min": 1, "max": 32,
                                       "tooltip": "Tiles stacked into one forward — small tiles underfill the "
                                                  "GPU one-at-a-time, so batching amortizes launch overhead. "
                                                  "VRAM scales ~linearly; tune to the card."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ResShift"

    def upscale(self, rsd_model, image, seed, chop, overlap, tile_batch):
        align = rsd_model.align
        # Swin needs each tile's dims % align == 0; snap chop up, clamp overlap so stride > 0.
        chop = max(align, (chop // align) * align)
        overlap = max(0, min(overlap, chop - align))

        mm.load_models_gpu([rsd_model.patcher])
        device = rsd_model.patcher.load_device
        student, vqgan, cfg, diff = rsd_model.student, rsd_model.vqgan, rsd_model.cfg, rsd_model.diff
        scale, amp = rsd_model.scale, rsd_model.amp

        # ComfyUI IMAGE: (B,H,W,C) in [0,1] -> (B,3,H,W) in [-1,1].
        x = image.permute(0, 3, 1, 2).contiguous().to(device) * 2.0 - 1.0
        B, _, H, W = x.shape

        ticks = 0
        for _ in range(B):
            import math
            n = R.count_tiles(H * scale, W * scale, chop, overlap)
            ticks += math.ceil(n / tile_batch)
        pbar = ProgressBar(ticks)

        outs = []
        for b in range(B):
            sr = R.upscale(cfg, diff, vqgan, student, x[b:b + 1], device, scale, align,
                           overlap=overlap, chop=chop, tile_batch=tile_batch, amp=amp,
                           seed=(int(seed) + b), progress=lambda: pbar.update(1))
            outs.append(sr)
        out = torch.cat(outs, dim=0)  # (B,3,H*s,W*s) in [-1,1]
        img = (out.clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).contiguous().float().cpu()
        return (img,)


NODE_CLASS_MAPPINGS = {
    "ResShiftLoader": ResShiftLoader,
    "ResShiftUpscale": ResShiftUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ResShiftLoader": "ResShift SR Loader (distilled ×4/×2)",
    "ResShiftUpscale": "ResShift SR Upscale (1-step)",
}
