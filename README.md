# ComfyUI — Distilled ResShift SR (×4, 1-step)

A **1-step ResShift** super-resolution student as a pixel-space **`IMAGE → IMAGE` ×4**
upscaler for ComfyUI. The student is distilled (paper [2603.22490](https://arxiv.org/abs/2603.22490),
*One-Step Residual Shifting Diffusion via Distillation*) from a 15-step ResShift teacher,
so it does the whole super-resolution jump in a **single stochastic network step** —
no multi-step sampler, no CFG.

It is **model-agnostic**: drop it after `VAEDecode` / before `SaveImage` and it upscales
any RGB image, not just Anima / diffusion output.

```
… → VAEDecode → IMAGE ─┐
                       ├─► ResShiftUpscale ─► IMAGE (4×) → SaveImage
ResShiftLoader ───┘
```

## Nodes

| Node | In → Out | Notes |
|---|---|---|
| **ResShift SR Loader (distilled ×4)** | → `RESSHIFT_MODEL` | Loads the student + vq-f4 VQGAN. Pick `(auto-download)` to fetch the released student, or select a local checkpoint from `ComfyUI/models/resshift/`. `dtype`: bf16 (default, ~2× faster / ~half VRAM) or fp32. |
| **ResShift SR Upscale (1-step ×4)** | `RESSHIFT_MODEL` + `IMAGE` → `IMAGE` | `seed` (the step is stochastic), `chop` / `overlap` / `tile_batch` for tiling large images. |

## Weights (auto-downloaded on first use)

Both land in `ComfyUI/models/resshift/`:

- **Student** (~478 MB, ema-only `.safetensors`) — from the public HF repo
  [`sorryhyun/Distilled-ResShift-4x`](https://huggingface.co/sorryhyun/Distilled-ResShift-4x).
  Arch metadata (`noise_mode` / `noise_channels`) is carried in the safetensors header,
  so the loader rebuilds the network from the file alone.
- **vq-f4 VQGAN** (~211 MB) — from the upstream ResShift v1.0 GitHub release
  (`zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth`). If that download is
  blocked in your environment, place the file in `models/resshift/` by hand.

To use your own student, drop a `.safetensors` (or a training `.pth` with an `ema` key)
into `models/resshift/` and select it in the loader.

## How it works

The student is spatially **same-resolution** — the ×4 lives in the residual-shift, not a
spatial upscale. Each image is bicubic-upsampled ×4, encoded to the vq-f4 latent, run
through one student step (`z_T = z_y + κ·randn`, plus injected noise ε), and decoded. Large
images are tiled (`ImageSpliterTh`, overlap-averaged); each tile is reflect-padded to a
multiple of the Swin alignment (256 px) — `chop` is snapped to that automatically.

`bf16` runs the student under bf16 autocast and the VQGAN in native bf16 (keeping the
full-resolution GroupNorm activations — the VRAM peak — out of fp32).

## Dependencies

`torch` / `numpy` / `safetensors` ship with ComfyUI. The vendored ResShift network also
needs `einops`, `timm`, `omegaconf`, and `huggingface_hub` (auto-installed from
`pyproject.toml`). **No xformers** is required — the vendored VQGAN mid-attention uses a
query-chunked exact-SDPA path that runs on any GPU (including Blackwell / sm_120).

## Licensing

Node code: see `LICENSE`. The vendored ResShift network under `_vendor/resshift/` is
**S-Lab License 1.0 (non-commercial)**, from [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift).
The distilled student weights inherit the same non-commercial terms.
