"""Self-contained 1-step RSD student inference core (no anima_lora / ResShift-clone dep).

The student is a single-step stochastic generator distilled from a 15-step ResShift
teacher (paper 22490, "One-Step Residual Shifting Diffusion via Distillation"):

    ẑ0 = G_θ(z_T, z_y, T-1, ε),   z_T = z_y + κ·randn,   ε ~ N(0, I)

z_y is the vq-f4 latent of the bicubic-×4-upsampled LR image; the ×4 lives in the
residual-shift, not a spatial upscale, so the student runs same-resolution and the
decoded RGB is 4× the LR edge. Everything here is inference-only — the teacher / fake
critic / discriminator / grad-checkpointing from the training loop are dropped.

The ResShift arch (SwinUNet + vq-f4 VQGAN + residual-shift GaussianDiffusion) is
vendored under _vendor/resshift/ and already carries the query-chunked-SDPA patch for
the single-head head_dim=512 VQGAN mid-attention (so no xformers is needed on any GPU).
ImageSpliterTh (pure-torch overlap tiling) is inlined here to avoid the cv2/scipy/skimage
top-level imports of ResShift's utils.util_image.
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor" / "resshift"
# Put the _vendor dir on the path and import the vendored ResShift as a `resshift.*`
# package. It MUST be namespaced (not bare `utils`/`models`/`ldm`) — those bare names
# collide with ComfyUI's own top-level `utils` package (and a stray `models` namespace),
# which silently shadows the vendored modules at import time inside ComfyUI.
if str(VENDOR.parent) not in sys.path:
    sys.path.insert(0, str(VENDOR.parent))

from resshift.utils import util_common  # noqa: E402  (vendored ResShift)
from resshift.models.unet import UNetModelSwin  # noqa: E402  (vendored ResShift)

DEFAULT_CONFIG = VENDOR / "configs" / "realsr_swinunet_inference.yaml"
X2_CONFIG = VENDOR / "configs" / "realsr_swinunet_x2_inference.yaml"

# Inference config per scale tag. x4/x2 differ ONLY in diffusion.params.sf (4 vs 2),
# which drives the bicubic pre-upsample factor and the Swin alignment (swin_align); the
# UNet arch + vq-f4 VQGAN + residual-shift schedule are identical, so one student arch
# and one VQGAN serve both.
CONFIGS = {"x4": DEFAULT_CONFIG, "x2": X2_CONFIG}


# --------------------------------------------------------------------------- config

class _Cfg(dict):
    """Attribute-access dict — the only OmegaConf feature the static inference YAMLs use.

    The configs carry no `${}` interpolations and no resolvers, so PyYAML (already a
    ComfyUI core dep) + this wrapper replaces omegaconf outright, keeping the node
    dependency-free.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return _Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _plain(obj):
    """_Cfg tree -> plain dict/list (the OmegaConf.to_container equivalent)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def load_configs(vqgan_ckpt, config_path=DEFAULT_CONFIG):
    """Load the inference config; point the autoencoder at the resolved VQGAN ckpt."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = _wrap(yaml.safe_load(fh))
    cfg.autoencoder.ckpt_path = str(vqgan_ckpt)
    return cfg


def swin_align(cfg):
    """Pixel alignment the SwinUNet needs. Tiles are cut at MODEL-INPUT (post-upsample)
    resolution, so latent = pixels / f_vq and the deepest level runs at latent / 2^(L-1)
    with a hard window_partition view (no internal pad): each tile dim must be divisible
    by f_vq·window·2^(L-1) = 256, INDEPENDENT of sf. (The old ×sf formula was correct at
    ×4 only because sf == f_vq == 4; at ×2 it under-aligned to 128, so an edge tile
    reflect-padded to a 128-but-not-256 multiple crashed the deepest window_partition.)"""
    n_levels = len(cfg.model.params.get("channel_mult", [1, 2, 2, 4]))
    f_vq = 2 ** (len(cfg.autoencoder.params.ddconfig.ch_mult) - 1)
    return f_vq * int(cfg.model.params.window_size) * (2 ** (n_levels - 1))


# --------------------------------------------------------------- stochastic student

def _strip_module(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


class StochasticUNet(UNetModelSwin):
    """UNetModelSwin + a zero-init noise branch → stochastic one-step generator.

    Two injection modes (both zero-init ⇒ bit-identical to the frozen teacher at step 0):
      - "concat" (the released RSD student uses this): the first input conv is WIDENED by
        `noise_channels` (default 1) input planes, zero on the new slice, and ε is CONCAT'd
        onto the [latent, lq] stack before that conv.
      - "add": a separate zero-init conv maps ε to the post-conv width and is ADDED after
        input_blocks[0]. Kept for older "add"-trained checkpoints.

    forward(x, timesteps, lq, eps): injects ε per `noise_mode`. Inference-only — the
    training-side encode_features / grad-checkpointing were dropped.
    """

    def __init__(self, *args, noise_mode="concat", noise_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_mode = noise_mode
        self.noise_channels = noise_channels if noise_channels is not None \
            else (1 if noise_mode == "concat" else 3)
        conv = self.input_blocks[0][0]
        if noise_mode == "add":
            self.noise_proj = nn.Conv2d(self.noise_channels, conv.out_channels, 3, padding=1)
            nn.init.zeros_(self.noise_proj.weight)
            nn.init.zeros_(self.noise_proj.bias)
        elif noise_mode == "concat":
            wide = nn.Conv2d(conv.in_channels + self.noise_channels, conv.out_channels,
                             conv.kernel_size, padding=conv.padding)
            with torch.no_grad():
                nn.init.zeros_(wide.weight)
                wide.weight[:, :conv.in_channels].copy_(conv.weight)
                wide.bias.copy_(conv.bias)
            self.input_blocks[0][0] = wide
        else:
            raise ValueError(f"noise_mode must be 'add' or 'concat', got {noise_mode!r}")

    def _embed_and_concat(self, x, timesteps, lq):
        from resshift.models.unet import timestep_embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)
        if lq is not None:
            assert self.cond_lq
            lq = self.feature_extractor(lq.type(self.dtype))
            x = torch.cat([x, lq], dim=1)
        return x.type(self.dtype), emb

    def forward(self, x, timesteps, lq=None, eps=None, mask=None):
        h, emb = self._embed_and_concat(x, timesteps, lq)
        if self.noise_mode == "concat" and eps is not None:
            h = torch.cat([h, eps.type(self.dtype)], dim=1)
        hs = []
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if self.noise_mode == "add" and ii == 0 and eps is not None:
                h = h + self.noise_proj(eps.type(self.dtype))
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h.type(x.dtype))


def make_eps(model, ref, generator=None):
    """Injection noise shaped for `model`'s mode: (B, model.noise_channels, H, W), like `ref`.

    NOT the residual-shift diffusion noise (that stays latent-shaped, 3-ch)."""
    return torch.randn(ref.shape[0], model.noise_channels, *ref.shape[2:],
                       device=ref.device, dtype=ref.dtype, generator=generator)


def predict_x0(diff, model, z_t, c_y, t, eps=None):
    """ResShift x0 prediction (predict_type=xstart): scale input, forward, output IS x0.

    `c_y` is the `lq` CONDITIONING map from make_cond_lq — NOT the residual base z_y."""
    return model(diff._scale_input(z_t, t), t, lq=c_y, eps=eps)


def make_cond_lq(lq_img, z_y, mode="latent"):
    """Build the tensor fed to the UNet's `lq` conditioning input.

    `lq` is NOT the residual-shift base — that's `z_y`, the VQ latent of the upsampled
    LR, which keeps its own latent identity in the prior. `lq` is a separate 3-channel
    map concatenated onto the latent inside the first conv, and which convention a
    student expects depends on how it was distilled:

      "pixel"  — the LR PIXELS at latent resolution, in [-1,1]. The released realsr
                 teacher's convention (lq_size == image_size == 64 resolves the
                 feature_extractor to nn.Identity), so students distilled after
                 2026-07-11 inherit it and the released prior actually transfers.
      "latent" — the VQ latent itself. Off-manifold for pixel-trained weights (x0 MSE
                 degrades 0.004 -> 1.50 at t=14 and output comes out neon green), but
                 correct for the x4 12k student and the whole x2 line, which were both
                 distilled under it and are therefore self-consistent.

    The mode travels in the checkpoint metadata; default "latent" so pre-2026-07-11
    students that carry no `cond_lq` key keep their original behaviour.

    lq_img: the HR-sized bicubic-upsampled LR tile in [-1,1] (post pad-to-align).
    z_y:    the LR latent — only its spatial dims are used here.
    """
    if mode == "latent":
        return z_y
    if mode != "pixel":
        raise ValueError(f"cond_lq mode must be 'pixel' or 'latent', got {mode!r}")
    import torch.nn.functional as F
    return F.interpolate(
        lq_img.float(), size=z_y.shape[-2:], mode="bicubic", align_corners=False,
    ).clamp(-1, 1).to(z_y.dtype)


# ---------------------------------------------------------------------- builders

def _unet_params(cfg):
    return _plain(cfg.model.params)


def build_student(cfg, device, dtype=torch.float32, noise_mode="concat", noise_channels=None):
    """Random-init StochasticUNet (the caller load_state_dict's the released ckpt over it)."""
    model = StochasticUNet(noise_mode=noise_mode, noise_channels=noise_channels, **_unet_params(cfg))
    return model.to(device, dtype).eval()


def build_autoencoder(cfg, device, dtype=torch.float32):
    ae = util_common.get_obj_from_str(cfg.autoencoder.target)(
        **_plain(cfg.autoencoder.params)
    )
    sd = torch.load(cfg.autoencoder.ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    ae.load_state_dict(_strip_module(sd), strict=False)
    ae = ae.to(device, dtype).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def build_diffusion(cfg):
    """ResShift residual-shift GaussianDiffusion (used for _scale_input + num_timesteps)."""
    return util_common.get_obj_from_str(cfg.diffusion.target)(
        **_plain(cfg.diffusion.params)
    )


def load_student_weights(student, state_dict):
    """Load an ema state_dict into the student (strict — arch metadata must match the ckpt)."""
    student.load_state_dict(_strip_module(state_dict), strict=True)
    return student


# --------------------------------------------------------------- ImageSpliterTh (pure torch)

class ImageSpliterTh:
    """Overlap-tiling helper (ResShift utils.util_image, inlined — pure torch, no cv2).

    Splits an (B,C,H,W) tensor into pch_size tiles on a `stride` grid, gathers the
    per-tile results with overlap blending. sf=1 here (the student is same-resolution).

    Blending is FEATHERED, not the upstream box average: each tile's weight ramps 0→1
    over the overlap width from its own edge (Hann profile, separable). The box average
    steps the blend weight at the overlap edges (1 tile ↔ 2 tiles), so any per-tile
    disagreement (each tile's VQGAN encode + Swin sees different context — present even
    with the shared noise field) lands as a visible line exactly there; feathering fades
    it in smoothly instead. gather() renormalizes by accumulated weight, so borders
    (single tile at partial weight) come out exact; overlap == 0 keeps hard box tiles.
    """

    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        assert stride <= pch_size
        self.stride, self.pch_size, self.sf, self.extra_bs = stride, pch_size, sf, extra_bs
        bs, chn, height, width = im.shape
        self.true_bs = bs
        self.height_starts_list = self._starts(height)
        self.width_starts_list = self._starts(width)
        self.starts_list = [[ii, jj] for ii in self.height_starts_list for jj in self.width_starts_list]
        self.length = len(self.starts_list)
        self.count_pchs = 0
        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height * sf, width * sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros_like(self.im_res)
        self.ramp = (pch_size - stride) * sf   # overlap width at output res = feather span
        self._wcache = {}

    def _tile_weight(self, h, w, dtype, device):
        if self.ramp <= 0:
            return None
        key = (h, w, dtype, device)
        if key not in self._wcache:
            def prof(length):
                d = torch.arange(length, device=device, dtype=torch.float32) + 0.5
                d = torch.minimum(d, length - d).clamp(max=self.ramp) / self.ramp
                return 0.5 - 0.5 * torch.cos(torch.pi * d)
            w2d = (prof(h)[:, None] * prof(w)[None, :]).clamp(min=1e-3)
            self._wcache[key] = w2d.to(dtype)[None, None]
        return self._wcache[key]

    def _starts(self, length):
        if length <= self.pch_size:
            return [0]
        starts = list(range(0, length, self.stride))
        for ii in range(len(starts)):
            if starts[ii] + self.pch_size > length:
                starts[ii] = length - self.pch_size
        return sorted(set(starts), key=starts.index)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs >= self.length:
            raise StopIteration()
        index_infos = []
        current = self.starts_list[self.count_pchs:self.count_pchs + self.extra_bs]
        pch = None
        for ii, (h0, w0) in enumerate(current):
            cur = self.im_ori[:, :, h0:h0 + self.pch_size, w0:w0 + self.pch_size]
            pch = cur if ii == 0 else torch.cat([pch, cur], dim=0)
            index_infos.append([h0 * self.sf, (h0 + self.pch_size) * self.sf,
                                w0 * self.sf, (w0 + self.pch_size) * self.sf])
        self.count_pchs += len(current)
        return pch, index_infos

    def update(self, pch_res, index_infos):
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        wt = self._tile_weight(pch_res.shape[-2], pch_res.shape[-1],
                               pch_res.dtype, pch_res.device)
        for cur, (h0, h1, w0, w1) in zip(pch_list, index_infos):
            if wt is None:
                self.im_res[:, :, h0:h1, w0:w1] += cur
                self.pixel_count[:, :, h0:h1, w0:w1] += 1
            else:
                self.im_res[:, :, h0:h1, w0:w1] += cur * wt
                self.pixel_count[:, :, h0:h1, w0:w1] += wt

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)


def count_tiles(H, W, chop, overlap):
    """Number of tiles upscale() will produce for an H×W (post-upsample) image.
    Mirrors ImageSpliterTh._starts on both axes; the caller divides by tile_batch
    (one forward per batch) to get progress-bar ticks."""
    if H <= chop and W <= chop:
        return 1

    def _n(length):
        if length <= chop:
            return 1
        stride = chop - overlap
        starts = list(range(0, length, stride))
        for ii in range(len(starts)):
            if starts[ii] + chop > length:
                starts[ii] = length - chop
        return len(sorted(set(starts), key=starts.index))

    return _n(H) * _n(W)


# ------------------------------------------------------------------------- upscale

@torch.no_grad()
def _sample_tile(cfg, diff, vqgan, student, hr_t, align, device, amp, generator,
                 zT_noise=None, eps_noise=None, cond_mode="latent"):
    """One tile: reflect-pad to `align`, encode → 1-step student → decode → crop back.
    In/out (1,3,H,W) in [-1,1]. Heavy matmuls under bf16 autocast when `amp`; the VQGAN
    runs in its native (bf16 or fp32) dtype OUTSIDE autocast so GroupNorm activations
    stay in that dtype (autocast would force the full-res norms to fp32 = the VRAM peak).

    `zT_noise`/`eps_noise` (latent-shaped, matching z_y): a precomputed noise slice from
    the shared full-image field (see `upscale`), so overlapping tiles draw IDENTICAL noise
    and the overlap-average is seam-free. None = the original independent per-tile draw."""
    import torch.nn.functional as F
    h, w = hr_t.shape[2:]
    ph, pw = (-h) % align, (-w) % align
    if ph or pw:
        hr_t = F.pad(hr_t, (0, pw, 0, ph), mode="reflect")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    vdt = next(vqgan.parameters()).dtype
    z_y = vqgan.encode(hr_t.to(vdt)) * cfg.diffusion.params.scale_factor
    # Noise slices are chop//f-sized; a skinny tile (image dim ≤ chop on one axis) pads to
    # LESS than chop, so crop the slice to the actual latent dims before injecting.
    hz, wz = z_y.shape[2:]
    zt_n = (torch.randn(z_y.shape, device=device, dtype=z_y.dtype, generator=generator)
            if zT_noise is None else zT_noise[:, :, :hz, :wz].to(z_y.dtype))
    z_T = z_y + diff.kappa * zt_n
    eps = (make_eps(student, z_y, generator) if eps_noise is None
           else eps_noise[:, :, :hz, :wz].to(z_y.dtype))
    tT = torch.full((z_y.shape[0],), diff.num_timesteps - 1, device=device, dtype=torch.long)
    # `lq` conditioning is NOT the residual base z_y — the convention travels in the ckpt
    # meta (see make_cond_lq). hr_t is the padded tile, matching what z_y was encoded from.
    c_y = make_cond_lq(hr_t, z_y, cond_mode)
    with ctx:
        z0 = predict_x0(diff, student, z_T, c_y, tT, eps)
    img = vqgan.decode(z0.to(vdt), force_not_quantize=True).clamp(-1, 1)
    return img[:, :, :h, :w].float()


@torch.no_grad()
def upscale(cfg, diff, vqgan, student, lr_m1p1, device, scale, align,
            overlap, chop, tile_batch, amp=True, seed=None, progress=None, shared_noise=True,
            cond_mode="latent"):
    """1-step student ×`scale` SR. `lr_m1p1` is (1,3,H,W) in [-1,1] at LR resolution;
    it is bicubic-upsampled ×scale first (the student is spatially same-resolution),
    then tiled with ImageSpliterTh(sf=1). Returns (1,3,H*scale,W*scale) in [-1,1].

    `shared_noise` (default on): the student injects two independent noise draws per
    forward (3-ch residual-shift `z_T` + 1-ch `make_eps`). Drawn per tile, neighbouring
    tiles get DIFFERENT noise and the overlap-average leaves a faint tile-block seam in
    flat regions. Instead we draw ONE full-image latent noise field per source and slice
    each tile's noise from it by latent position, so overlapping tiles read IDENTICAL
    noise → seam-free. A few MB (latent-res), quality-neutral, marginally cheaper than
    per-tile. Only the tiled path uses it (a single un-tiled tile has no seam)."""
    import torch.nn.functional as F
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    up = F.interpolate(lr_m1p1, scale_factor=scale, mode="bicubic", align_corners=False)
    H, W = up.shape[2:]
    if H > chop or W > chop:
        spliter = ImageSpliterTh(up, chop, stride=chop - overlap, sf=1, extra_bs=tile_batch)
        g_zT = g_eps = None
        if shared_noise:
            ch_mult = cfg.autoencoder.params.ddconfig.ch_mult
            f = 2 ** (len(ch_mult) - 1)               # VQ-f4 -> 4
            z_ch = cfg.autoencoder.params.ddconfig.z_channels
            gpad = chop // f                          # margin so any tile start slices in-bounds
            hl, wl = H // f + gpad, W // f + gpad
            g_zT = torch.randn(1, z_ch, hl, wl, device=device, generator=gen)
            g_eps = torch.randn(1, student.noise_channels, hl, wl, device=device, generator=gen)
            lt = chop // f                            # per-tile latent size
        for pch, info in spliter:
            zt_noise = eps_noise = None
            if shared_noise:
                zt_noise = torch.cat([g_zT[:, :, h0 // f:h0 // f + lt, w0 // f:w0 // f + lt]
                                      for (h0, _, w0, _) in info], dim=0)
                eps_noise = torch.cat([g_eps[:, :, h0 // f:h0 // f + lt, w0 // f:w0 // f + lt]
                                       for (h0, _, w0, _) in info], dim=0)
            spliter.update(_sample_tile(cfg, diff, vqgan, student, pch, align, device, amp, gen,
                                        zT_noise=zt_noise, eps_noise=eps_noise,
                                        cond_mode=cond_mode), info)
            if progress is not None:
                progress()
        img = spliter.gather()
    else:
        img = _sample_tile(cfg, diff, vqgan, student, up, align, device, amp, gen,
                           cond_mode=cond_mode)
        if progress is not None:
            progress()
    return img.clamp(-1, 1)
