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
from omegaconf import OmegaConf

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

def load_configs(vqgan_ckpt, config_path=DEFAULT_CONFIG):
    """Load the inference config; point the autoencoder at the resolved VQGAN ckpt."""
    cfg = OmegaConf.load(str(config_path))
    cfg.autoencoder.ckpt_path = str(vqgan_ckpt)
    return cfg


def swin_align(cfg):
    """Pixel alignment the SwinUNet needs. The deepest level runs at latent / 2^(L-1)
    with a fixed window, so each tile's latent must be divisible by window·2^(L-1);
    ×sf for the pixel grid. A non-aligned tile crashes window_partition at the deep level."""
    n_levels = len(cfg.model.params.get("channel_mult", [1, 2, 2, 4]))
    return int(cfg.diffusion.params.sf) * int(cfg.model.params.window_size) * (2 ** (n_levels - 1))


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


def predict_x0(diff, model, z_t, z_y, t, eps=None):
    """ResShift x0 prediction (predict_type=xstart): scale input, forward, output IS x0."""
    return model(diff._scale_input(z_t, t), t, lq=z_y, eps=eps)


# ---------------------------------------------------------------------- builders

def _unet_params(cfg):
    return OmegaConf.to_container(cfg.model.params, resolve=True)


def build_student(cfg, device, dtype=torch.float32, noise_mode="concat", noise_channels=None):
    """Random-init StochasticUNet (the caller load_state_dict's the released ckpt over it)."""
    model = StochasticUNet(noise_mode=noise_mode, noise_channels=noise_channels, **_unet_params(cfg))
    return model.to(device, dtype).eval()


def build_autoencoder(cfg, device, dtype=torch.float32):
    ae = util_common.get_obj_from_str(cfg.autoencoder.target)(
        **OmegaConf.to_container(cfg.autoencoder.params, resolve=True)
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
        **OmegaConf.to_container(cfg.diffusion.params, resolve=True)
    )


def load_student_weights(student, state_dict):
    """Load an ema state_dict into the student (strict — arch metadata must match the ckpt)."""
    student.load_state_dict(_strip_module(state_dict), strict=True)
    return student


# --------------------------------------------------------------- ImageSpliterTh (pure torch)

class ImageSpliterTh:
    """Overlap-tiling helper (ResShift utils.util_image, inlined — pure torch, no cv2).

    Splits an (B,C,H,W) tensor into pch_size tiles on a `stride` grid, gathers the
    per-tile results with overlap averaging. sf=1 here (the student is same-resolution).
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
        for cur, (h0, h1, w0, w1) in zip(pch_list, index_infos):
            self.im_res[:, :, h0:h1, w0:w1] += cur
            self.pixel_count[:, :, h0:h1, w0:w1] += 1

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
def _sample_tile(cfg, diff, vqgan, student, hr_t, align, device, amp, generator):
    """One tile: reflect-pad to `align`, encode → 1-step student → decode → crop back.
    In/out (1,3,H,W) in [-1,1]. Heavy matmuls under bf16 autocast when `amp`; the VQGAN
    runs in its native (bf16 or fp32) dtype OUTSIDE autocast so GroupNorm activations
    stay in that dtype (autocast would force the full-res norms to fp32 = the VRAM peak)."""
    import torch.nn.functional as F
    h, w = hr_t.shape[2:]
    ph, pw = (-h) % align, (-w) % align
    if ph or pw:
        hr_t = F.pad(hr_t, (0, pw, 0, ph), mode="reflect")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    vdt = next(vqgan.parameters()).dtype
    z_y = vqgan.encode(hr_t.to(vdt)) * cfg.diffusion.params.scale_factor
    z_T = z_y + diff.kappa * torch.randn(z_y.shape, device=device, dtype=z_y.dtype, generator=generator)
    tT = torch.full((z_y.shape[0],), diff.num_timesteps - 1, device=device, dtype=torch.long)
    with ctx:
        z0 = predict_x0(diff, student, z_T, z_y, tT, make_eps(student, z_y, generator))
    img = vqgan.decode(z0.to(vdt), force_not_quantize=True).clamp(-1, 1)
    return img[:, :, :h, :w].float()


@torch.no_grad()
def upscale(cfg, diff, vqgan, student, lr_m1p1, device, scale, align,
            overlap, chop, tile_batch, amp=True, seed=None, progress=None):
    """1-step student ×`scale` SR. `lr_m1p1` is (1,3,H,W) in [-1,1] at LR resolution;
    it is bicubic-upsampled ×scale first (the student is spatially same-resolution),
    then tiled with ImageSpliterTh(sf=1). Returns (1,3,H*scale,W*scale) in [-1,1]."""
    import torch.nn.functional as F
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    up = F.interpolate(lr_m1p1, scale_factor=scale, mode="bicubic", align_corners=False)
    H, W = up.shape[2:]
    if H > chop or W > chop:
        spliter = ImageSpliterTh(up, chop, stride=chop - overlap, sf=1, extra_bs=tile_batch)
        for pch, info in spliter:
            spliter.update(_sample_tile(cfg, diff, vqgan, student, pch, align, device, amp, gen), info)
            if progress is not None:
                progress()
        img = spliter.gather()
    else:
        img = _sample_tile(cfg, diff, vqgan, student, up, align, device, amp, gen)
        if progress is not None:
            progress()
    return img.clamp(-1, 1)
