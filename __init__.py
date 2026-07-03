"""Distilled ResShift ×4 super-resolution ComfyUI custom nodes.

A 1-step ResShift student (distilled from a 15-step teacher, paper 22490) as a
pixel-space IMAGE -> IMAGE ×4 upscaler. Drop it after VAEDecode / SaveImage — it is
model-agnostic (works on any RGB image, not just Anima output).

* ``ResShiftLoader``  — load the RSD student + vq-f4 VQGAN -> ``RESSHIFT_MODEL`` socket.
* ``ResShiftUpscale`` — ``RESSHIFT_MODEL`` + ``IMAGE`` -> ``IMAGE`` (×4), one step.

The student (~478 MB ema-only safetensors) auto-downloads from the public
``sorryhyun/Distilled-ResShift-4x`` HF repo; the vq-f4 VQGAN (~211 MB) auto-downloads
from the upstream ResShift v1.0 GitHub release. Both land in ``ComfyUI/models/resshift/``.

The ResShift network is vendored self-contained under ``_vendor/resshift/`` (S-Lab
license, from zsyOAOA/ResShift), already carrying the query-chunked-SDPA patch that
lets the single-head VQGAN mid-attention run without xformers on any GPU.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
