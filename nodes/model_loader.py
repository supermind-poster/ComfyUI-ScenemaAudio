# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio model loader node for ComfyUI.

Loads the 3.3B audio-only transformer checkpoint from a local file under
ComfyUI/models/diffusion_models/, the same folder and dropdown behavior
as the native "Load Diffusion Model" node. No network access — place the
.safetensors file there yourself before using this node.
"""

import logging

import comfy.model_management

from .utils import get_diffusion_model_path, list_diffusion_models, load_transformer

logger = logging.getLogger(__name__)


class ScenemaAudioModelLoader:
    """Loads the Scenema Audio transformer model from a local checkpoint.

    Reads the same models/diffusion_models folder ComfyUI's built-in
    "Load Diffusion Model" node uses. Supports both bf16
    (scenema-audio-transformer.safetensors) and INT8
    (scenema-audio-transformer-int8.safetensors) checkpoints — the
    format is auto-detected from the file's contents, so just pick the
    file you downloaded.
    Model stays on CPU until sampling to leave VRAM free for text encoding.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": (list_diffusion_models(), {
                    "tooltip": (
                        "Scenema Audio transformer checkpoint. Place "
                        "scenema-audio-transformer.safetensors (bf16) or "
                        "scenema-audio-transformer-int8.safetensors (INT8) in "
                        "ComfyUI/models/diffusion_models/."
                    ),
                }),
            },
        }

    def load(self, diffusion_model):
        path = get_diffusion_model_path(diffusion_model)
        logger.info("Loading transformer from %s", path)

        mdl_wrapper, config = load_transformer(path)

        # Keep on CPU — moved to GPU on demand during sampling
        device = comfy.model_management.get_torch_device()

        return ({"model": mdl_wrapper, "config": config, "device": device},)
