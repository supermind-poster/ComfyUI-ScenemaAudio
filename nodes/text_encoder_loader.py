# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoder loader node for ComfyUI.

Loads Gemma 3 12B IT + the pipeline checkpoint from local files only,
in the same "pick from a dropdown of what's in models/" style as
ComfyUI's native Load CLIP / Load Diffusion Model nodes. No network
access ever happens here — Gemma 3 12B IT must already exist locally as
a standard HF-layout folder (config.json + tokenizer + *.safetensors)
under ComfyUI/models/text_encoders/<name>/. See README for how to fetch
it once (a single manual step, since Google's repo is gated).
"""

import logging

from .text_encode import build_text_encoder
from .utils import (
    get_diffusion_model_path,
    get_text_encoder_path,
    list_diffusion_models,
    list_text_encoder_dirs,
)

logger = logging.getLogger(__name__)


class ScenemaAudioTextEncoderLoader:
    """Loads the Gemma-3-12B-IT-driven text encoder from local files.

    `gemma_dir` lists subfolders of ComfyUI/models/text_encoders/ that
    contain a config.json (i.e. a local HF-style model folder).
    `pipeline_file` lists ComfyUI/models/diffusion_models/ for
    scenema-audio-pipeline.safetensors, which carries the prompt-encoder
    config the pipeline needs alongside Gemma.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma_dir": (list_text_encoder_dirs(), {
                    "tooltip": (
                        "Local folder with Gemma 3 12B IT weights (config.json, "
                        "tokenizer files, *.safetensors shards). Place under "
                        "ComfyUI/models/text_encoders/<folder name>/."
                    ),
                }),
                "pipeline_file": (list_diffusion_models(), {
                    "tooltip": (
                        "scenema-audio-pipeline.safetensors — place in "
                        "ComfyUI/models/diffusion_models/."
                    ),
                }),
                "precision": (["auto", "nf4", "bf16_gpu", "cpu_streaming"], {
                    "default": "auto",
                    "tooltip": (
                        "auto picks based on VRAM: bf16_gpu (40GB+), nf4 (12-39GB), "
                        "cpu_streaming (<12GB). Override to force a specific path."
                    ),
                }),
            },
        }

    def load(self, gemma_dir, pipeline_file, precision="auto"):
        gemma_path = get_text_encoder_path(gemma_dir)
        pipeline_path = get_diffusion_model_path(pipeline_file)

        logger.info("Loading text encoder: gemma=%s pipeline=%s precision=%s",
                     gemma_path, pipeline_path, precision)

        entry = build_text_encoder(gemma_path, pipeline_path, precision)
        logger.info("Text encoder ready (mode=%s)", entry["mode"])

        return (entry,)
