# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio VAE loader node for ComfyUI.

Loads the Audio VAE decoder and encoder from local checkpoints under
ComfyUI/models/vae/, the same folder the native "Load VAE" node reads
from. No network access — place both .safetensors files there yourself.
"""

import json
import logging

import torch
from ltx_pipelines.distilled import AudioDecoder
from safetensors import safe_open

from .utils import get_vae_path, list_vae_files, load_vae_encoder

logger = logging.getLogger(__name__)


class ScenemaAudioVAELoader:
    """Loads the Scenema Audio VAE (encoder + decoder) from local files.

    The decoder comes from the pipeline-audio checkpoint (~6.7 GB) and
    the encoder from the standalone VAE encoder checkpoint (~42.7 MB).
    Both dropdowns read ComfyUI/models/vae/, same as the native VAE
    loader — place the two files there.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_VAE",)
    RETURN_NAMES = ("vae",)

    @classmethod
    def INPUT_TYPES(cls):
        vae_files = list_vae_files()
        return {
            "required": {
                "decoder_file": (vae_files, {
                    "tooltip": (
                        "scenema-audio-pipeline-audio.safetensors — the audio "
                        "decoder checkpoint. Place in ComfyUI/models/vae/."
                    ),
                }),
                "encoder_file": (vae_files, {
                    "tooltip": (
                        "scenema-audio-vae-encoder.safetensors — the standalone "
                        "encoder used for voice-cloning reference audio. Place "
                        "in ComfyUI/models/vae/."
                    ),
                }),
            },
        }

    def load(self, decoder_file, encoder_file):
        pipeline_path = get_vae_path(decoder_file)
        encoder_path = get_vae_path(encoder_file)
        logger.info("Loading VAE decoder from %s", pipeline_path)
        logger.info("Loading VAE encoder from %s", encoder_path)

        # Load audio decoder directly (no Gemma dependency)
        audio_decoder = AudioDecoder(
            checkpoint_path=pipeline_path,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

        # Read config from decoder checkpoint metadata for VAE encoder
        with safe_open(pipeline_path, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        encoder, vae_sr = load_vae_encoder(config, encoder_path)

        return ({
            "decoder": audio_decoder,
            "encoder": encoder,
            "sample_rate": vae_sr,
        },)
