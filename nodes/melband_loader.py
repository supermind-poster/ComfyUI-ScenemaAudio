# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio MelBandRoFormer loader node for ComfyUI.

Loads the vocal/background separator used to strip ambient bleed from
generated speech (Generate's strip_background_sfx). Lives under
ComfyUI's shared models/checkpoints/ folder — dropdown reads the same
category ComfyUI's native checkpoint-picking nodes use, so pick the
MelBandRoFormer one out of whatever else is in there.

This is deliberately manual-only: nothing in this package auto-downloads
it (not even the prefetch convenience script, which only fetches the
voice-clone helper models). Download MelBandRoformer_fp16.safetensors
yourself (see README) and place it in ComfyUI/models/checkpoints/; the
dropdown below picks it up from there.

Connecting this loader is optional on Generate — if you don't, and
strip_background_sfx never actually triggers a strip, nothing needs it.
If it does trigger and this isn't connected, Generate raises a clear
error asking you to add this loader rather than silently loading
anything on its own.
"""

import logging

from .utils import get_checkpoint_path, list_checkpoint_files
from .vocal_separator import _load_melband_model

logger = logging.getLogger(__name__)


class ScenemaAudioMelBandLoader:
    """Loads MelBandRoFormer (vocal/background separator) from a local file.

    Dropdown reads ComfyUI/models/checkpoints/ — place
    MelBandRoformer_fp16.safetensors there by hand (see README). Not
    auto-downloaded by anything, including the prefetch script.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_SEPARATOR",)
    RETURN_NAMES = ("separator",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (list_checkpoint_files(), {
                    "tooltip": (
                        "MelBandRoformer_fp16.safetensors — place manually in "
                        "ComfyUI/models/checkpoints/. Used to strip ambient "
                        "bleed from generated speech."
                    ),
                }),
            },
        }

    def load(self, checkpoint):
        path = get_checkpoint_path(checkpoint)
        logger.info("Loading MelBandRoFormer from %s", path)
        model = _load_melband_model(path)
        return (model,)

