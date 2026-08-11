# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Generate (XML) — raw <speak> prompt mode.

Takes the exact XML prompt format the base LTX 2.3 audio model expects,
with no friendly widget layer in between. Use this if you already have
prompts written in Scenema Audio's native format (from their docs, API,
or generated programmatically) and want byte-for-byte the same input the
production model receives.

See audio_core/compiler.py for the full <speak> schema reference:

    <speak voice="..." scene="..." gender="male|female"
           language="en" shot="closeup|wide|scene">
      <action>delivery/performance cue</action>
      <sound>audio event, e.g. SFX or ambience</sound>
      Speech text goes here as plain text between tags.
    </speak>

`voice` is required. `scene`, `language` (default "en"), and `shot`
(default "closeup") are optional. `<action>` and `<sound>` tags and
plain text can be freely interleaved in document order — each becomes
its own line/cue in the compiled prompt, in the order it appears.
"""

import logging

import torch

from .generate_core import run_generation

logger = logging.getLogger(__name__)

EXAMPLE_XML = """<speak voice="Male, late 60s. Deep, gravelly. Slow and deliberate." scene="Absolute silence" gender="male">
  <action>He takes a slow breath</action>
  Look again at that dot. That's here. That's home. That's us.
</speak>"""


class ScenemaAudioGenerateXML:
    """Generate speech directly from a raw <speak> XML prompt — the base
    model's native format, with no widget layer in between. Everything
    the friendly Generate node auto-derives (shot, language code, etc.)
    must be spelled out explicitly in the XML here.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SA_MODEL",),
                "vae": ("SA_VAE",),
                "text_encoder": ("SA_TEXT_ENCODER", {
                    "tooltip": "Connect from Scenema Audio Text Encoder Loader.",
                }),
                "xml_prompt": ("STRING", {
                    "multiline": True,
                    "default": EXAMPLE_XML,
                    "tooltip": (
                        "A single, complete <speak voice=\"...\" ...>...</speak> "
                        "prompt in the base model's native XML format. See "
                        "audio_core/compiler.py for the full schema (voice, "
                        "scene, gender, language, shot attributes; <action> "
                        "and <sound> tags)."
                    ),
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "pace": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on Kokoro's duration estimate. Higher = slower speech.",
                }),
                "strip_background_sfx": (["auto", "yes", "no"], {
                    "default": "auto",
                    "tooltip": "auto = strip only when the XML's scene attribute implies studio-clean intent ('Absolute silence', 'Broadcast studio').",
                }),
                "ref_latent": ("SA_LATENT", {
                    "tooltip": "Optional voice reference for zero-shot cloning. Connect from VAE Encode fed by LoadAudio.",
                }),
                "separator": ("SA_SEPARATOR", {
                    "tooltip": "Pre-loaded MelBandRoFormer from its loader node. Required if strip_background_sfx ends up stripping.",
                }),
                "voice_clone": ("SA_VOICE_CLONE", {
                    "tooltip": "Pre-loaded SeedVC stack from Scenema Audio Voice Clone Loader. Required whenever the generation splits into multiple chunks and skip_vc=False.",
                }),
                "validate": ("BOOLEAN", {"default": False}),
                "min_match_ratio": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "skip_vc": ("BOOLEAN", {"default": False}),
                "vc_steps": ("INT", {"default": 25, "min": 5, "max": 50, "step": 5}),
                "vc_cfg_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, text_encoder, xml_prompt, seed,
                 pace=1.5, strip_background_sfx="auto",
                 ref_latent=None, separator=None, voice_clone=None,
                 validate=False, min_match_ratio=0.9,
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):

        xml_prompt = xml_prompt.strip()
        if not xml_prompt.startswith("<speak"):
            raise ValueError(
                "xml_prompt must be a single <speak ...>...</speak> element, "
                "the base model's native prompt format. Got: "
                f"{xml_prompt[:60]!r}..."
            )

        combined_audio = run_generation(
            model, vae, text_encoder, xml_prompt, seed, separator, voice_clone,
            pace, strip_background_sfx, ref_latent, validate, min_match_ratio,
            skip_vc, vc_steps, vc_cfg_rate,
        )

        return (combined_audio,)
