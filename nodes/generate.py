# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Generate — the main speech generation node.

Takes voice description, gender, speech text, and scene directly. Builds
the <speak> XML the LTX model expects internally (see generate_core.py
for the shared diffusion engine also used by ScenemaAudioGenerateXML and
ScenemaAudioDialogueGenerate).
"""

import logging
import re as _re

import torch

from .presets import CUSTOM, PRESETS, PRESET_NAMES
from .generate_core import run_generation

logger = logging.getLogger(__name__)

# Curated scene presets. First entry is a sentinel that forces the user
# to make an explicit choice — we raise an error if it's not overridden.
SCENE_SENTINEL = "Choose a scene..."
SCENE_PRESETS = [
    SCENE_SENTINEL,
    "Absolute silence",
    "Quiet indoor room",
    "Reverberant hall",
    "Broadcast studio",
    "Outdoor, open air",
    "Café or restaurant",
    "Windy outdoors",
    "Rainy outdoors",
]

CLEAN_SPEECH_SCENES = {"Absolute silence", "Broadcast studio"}

# 12 tested languages (from the announcement + Pro blog posts on scenema.ai)
LANGUAGE_OPTIONS = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Korean", "Chinese", "Hindi", "Arabic", "Swahili",
]

LANGUAGE_CODES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
    "Chinese": "zh", "Hindi": "hi", "Arabic": "ar", "Swahili": "sw",
}


def _derive_shot(scene):
    """Auto-derive the film 'shot' attribute from scene semantics.

    Clean scenes (silence, studio) get 'closeup' — dry, dialogue-focused.
    All other scenes get 'wide' — ambient bleeds in around the speech.
    """
    return "closeup" if scene in CLEAN_SPEECH_SCENES else "wide"


def _xml_escape(text):
    """Escape &, <, > for XML text content. Attribute values need extra quoting."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _speech_body_parts(speech_text):
    """Parse [bracketed] cues in speech text and yield interleaved
    text and <action> XML fragments in document order.
    """
    parts = _re.split(r'\[([^\]]+)\]', speech_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                yield f"  {_xml_escape(text)}"
        else:
            cue = part.strip()
            if cue:
                yield f"  <action>{_xml_escape(cue)}</action>"


def build_xml(voice_description, gender, speech_text, scene, custom_scene,
              action_tags, language):
    """Construct the <speak> XML the compiler expects from friendly form fields.

    action_tags: multiline field, one cue per line, prepended before the first
        sentence to set the opening delivery.
    speech_text: may contain inline [bracketed cues] that become <action>
        tags at the exact position they appear.
    """
    scene_text = custom_scene.strip() if custom_scene and custom_scene.strip() else scene
    lang_code = LANGUAGE_CODES.get(language, "en")
    shot = _derive_shot(scene)

    voice_attr = _xml_escape(voice_description).replace('"', '&quot;')
    attrs = f'voice="{voice_attr}" gender="{gender}"'
    if scene_text:
        attrs += f' scene="{_xml_escape(scene_text)}"'
    if lang_code != "en":
        attrs += f' language="{lang_code}"'
    if shot != "closeup":
        attrs += f' shot="{shot}"'

    body_parts = []
    if action_tags and action_tags.strip():
        for line in action_tags.strip().split("\n"):
            line = line.strip()
            if line:
                body_parts.append(f"  <action>{_xml_escape(line)}</action>")
    body_parts.extend(_speech_body_parts(speech_text))
    body = "\n".join(body_parts)

    return f"<speak {attrs}>\n{body}\n</speak>"


class ScenemaAudioGenerate:
    """Generate expressive speech from a voice description + text.

    Handles long text automatically by splitting at sentence boundaries
    (Kokoro-timed), chaining chunks with A2V voice conditioning for
    consistency, then polishing with SeedVC. Auto-strips background bleed
    for clean scenes. Optional Whisper validation retries chunks that
    dropped words.

    For raw <speak> XML prompts (the base model's native format), use
    Scenema Audio Generate (XML) instead. For multi-speaker dialogue, use
    Scenema Audio Dialogue Generate.
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
                "preset": (PRESET_NAMES, {
                    "default": CUSTOM,
                    "tooltip": "Pick a preset to auto-fill voice, gender, scene, action tags, and speech text. Choose Custom to write your own.",
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "Male, late 60s. Deep, gravelly. Slow and deliberate. The weight of the cosmos in every word.",
                    "tooltip": "Describe the voice: age, gender presentation, timbre, accent, delivery style.",
                }),
                "gender": (["male", "female"], {
                    "tooltip": "Grammatical gender used for pronouns in the compiled prompt (he/she).",
                }),
                "speech_text": ("STRING", {
                    "multiline": True,
                    "default": "Look again at that dot. That's here. That's home. That's us.",
                    "tooltip": "The text to speak. Use [bracketed cues] inline for mid-speech performance direction: [He laughs], [She whispers], [His voice cracks]. Long text is auto-split at sentence boundaries.",
                }),
                "scene": (SCENE_PRESETS, {
                    "tooltip": "Acoustic environment. Injected into the prompt so the model imagines the space. 'Absolute silence' and 'Broadcast studio' auto-strip background bleed after generation.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_scene": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Freeform scene description. When non-empty, overrides the scene dropdown (e.g. 'Empty subway platform, distant train').",
                }),
                "action_tags": ("STRING", {
                    "multiline": True,
                    "default": "He pauses, weighing his words\nHe leans forward",
                    "tooltip": "Delivery cues. One per line. Each becomes a stage direction the model performs (e.g. 'He whispers', 'She laughs bitterly').",
                }),
                "language": (LANGUAGE_OPTIONS, {
                    "default": "English",
                    "tooltip": "Target language for the speech text. Write the text in that language.",
                }),
                "pace": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on Kokoro's duration estimate. Higher = slower speech (more time per word). 1.5 is validated as a safe default.",
                }),
                "strip_background_sfx": (["auto", "yes", "no"], {
                    "default": "auto",
                    "tooltip": "auto = strip only for clean scenes (silence, studio). yes = always strip. no = never strip.",
                }),
                "ref_latent": ("SA_LATENT", {
                    "tooltip": "Optional voice reference for zero-shot cloning. Connect from VAE Encode fed by LoadAudio.",
                }),
                "separator": ("SA_SEPARATOR", {
                    "tooltip": "Pre-loaded MelBandRoFormer from its loader node. Required if strip_background_sfx ends up stripping (auto/yes) — raises a clear error at runtime if needed and not connected.",
                }),
                "voice_clone": ("SA_VOICE_CLONE", {
                    "tooltip": "Pre-loaded SeedVC stack from Scenema Audio Voice Clone Loader. Required whenever the generation splits into multiple chunks and skip_vc=False — raises a clear error at runtime if needed and not connected.",
                }),
                "validate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whisper-check each chunk and retry with more time if words were dropped. Adds ~1s per chunk.",
                }),
                "min_match_ratio": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "skip_vc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the cross-chunk voice consistency polish. Faster, but voice may drift subtly between chunks on longer generations.",
                }),
                "vc_steps": ("INT", {"default": 25, "min": 5, "max": 50, "step": 5}),
                "vc_cfg_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, text_encoder, preset, voice_description, gender,
                 speech_text, scene, seed,
                 custom_scene="", action_tags="", language="English", pace=1.5,
                 strip_background_sfx="auto",
                 ref_latent=None, separator=None, voice_clone=None,
                 validate=False, min_match_ratio=0.9,
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):

        # Preset override — populated by the JS hook client-side, but also
        # applied here as a safety net in case widgets weren't synced.
        if preset != CUSTOM and preset in PRESETS:
            p = PRESETS[preset]
            voice_description = p["voice_description"]
            gender = p["gender"]
            speech_text = p["speech_text"]
            scene = p["scene"]
            action_tags = p["action_tags"]
            custom_scene = p.get("custom_scene", "")
            logger.info("Preset applied: %s", preset)

        if scene == SCENE_SENTINEL:
            raise ValueError(
                "scene: required field. Pick a preset, one of the acoustic "
                "options, or provide a custom_scene string."
            )

        xml_prompt = build_xml(voice_description, gender, speech_text,
                                scene, custom_scene, action_tags, language)
        logger.info("XML prompt:\n%s", xml_prompt)

        combined_audio = run_generation(
            model, vae, text_encoder, xml_prompt, seed, separator, voice_clone,
            pace, strip_background_sfx, ref_latent, validate, min_match_ratio,
            skip_vc, vc_steps, vc_cfg_rate,
        )

        return (combined_audio,)
