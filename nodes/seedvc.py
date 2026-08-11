# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio voice conversion node for ComfyUI.

Converts voice identity of source audio to match a reference speaker
while preserving prosody, rhythm, and emotion. Uses vendored Seed-VC
code. Requires a pre-loaded `voice_clone` stack from Scenema Audio Voice
Clone Loader — this node no longer loads (or auto-downloads) anything
itself. See nodes/voice_clone_loader.py for what's local vs. auto-fetched.
"""

import inspect
import logging
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

DEFAULT_STEPS = 25
DEFAULT_CFG_RATE = 0.5


def _run_conversion(app_vc, source_path, target_path, steps, cfg_rate):
    """Run SeedVC voice conversion and return audio samples."""
    vc_kwargs = {
        "source": source_path,
        "target": target_path,
        "diffusion_steps": steps,
        "length_adjust": 1.0,
        "inference_cfg_rate": cfg_rate,
    }
    sig = inspect.signature(app_vc.voice_conversion)
    if "n_quantizers" in sig.parameters:
        vc_kwargs["n_quantizers"] = 3

    audio_tuple = None
    for result in app_vc.voice_conversion(**vc_kwargs):
        if isinstance(result, tuple) and len(result) == 2:
            _, audio_tuple = result

    if audio_tuple is None:
        raise RuntimeError("SeedVC produced no output")

    sample_rate, samples = audio_tuple
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    peak = np.abs(samples).max()
    if peak > 1.0:
        samples = samples / peak

    return samples, sample_rate


def _audio_to_temp_wav(audio_data, target_sr):
    """Write ComfyUI AUDIO to a temp WAV file at target sample rate."""
    wav = audio_data["waveform"][0]
    sr = audio_data["sample_rate"]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.float(), sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav.squeeze().numpy(), target_sr)
    return tmp.name


def convert_voice(source_audio, reference_audio, voice_clone,
                   steps=DEFAULT_STEPS, cfg_rate=DEFAULT_CFG_RATE):
    """Convert voice identity. Used by both the standalone node and the
    shared generation engine (nodes/generate_core.py) for cross-chunk
    voice consistency.

    Args:
        source_audio: ComfyUI AUDIO dict (source speech)
        reference_audio: ComfyUI AUDIO dict (target voice identity)
        voice_clone: SA_VOICE_CLONE stack from Scenema Audio Voice Clone
            Loader — required, no fallback loading happens here.
        steps: SeedVC diffusion steps
        cfg_rate: Classifier-free guidance rate

    Returns:
        ComfyUI AUDIO dict with converted voice
    """
    app_vc = voice_clone

    src_path = _audio_to_temp_wav(source_audio, app_vc.sr)
    ref_path = _audio_to_temp_wav(reference_audio, app_vc.sr)

    try:
        samples, out_sr = _run_conversion(app_vc, src_path, ref_path, steps, cfg_rate)
    finally:
        os.unlink(src_path)
        os.unlink(ref_path)
        # NOTE: the stack is NOT unloaded here anymore. It's owned by the
        # loader node now (that's the point of loading it there), so it
        # stays resident across calls instead of being freed after every
        # use. If you're on an 8GB card and stacking this with the main
        # transformer + Gemma, watch total VRAM — see README.

    out_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    return {"waveform": out_tensor, "sample_rate": out_sr}


class ScenemaAudioVoiceClone:
    """Voice conversion using SeedVC.

    Converts the voice identity of source audio to match a reference
    speaker while preserving the source's delivery, emotion, and pacing.
    Requires a `voice_clone` stack from Scenema Audio Voice Clone Loader.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "convert"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("AUDIO",),
                "reference": ("AUDIO",),
                "voice_clone": ("SA_VOICE_CLONE", {
                    "tooltip": "Connect from Scenema Audio Voice Clone Loader.",
                }),
            },
            "optional": {
                "steps": ("INT", {
                    "default": DEFAULT_STEPS, "min": 5, "max": 50, "step": 5,
                }),
                "cfg_rate": ("FLOAT", {
                    "default": DEFAULT_CFG_RATE, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
            },
        }

    @torch.inference_mode()
    def convert(self, source, reference, voice_clone, steps=DEFAULT_STEPS, cfg_rate=DEFAULT_CFG_RATE):
        logger.info("Running voice conversion (%d steps, cfg_rate=%.2f)...", steps, cfg_rate)
        result = convert_voice(source, reference, voice_clone, steps, cfg_rate)
        logger.info("Voice conversion complete: %.1fs",
                     result["waveform"].shape[-1] / result["sample_rate"])
        return (result,)
