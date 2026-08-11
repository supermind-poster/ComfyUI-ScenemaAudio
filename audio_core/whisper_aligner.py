# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Whisper alignment for audio validation in Scenema Audio.

Used for validate=True's word-match retry and for <speak_interruption>'s
!INTERRUPTION! marker alignment in Dialogue Generate.

Reuses the SAME local whisper-small checkpoint Scenema Audio Voice Clone
Loader already loads for SeedVC's speech tokenizer — there is no reason
for this to be a second copy in a second format. Both go through
`transformers`, both read from <this package>/models/voice_clone/. If
you've already placed a whisper-small folder there for Voice Clone
Loader (per the README), this just works — nothing extra to download,
and this module never touches HuggingFace under any circumstance (it
isn't the Voice Clone Loader node, so per this package's policy it
doesn't get to).

If no local checkpoint is found, validate=True and the !INTERRUPTION!
marker both fail gracefully (validate raises with a clear message;
!INTERRUPTION! logs a warning and falls back to a percentage-based
splice point) rather than reaching for the network.
"""

import logging
import os
import re
import unicodedata

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_whisper_pipe = None


def _find_local_whisper_checkpoint():
    """Look for a full, generation-capable Whisper checkpoint under
    <this package>/models/voice_clone/ — the same folder Scenema Audio
    Voice Clone Loader reads from. Prefers a folder literally named
    "whisper-small" (the README's suggested layout); otherwise scans for
    any subfolder that looks like a full checkpoint (has
    generation_config.json — encoder-only exports won't have this).
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voice_clone_dir = os.path.join(pkg_root, "models", "voice_clone")
    if not os.path.isdir(voice_clone_dir):
        return None

    preferred = os.path.join(voice_clone_dir, "whisper-small")
    if os.path.isfile(os.path.join(preferred, "generation_config.json")):
        return preferred

    for name in sorted(os.listdir(voice_clone_dir)):
        candidate = os.path.join(voice_clone_dir, name)
        if os.path.isfile(os.path.join(candidate, "generation_config.json")):
            return candidate

    return None


def _get_whisper():
    """Get or initialize the Whisper ASR pipeline from a local checkpoint.
    Never touches the network — raises a clear FileNotFoundError if no
    checkpoint is found.

    Tries GPU float16 first; if that fails, falls back to CPU
    automatically. Slower, but still works.
    """
    global _whisper_pipe

    if _whisper_pipe is not None:
        return _whisper_pipe

    model_dir = _find_local_whisper_checkpoint()
    if model_dir is None:
        raise FileNotFoundError(
            "No local Whisper checkpoint found under "
            "<this package>/models/voice_clone/. This is needed for "
            "validate=True and the !INTERRUPTION! marker — it reuses the "
            "SAME whisper-small folder Scenema Audio Voice Clone Loader "
            "uses for SeedVC (e.g. models/voice_clone/whisper-small/, "
            "downloaded from openai/whisper-small). If you've already set "
            "that up for Voice Clone Loader, this should just work; if not, "
            "see the README. This module never fetches anything automatically."
        )

    import torch
    from transformers import pipeline

    logger.info("Loading Whisper for alignment from %s...", model_dir)
    try:
        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_dir,
            device="cuda",
            torch_dtype=torch.float16,
        )
        logger.info("Whisper alignment pipeline loaded (GPU, float16)")
    except Exception as e:
        logger.warning("GPU load failed (%s) — falling back to CPU (slower).", e)
        _whisper_pipe = pipeline("automatic-speech-recognition", model=model_dir, device="cpu")
        logger.info("Whisper alignment pipeline loaded (CPU)")
    return _whisper_pipe


def _to_mono_16k(audio_np, sr):
    if audio_np.ndim == 2:
        audio_mono = audio_np.mean(axis=1).astype(np.float32)
    else:
        audio_mono = audio_np.astype(np.float32)
    if sr != 16000:
        audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
    return audio_mono


def transcribe(audio_np: np.ndarray, sr: int, language: str = "en") -> str:
    """Transcribe audio and return the text."""
    pipe = _get_whisper()
    audio_mono = _to_mono_16k(audio_np, sr)

    try:
        result = pipe(
            {"raw": audio_mono, "sampling_rate": 16000},
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return (result.get("text") or "").strip()
    except (ValueError, TypeError):
        logger.debug("Whisper transcribe returned unexpected type (test env?)")
        return ""


def find_marker_time(audio_np: np.ndarray, sr: int, expected_text: str,
                      marker_char_offset: int, language: str = "en") -> float | None:
    """Locate the playback time in `audio_np` corresponding to a text
    position, using Whisper's word-level timestamps.

    Used for the `!INTERRUPTION!` marker in <speak_interruption>: counts
    how many words of `expected_text` precede `marker_char_offset`, then
    returns the end time of that many words in the actual transcription.

    This is a best-effort positional alignment, not exact forced
    alignment — it assumes Whisper's word count roughly tracks the
    expected text's word count up to the marker. Returns None if
    transcription fails or has fewer words than expected (caller should
    fall back to a percentage-based splice point in that case).
    """
    words_before_marker = len(expected_text[:marker_char_offset].split())
    if words_before_marker <= 0:
        return 0.0

    pipe = _get_whisper()
    audio_mono = _to_mono_16k(audio_np, sr)

    try:
        result = pipe(
            {"raw": audio_mono, "sampling_rate": 16000},
            return_timestamps="word",
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        chunks = result.get("chunks") or []
    except (ValueError, TypeError, KeyError):
        logger.debug("Whisper word-timestamp transcribe failed (test env?)")
        return None

    if len(chunks) < words_before_marker:
        logger.warning(
            "!INTERRUPTION! marker alignment: transcription had %d words, "
            "expected at least %d before the marker — falling back to "
            "percentage-based splice point.",
            len(chunks), words_before_marker,
        )
        return None

    target = chunks[words_before_marker - 1]
    end_time = target["timestamp"][1]
    if end_time is None:
        # Whisper occasionally can't pin down a word's exact end,
        # usually the last one in a chunk — use the next word's start,
        # or this word's own start, as the best available fallback.
        if words_before_marker < len(chunks):
            end_time = chunks[words_before_marker]["timestamp"][0]
        else:
            end_time = target["timestamp"][0]

    return end_time


def validate_text(
    audio_np: np.ndarray,
    sr: int,
    expected_text: str,
    language: str = "en",
    min_word_ratio: float = 0.6,
) -> tuple[bool, str, float]:
    """Validate that generated audio contains the expected text."""
    transcribed = transcribe(audio_np, sr, language)

    def normalize(t):
        t = unicodedata.normalize("NFD", t)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return set(t.split())

    expected_words = normalize(expected_text)
    transcribed_words = normalize(transcribed)

    if not expected_words:
        return True, transcribed, 1.0

    matched = expected_words & transcribed_words
    ratio = len(matched) / len(expected_words)

    passed = ratio >= min_word_ratio
    if not passed:
        logger.warning(
            "Validation failed: %.0f%% word match (need %.0f%%). "
            "Expected: %s... Got: %s...",
            ratio * 100,
            min_word_ratio * 100,
            expected_text[:60],
            transcribed[:60],
        )

    return passed, transcribed, ratio
