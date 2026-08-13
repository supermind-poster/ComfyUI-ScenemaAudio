# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Audio composition primitives for Scenema Audio Dialogue Generate's
<speak_overlap> and <speak_interruption> tags.

The underlying model is single-voice TTS — it generates one speaker's
line at a time, cleanly, with no concept of another voice talking over
it. "Overlap" and "interruption" here are literal post-generation audio
mixing: each participating speaker's line is generated independently and
cleanly (same as any other turn), then layered together in time. This
sounds convincing for dialogue purposes but is not the model producing
genuine simultaneous conversational dynamics (interruption-aware timing,
one speaker trailing off because they got cut off, etc.) — it's audio
editing, the same technique used to build overlapping dialogue in audio
drama post-production.
"""

import torch
import torchaudio


def _resample_to(audio, target_sr):
    """Return audio's waveform resampled to target_sr if needed."""
    w = audio["waveform"]
    if audio["sample_rate"] != target_sr:
        w = torchaudio.functional.resample(w.float(), audio["sample_rate"], target_sr)
    return w


def _pad_to(waveform, length):
    """Pad a (..., samples) waveform with trailing silence to `length` samples."""
    cur = waveform.shape[-1]
    if cur >= length:
        return waveform
    pad = torch.zeros((*waveform.shape[:-1], length - cur), dtype=waveform.dtype)
    return torch.cat([waveform, pad], dim=-1)


def _sum_normalize(waveforms):
    """Sum same-length waveforms, peak-normalizing only if the sum clips."""
    summed = waveforms[0].clone()
    for w in waveforms[1:]:
        summed = summed + w
    peak = summed.abs().max()
    if peak > 1.0:
        summed = summed / peak
    return summed


def duration_seconds(audio):
    return audio["waveform"].shape[-1] / audio["sample_rate"]


def mix_overlap(clips):
    """Layer N independently generated clips so they all start at time 0
    (full overlap — unison speech, or several different simultaneous
    lines). Shorter clips are padded with trailing silence to match the
    longest. Returns one {"waveform","sample_rate"} dict.
    """
    if len(clips) == 1:
        return clips[0]

    sr = clips[0]["sample_rate"]
    waves = [_resample_to(c, sr) for c in clips]
    max_len = max(w.shape[-1] for w in waves)
    padded = [_pad_to(w, max_len) for w in waves]
    mixed = _sum_normalize(padded)
    return {"waveform": mixed, "sample_rate": sr}


def splice_interrupt(audio_a, audio_b, splice_seconds):
    """Build an interruption: party A plays cleanly up to `splice_seconds`,
    then A's tail (from splice_seconds to A's end) plays layered under the
    start of B's line, then B continues alone past wherever A finished.

    Args:
        audio_a: the party being interrupted — {"waveform","sample_rate"}.
        audio_b: the interrupting party — {"waveform","sample_rate"}.
        splice_seconds: how far into audio_a's own timeline B cuts in.
            Clamped to [0, duration(audio_a)].

    Returns:
        One {"waveform","sample_rate"} dict — A's head, then the overlap
        window (A's tail mixed with B's head), then whatever's left of B.
    """
    sr = audio_a["sample_rate"]
    wa = audio_a["waveform"]
    wb = _resample_to(audio_b, sr)

    a_samples = wa.shape[-1]
    splice_idx = int(max(0.0, min(splice_seconds, a_samples / sr)) * sr)

    a_head = wa[..., :splice_idx]
    a_tail = wa[..., splice_idx:]
    overlap_len = a_tail.shape[-1]

    b_len = wb.shape[-1]
    b_overlap_len = min(overlap_len, b_len)
    b_head = wb[..., :b_overlap_len]
    b_rest = wb[..., b_overlap_len:]

    mix_len = max(a_tail.shape[-1], b_head.shape[-1])
    overlap_mixed = _sum_normalize([_pad_to(a_tail, mix_len), _pad_to(b_head, mix_len)])

    parts = [a_head, overlap_mixed]
    if b_rest.shape[-1] > 0:
        parts.append(b_rest)

    max_channels = max(p.shape[1] for p in parts if p is not None)
    normalized_parts = []
    for p in parts:
        if p is None:
            continue
        if p.shape[1] < max_channels:
            # Doudling mono to stereo
            p = p.repeat(1, max_channels, 1)
        normalized_parts.append(p)

    combined = torch.cat(normalized_parts, dim=-1)
    return {"waveform": combined, "sample_rate": sr}
