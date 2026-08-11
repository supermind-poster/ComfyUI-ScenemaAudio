# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Shared generation engine for Scenema Audio.

This is the actual diffusion pipeline: XML -> chunk plan -> Gemma encode
-> transformer diffuse (with A2V chaining across chunks) -> decode ->
trim/normalize/concat -> optional SeedVC voice-lock -> optional
MelBandRoFormer background strip.

Three nodes all call run_generation() with a fully-formed <speak> XML
string, differing only in how they build/obtain that XML:
  - ScenemaAudioGenerate       builds it from friendly widget fields
  - ScenemaAudioGenerateXML    takes it verbatim from the user (raw
                                <speak> XML, exactly the base model's
                                native prompt format)
  - ScenemaAudioDialogueGenerate  splits a <dialogue> of many <speak>
                                turns and calls run_generation() once
                                per turn, batched by speaker
"""

import logging
import os
import sys

import numpy as np
import torch
import torchaudio
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio
from ltx_pipelines.distilled import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop

from .sampler import (
    _build_pixel_shape, _build_video_state, _build_audio_state,
    _apply_a2v_reference, _strip_reference_frames,
)
from .text_encode import encode_prompt
from .vocal_separator import _run_separator
from .seedvc import convert_voice

# audio_core/ lives at the package root (sibling of nodes/), not inside
# nodes/, so it isn't importable as a bare "audio_core.xxx" unless the
# package root is on sys.path. ComfyUI does not add a custom node's own
# root directory to sys.path automatically, so we do it ourselves here.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.chunker import plan_chunks, estimate_duration, ChunkSpec
from audio_core.compiler import compile_prompt
from audio_core.audio_utils import trim_silence, normalize_volume, shorten_long_silence
from audio_core.whisper_aligner import validate_text

logger = logging.getLogger(__name__)

REF_TAIL_SECONDS = 3.0
MAX_RETRIES = 3
RETRY_DURATION_FACTOR = 1.3

# Scenes that carry the "no ambient / studio-clean" intent — kept here too
# (mirrors generate.py's CLEAN_SPEECH_SCENES) since run_generation makes its
# own strip decision from the raw XML's scene attribute when possible.
CLEAN_SPEECH_SCENES = {"Absolute silence", "Broadcast studio"}


def _log_vram(label):
    """Log current and peak VRAM usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    logger.info("VRAM [%s]: %.2fGB allocated, %.2fGB peak, %.2fGB reserved",
                label, allocated, peak, reserved)


def _decode_latent(vae_data, latent):
    """Decode audio latent to waveform."""
    decoder = vae_data["decoder"]
    audio_obj = decoder(latent.cuda())
    waveform = audio_obj.waveform.cpu()
    sr = audio_obj.sampling_rate
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sr


def _encode_reference(vae_data, waveform, sr, max_seconds=REF_TAIL_SECONDS):
    """Encode tail of waveform as A2V reference for the next chunk/turn."""
    encoder = vae_data["encoder"]
    vae_sr = vae_data["sample_rate"]

    tail_samples = int(max_seconds * sr)
    wav = waveform[0, :, -tail_samples:]

    if sr != vae_sr:
        wav = torchaudio.functional.resample(wav.float(), sr, vae_sr)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    encoder_was_cpu = str(next(encoder.parameters()).device) == "cpu"
    if encoder_was_cpu:
        encoder.cuda()

    audio_obj = Audio(waveform=wav.unsqueeze(0).cuda(), sampling_rate=vae_sr)
    latent = encode_audio(audio_obj, encoder)

    if encoder_was_cpu:
        encoder.cpu()

    return latent


def _plan(xml_prompt, compiled_prompt, speech_text, seed, pace):
    """Plan chunks from the XML — falls back to single chunk on failure."""
    try:
        chunks = plan_chunks(xml_prompt, base_seed=seed, pace=pace)
        if chunks:
            return chunks
    except Exception as e:
        logger.warning("Chunking failed, falling back to single chunk: %s", e)

    duration = estimate_duration(speech_text, multiplier=pace)
    return [ChunkSpec(
        compiled_prompt=compiled_prompt,
        duration_s=duration,
        seed=seed,
        expected_text=speech_text,
    )]


def _encode_all_chunks(chunks, text_encoder):
    """Encode all chunk prompts using an already-loaded text_encoder
    (from Scenema Audio Text Encoder Loader).
    """
    all_encodings = []
    for i, chunk in enumerate(chunks):
        logger.info("  Encoding chunk %d/%d", i + 1, len(chunks))
        vc, ac = encode_prompt(text_encoder, chunk.compiled_prompt)
        all_encodings.append((vc, ac))
    return all_encodings


def _diffuse_chunk(mdl_wrapper, device, vc, ac, duration_s, seed, ref_latent=None):
    """Run diffusion for a single chunk. Transformer must already be on GPU."""
    pixel_shape = _build_pixel_shape(duration_s)
    gen = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=gen)

    video_state = _build_video_state(pixel_shape, vc, noiser, device)
    audio_state, audio_tools = _build_audio_state(pixel_shape, ac, noiser, device)

    ref_frames = 0
    if ref_latent is not None:
        audio_state, ref_frames = _apply_a2v_reference(
            audio_state, ac, ref_latent, seed, device
        )

    sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    stepper = EulerDiffusionStep()
    wrapped = BatchSplitAdapter(mdl_wrapper, max_batch_size=1)

    _, audio_state_out = euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        transformer=wrapped,
        denoiser=SimpleDenoiser(vc, ac),
    )

    if ref_frames > 0 and audio_state_out is not None:
        audio_state_out = _strip_reference_frames(audio_state_out, ref_frames)

    audio_state_out = audio_tools.clear_conditioning(audio_state_out)
    audio_state_out = audio_tools.unpatchify(audio_state_out)

    return audio_state_out.latent


def _diffuse_with_validation(mdl_wrapper, device, vae, vc, ac, chunk, ref_gpu, min_match_ratio):
    """Diffuse a chunk with Whisper validation and retry on word-match failure."""
    duration = chunk.duration_s
    seed = chunk.seed
    best_waveform = None
    best_sr = None
    best_ratio = -1.0

    for attempt in range(MAX_RETRIES + 1):
        latent = _diffuse_chunk(mdl_wrapper, device, vc, ac, duration, seed, ref_gpu)
        waveform, sr = _decode_latent(vae, latent)

        wav_np = waveform.squeeze(0).numpy()
        if wav_np.ndim == 2:
            wav_np = wav_np.T
        passed, transcribed, ratio = validate_text(
            wav_np, sr, chunk.expected_text,
            language=chunk.language, min_word_ratio=min_match_ratio,
        )

        if ratio > best_ratio:
            best_waveform = waveform
            best_sr = sr
            best_ratio = ratio

        if passed:
            logger.info("  Validated: %.0f%% word match", ratio * 100)
            return {"waveform": best_waveform, "sample_rate": best_sr}

        if attempt < MAX_RETRIES:
            duration = min(duration * RETRY_DURATION_FACTOR, 20.0)
            seed += 1
            logger.info("  Retry %d: %.0f%% match, extending to %.1fs, seed=%d",
                         attempt + 1, ratio * 100, duration, seed)

    logger.warning("  Best %.0f%% match after %d retries, accepting",
                    best_ratio * 100, MAX_RETRIES)
    return {"waveform": best_waveform, "sample_rate": best_sr}


def _apply_vc(combined_audio, chunk_waveforms, sr, ref_latent, vae, voice_clone, vc_steps, vc_cfg_rate):
    """Apply SeedVC for voice consistency across chunks."""
    chunk0_audio = {"waveform": chunk_waveforms[0], "sample_rate": sr}

    logger.info("Applying SeedVC (%d steps, cfg_rate=%.2f)...", vc_steps, vc_cfg_rate)
    result = convert_voice(combined_audio, chunk0_audio, voice_clone, vc_steps, vc_cfg_rate)

    if result["sample_rate"] != sr:
        result_wav = torchaudio.functional.resample(
            result["waveform"].float(), result["sample_rate"], sr
        )
        result = {"waveform": result_wav, "sample_rate": sr}

    return result


def should_strip_background(scene, override):
    """Decide whether to run MelBandRoFormer post-processing.

    auto: strip if scene implies studio-clean intent (Absolute silence,
          Broadcast studio). Ambient scenes are left alone.
    yes: always strip.
    no: never strip.
    """
    if override == "yes":
        return True
    if override == "no":
        return False
    return scene in CLEAN_SPEECH_SCENES


def strip_background(audio, separator):
    """Run MelBandRoFormer to isolate speech from any ambient bleed.

    separator: pre-loaded model from Scenema Audio MelBandRoFormer
        Loader. Required — no internal lazy-loading fallback.
    """
    logger.info("Stripping background SFX...")
    vocals_t, _, sr = _run_separator(audio, separator=separator)
    return {"waveform": vocals_t, "sample_rate": sr}


def run_generation(model, vae, text_encoder, xml_prompt, seed, separator, voice_clone,
                    pace=1.5, strip_background_sfx="auto", ref_latent=None,
                    validate=False, min_match_ratio=0.9,
                    skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):
    """Run the full Scenema Audio pipeline for one <speak> XML prompt.

    This is the shared engine behind ScenemaAudioGenerate,
    ScenemaAudioGenerateXML, and (once per turn) ScenemaAudioDialogueGenerate.

    Args:
        model: SA_MODEL dict from Scenema Audio Diffusion Loader.
        vae: SA_VAE dict from Scenema Audio VAE Loader.
        text_encoder: SA_TEXT_ENCODER dict from Scenema Audio Text Encoder Loader.
        xml_prompt: a single, complete <speak ...>...</speak> string.
        seed: base seed (chunks after the first derive their own).
        separator: SA_SEPARATOR from Scenema Audio MelBandRoFormer Loader.
            Required whenever strip_background_sfx could trigger a strip
            (raises a clear error if it's actually needed and missing).
        voice_clone: SA_VOICE_CLONE from Scenema Audio Voice Clone Loader.
            Required whenever cross-chunk voice-lock could run (multiple
            chunks and skip_vc=False; raises a clear error otherwise).
        pace: Kokoro duration-estimate multiplier.
        strip_background_sfx: "auto" | "yes" | "no".
        ref_latent: optional SA_LATENT voice-clone / continuity reference.
        validate: Whisper-check + retry chunks that dropped words.
        min_match_ratio: word-match threshold for validate.
        skip_vc: skip the cross-chunk SeedVC voice-lock pass.
        vc_steps, vc_cfg_rate: SeedVC parameters.

    Returns:
        {"waveform": tensor, "sample_rate": int} — a ComfyUI AUDIO dict.
    """
    compiled = compile_prompt(xml_prompt)
    logger.info("Compiled prompt: %s", compiled.prompt)
    chunks = _plan(xml_prompt, compiled.prompt, compiled.speech_text, seed, pace)
    for i, c in enumerate(chunks):
        logger.info("Chunk %d prompt: %s", i + 1, c.compiled_prompt)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _log_vram("start")
    logger.info("Generating %d chunk(s) (validate=%s, skip_vc=%s)...",
                len(chunks), validate, skip_vc)

    # Offload transformer to CPU before Phase 1: Gemma needs the VRAM.
    mdl_wrapper = model["model"]
    device = model["device"]
    mdl_wrapper.to("cpu")
    torch.cuda.empty_cache()
    _log_vram("transformer offloaded pre-Phase 1")

    # ── Phase 1: Encode ALL chunk prompts in one Gemma session ──
    logger.info("Phase 1: Encoding %d prompts...", len(chunks))
    chunk_encodings = _encode_all_chunks(chunks, text_encoder)
    _log_vram("after all encoding")

    # ── Phase 2: Diffuse + decode ALL chunks in one transformer session ──
    logger.info("Phase 2: Diffusing %d chunks...", len(chunks))
    mdl_wrapper.to(device)
    _log_vram("transformer on GPU")

    chunk_encodings_cpu = [(vc.cpu(), ac.cpu()) for vc, ac in chunk_encodings]
    del chunk_encodings
    torch.cuda.empty_cache()

    waveforms = []
    sr = None
    current_ref = ref_latent.cpu() if ref_latent is not None else None
    for i, (chunk, (vc_cpu, ac_cpu)) in enumerate(zip(chunks, chunk_encodings_cpu)):
        logger.info("  Diffuse chunk %d/%d (%.1fs)", i + 1, len(chunks), chunk.duration_s)
        vc = vc_cpu.to(device)
        ac = ac_cpu.to(device)
        ref_gpu = current_ref.to(device) if current_ref is not None else None

        if validate:
            waveform = _diffuse_with_validation(
                mdl_wrapper, device, vae, vc, ac, chunk, ref_gpu, min_match_ratio
            )
        else:
            latent = _diffuse_chunk(mdl_wrapper, device, vc, ac,
                                     chunk.duration_s, chunk.seed, ref_gpu)
            waveform, sr = _decode_latent(vae, latent)

        if validate:
            sr = waveform["sample_rate"]
            waveform = waveform["waveform"]

        del vc, ac, ref_gpu
        waveforms.append(waveform)

        if i < len(chunks) - 1:
            current_ref = _encode_reference(vae, waveform, sr).cpu()

    mdl_wrapper.to("cpu")
    torch.cuda.empty_cache()
    _log_vram("after all chunks")

    # Per-chunk trim + normalize before concatenation (matches production).
    processed_np = []
    for w in waveforms:
        w_np = w.squeeze(0).numpy()  # (C, samples)
        w_np = w_np.T if w_np.ndim == 2 else w_np  # (samples, C)
        w_np = trim_silence(w_np, sr, max_silence=0.5)
        w_np = normalize_volume(w_np, sr)
        processed_np.append(w_np)
    combined_np = np.concatenate(processed_np, axis=0)
    combined_np = shorten_long_silence(combined_np, sr, max_duration=min(0.5 * pace, 1.5))

    if combined_np.ndim == 2:
        combined_np = combined_np.T  # (C, samples)
    combined = torch.from_numpy(combined_np).float()
    if combined.ndim == 1:
        combined = combined.unsqueeze(0)
    combined = combined.unsqueeze(0)  # (1, C, samples)
    combined_audio = {"waveform": combined, "sample_rate": sr}

    do_strip = should_strip_background(compiled.scene, strip_background_sfx)

    needs_vc = len(chunks) > 1
    if not skip_vc and needs_vc:
        if voice_clone is None:
            raise ValueError(
                "This generation split into multiple chunks, which needs "
                "the cross-chunk voice-lock pass (SeedVC) to keep the voice "
                "consistent. Connect a Scenema Audio Voice Clone Loader to "
                "the `voice_clone` input, or set skip_vc=True to accept "
                "possible voice drift between chunks instead."
            )
        combined_audio = _apply_vc(
            combined_audio, waveforms, sr, ref_latent, vae, voice_clone, vc_steps, vc_cfg_rate,
        )
    elif not needs_vc:
        logger.info("Single chunk — skipping cross-chunk voice polish")

    if do_strip:
        if separator is None:
            raise ValueError(
                f"strip_background_sfx={strip_background_sfx!r} needs "
                "MelBandRoFormer to run. Connect a Scenema Audio "
                "MelBandRoFormer Loader to the `separator` input, or set "
                "strip_background_sfx='no' to skip stripping instead."
            )
        combined_audio = strip_background(combined_audio, separator=separator)

    total_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
    logger.info("Generation complete: %.1fs from %d chunk(s)", total_duration, len(chunks))

    return combined_audio
