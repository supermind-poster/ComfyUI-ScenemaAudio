# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding — fully local, no HuggingFace network calls.

Loading the Gemma 3 12B IT weights happens once in
ScenemaAudioTextEncoderLoader (nodes/text_encoder_loader.py). This module
holds the shared build/cache/encode logic used by both the standalone
ScenemaAudioTextEncode node and ScenemaAudioGenerate.

Gemma 3 12B IT ships from Google as a standard HuggingFace-layout folder
(config.json + tokenizer files + sharded *.safetensors). safetensors
already memory-maps weights lazily — `transformers.from_pretrained`
loading a local directory with `local_files_only=True` is exactly that
"lazy safetensors" loading, no special format or conversion needed. You
just need the folder to already exist on disk; see README for how to
fetch it once.

Three encode paths, matching production Scenema Audio, auto-selected by
available VRAM (or forced via the loader's `precision` input):
  - nf4:           NF4-quantized Gemma on GPU (~8GB)   — 12-39GB cards
  - bf16_gpu:       full bf16 Gemma resident on GPU (~24GB) — 40GB+ cards
  - cpu_streaming:  Gemma streamed from CPU RAM per call — <12GB cards
"""

import logging
import os

import torch
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import OffloadMode
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

logger = logging.getLogger(__name__)

# Belt-and-suspenders: force transformers/huggingface_hub into offline
# mode process-wide. Nothing in this codepath calls hf_hub_download or
# snapshot_download anymore, but some libraries probe the network by
# default (e.g. checking for newer configs) unless this is set — and we
# want a hard guarantee that loading NEVER reaches out, even if a local
# folder happens to look incomplete.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HIGH_VRAM_THRESHOLD_GB = 40
NF4_MIN_VRAM_GB = 12

# Only one Gemma stack is kept resident at a time (12B is heavy) — keyed
# by (gemma_path, pipeline_path, precision) so switching loaders rebuilds.
_CACHE_KEY = None
_CACHE_ENTRY = None


def select_encode_mode(vram_gb: float, precision: str = "auto") -> str:
    """Match production's Gemma loading strategy for the current VRAM tier,
    unless the caller forces a specific precision.
    """
    if precision != "auto":
        return precision
    if vram_gb >= HIGH_VRAM_THRESHOLD_GB:
        return "bf16_gpu"
    if vram_gb >= NF4_MIN_VRAM_GB:
        return "nf4"
    return "cpu_streaming"


def _build_nf4_gemma(gemma_path):
    """Load Gemma 3 12B with BitsAndBytes NF4 quantization (~8GB on GPU),
    entirely from local disk.
    """
    logger.info("Loading Gemma NF4 on GPU from local files (one-time)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path,
        quantization_config=quant_config,
        device_map="cuda",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
    logger.info("Gemma NF4 resident: %.1fGB VRAM", vram_used_gb)
    return model


def build_text_encoder(gemma_path, pipeline_path, precision="auto"):
    """Build (or reuse a cached) text-encoding stack from local files only.

    Args:
        gemma_path: local directory containing the Gemma 3 12B IT weights.
        pipeline_path: local scenema-audio-pipeline.safetensors path.
        precision: "auto" | "nf4" | "bf16_gpu" | "cpu_streaming".

    Returns:
        An opaque entry dict — pass it straight through as SA_TEXT_ENCODER
        into ScenemaAudioTextEncode or ScenemaAudioGenerate.
    """
    global _CACHE_KEY, _CACHE_ENTRY

    if not os.path.isdir(gemma_path):
        raise FileNotFoundError(
            f"Gemma directory not found: {gemma_path}\n"
            "Expected a local folder with config.json, tokenizer files, and "
            "*.safetensors weight shards for google/gemma-3-12b-it. See the "
            "README for how to download it once and where to place it."
        )
    if not os.path.isfile(pipeline_path):
        raise FileNotFoundError(f"Pipeline checkpoint not found: {pipeline_path}")

    key = (gemma_path, pipeline_path, precision)
    if _CACHE_KEY == key:
        return _CACHE_ENTRY

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    mode = select_encode_mode(vram_gb, precision)
    offload = OffloadMode.NONE if mode == "bf16_gpu" else OffloadMode.CPU

    logger.info("Building DistilledPipeline (local, offline, mode=%s)...", mode)
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=offload,
    )
    pe = pipeline.prompt_encoder

    entry = {"pipeline": pipeline, "mode": mode}

    if mode == "nf4":
        entry["gemma_model"] = _build_nf4_gemma(gemma_path)
        entry["emb_proc"] = pe._embeddings_processor_builder.build(
            device="cuda", dtype=torch.bfloat16,
        ).eval()
        entry["tokenizer"] = LTXVGemmaTokenizer(gemma_path)
        logger.info("NF4 encode path ready (Gemma + emb_proc GPU-resident)")
    elif mode == "bf16_gpu":
        logger.info("Loading bf16 Gemma text encoder on GPU (local, one-time)...")
        entry["text_encoder"] = pe._text_encoder_builder.build(
            device=torch.device("cuda"), dtype=torch.bfloat16,
        ).eval()
        entry["emb_proc"] = pe._embeddings_processor_builder.build(
            device="cuda", dtype=torch.bfloat16,
        ).eval()
        vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info("bf16 Gemma resident: %.1fGB VRAM", vram_used_gb)
    else:
        logger.info("CPU-streaming encode path ready (Gemma streams from CPU per call)")

    _CACHE_KEY = key
    _CACHE_ENTRY = entry
    return entry


def encode_prompt(text_encoder_entry, compiled_prompt):
    """Encode a single compiled prompt string with an already-built
    text_encoder_entry (from build_text_encoder / ScenemaAudioTextEncoderLoader).

    Returns:
        (video_context, audio_context) tensors.
    """
    mode = text_encoder_entry["mode"]
    pipeline = text_encoder_entry["pipeline"]

    with torch.inference_mode():
        if mode == "nf4":
            tp = text_encoder_entry["tokenizer"].tokenize_with_weights(compiled_prompt)["gemma"]
            ids = torch.tensor([[t[0] for t in tp]], device="cuda")
            mask = torch.tensor([[w[1] for w in tp]], device="cuda")
            out = text_encoder_entry["gemma_model"].model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
            )
            emb = text_encoder_entry["emb_proc"].process_hidden_states(out.hidden_states, mask)
            del out, ids
        elif mode == "bf16_gpu":
            hs, am = text_encoder_entry["text_encoder"].encode(compiled_prompt)
            emb = text_encoder_entry["emb_proc"].process_hidden_states(hs, am)
        else:
            (emb,) = pipeline.prompt_encoder([compiled_prompt])

        vc = emb.video_encoding
        ac = emb.audio_encoding

    return vc, ac


class ScenemaAudioTextEncode:
    """Encodes a compiled prompt with an already-loaded Scenema Audio text
    encoder (connect from Scenema Audio Text Encoder Loader).
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "encode"
    RETURN_TYPES = ("SA_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "compiled_prompt": ("STRING", {"forceInput": True}),
                "text_encoder": ("SA_TEXT_ENCODER",),
            },
        }

    def encode(self, compiled_prompt, text_encoder):
        vc, ac = encode_prompt(text_encoder, compiled_prompt)
        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
