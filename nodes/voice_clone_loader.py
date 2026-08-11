# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Voice Clone Loader.

Loads the SeedVC stack used for cross-chunk voice consistency (inside
Generate / Generate XML / Dialogue Generate, whenever skip_vc=False) and
by the standalone Scenema Audio Voice Clone node.

These are helper checkpoints specific to this node's SeedVC integration,
not general-purpose models another custom node would ever look for — so
none of them live under ComfyUI's shared models/ tree. They live
directly inside this package instead:

  - CAMPPlus (speaker embedding)  <- <this package>/models/voice_clone/campplus_cn_common.bin
  - BigVGAN  (vocoder)            <- <this package>/models/voice_clone/bigvgan/
  - Whisper  (speech tokenizer)   <- <this package>/models/voice_clone/whisper-small/
  - DiT      (conversion model)   <- <this package>/models/voice_clone/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
                                     + config_dit_mel_seed_uvit_whisper_small_wavenet.yml

This is the only node in the whole package that ever talks to
HuggingFace, and even here only for the DiT checkpoint, and only as a
LAST RESORT: it checks for the file locally first (same folder as
everything else above); HuggingFace is only contacted if that local
check comes up empty, at which point it fetches once, caches the result
into that local folder, and never needs the network again. Not gated,
~98M parameters.

Implementation note: this reuses the vendored app_vc.load_models() to do
the actual model construction (architecture build, checkpoint loading,
vocoder/tokenizer wiring), but monkeypatches its three HuggingFace touch
points — hf_utils.load_custom_model_from_hf (for CAMPPlus and DiT),
BigVGAN's from_pretrained, and Whisper's from_pretrained — to redirect to
local files instead of downloading, falling back to a real fetch for DiT
only, and only if it isn't already sitting in the local folder.
"""

import logging
import os
import shutil
import sys
import types
from argparse import Namespace

import torch

from .utils import (
    get_voice_clone_dir_path,
    get_voice_clone_file_path,
    list_voice_clone_dirs,
    list_voice_clone_files,
    voice_clone_dir,
)

# audio_core/ lives at the package root, not inside nodes/ — see
# generate_core.py for why this needs an explicit sys.path entry.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
from audio_core.offline_utils import allow_network_for_one_fetch

logger = logging.getLogger(__name__)

_VENDOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "seedvc"
)

DIT_FILENAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
DIT_CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DIT_REPO_ID = "Plachta/Seed-VC"


def _ensure_seedvc_on_path():
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)
    if "gradio" not in sys.modules:
        sys.modules["gradio"] = types.ModuleType("gradio")


def _resolve_dit(orig_load_custom_model_from_hf):
    """Local-first, HuggingFace-as-last-resort resolution for the DiT
    checkpoint + its config. Checks <package>/models/voice_clone/ for
    both files first; only contacts HuggingFace if either is missing
    there, and caches whatever it fetches back into that same folder so
    every run after the first is fully local.
    """
    local_model = os.path.join(voice_clone_dir(), DIT_FILENAME)
    local_config = os.path.join(voice_clone_dir(), DIT_CONFIG_FILENAME)

    if os.path.isfile(local_model) and os.path.isfile(local_config):
        logger.info("DiT: using local files in %s", voice_clone_dir())
        return local_model, local_config

    logger.info("DiT not found in %s — fetching from HuggingFace (one-time)...", voice_clone_dir())
    with allow_network_for_one_fetch():
        fetched_model, fetched_config = orig_load_custom_model_from_hf(
            DIT_REPO_ID, DIT_FILENAME, DIT_CONFIG_FILENAME
        )

    # Cache into the local folder so this never needs the network again.
    try:
        if not os.path.isfile(local_model):
            shutil.copy2(fetched_model, local_model)
        if not os.path.isfile(local_config):
            shutil.copy2(fetched_config, local_config)
        logger.info("DiT cached locally in %s", voice_clone_dir())
        return local_model, local_config
    except OSError as e:
        logger.warning("Could not cache DiT into %s (%s) — using HF's own cache instead.",
                        voice_clone_dir(), e)
        return fetched_model, fetched_config


def _load_stack(campplus_path, bigvgan_dir, whisper_dir):
    """Build the SeedVC stack: DiT (local-first, HF-fallback) plus
    CAMPPlus, BigVGAN, and Whisper (all local). Returns the app_vc module
    with its globals populated — that's the object convert_voice() in
    seedvc.py expects as `voice_clone`.

    Note: app_vc is a singleton module — only one loaded stack is
    meaningfully "active" at a time. If a workflow uses two Voice Clone
    Loader nodes with different files, whichever one runs last wins for
    every voice-clone call downstream in that execution. This mirrors how
    the vendored SeedVC code itself is written (module-level globals),
    not a limitation this loader adds on top.
    """
    _ensure_seedvc_on_path()

    original_cwd = os.getcwd()
    os.chdir(_VENDOR_PATH)

    try:
        import hf_utils
        _orig_load_custom_model_from_hf = hf_utils.load_custom_model_from_hf

        def _patched_load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            if repo_id == "funasr/campplus" and model_filename == "campplus_cn_common.bin":
                logger.info("CAMPPlus: using local file %s", campplus_path)
                return campplus_path
            if repo_id == DIT_REPO_ID:
                return _resolve_dit(_orig_load_custom_model_from_hf)
            # Nothing else should reach here, but if the vendored code
            # ever asks for something else this way, don't silently
            # break it — pass through unchanged.
            return _orig_load_custom_model_from_hf(repo_id, model_filename, config_filename)

        hf_utils.load_custom_model_from_hf = _patched_load_custom_model_from_hf

        import modules.bigvgan.bigvgan as bigvgan_mod
        _orig_bigvgan_from_pretrained = bigvgan_mod.BigVGAN.from_pretrained

        @classmethod
        def _patched_bigvgan_from_pretrained(cls, name_or_path=None, **kwargs):
            kwargs.setdefault("use_cuda_kernel", False)
            logger.info("BigVGAN: using local folder %s", bigvgan_dir)
            return _orig_bigvgan_from_pretrained.__func__(cls, bigvgan_dir, **kwargs)

        bigvgan_mod.BigVGAN.from_pretrained = _patched_bigvgan_from_pretrained

        import transformers
        _orig_whisper_from_pretrained = transformers.WhisperModel.from_pretrained
        _orig_feat_extractor_from_pretrained = transformers.AutoFeatureExtractor.from_pretrained

        @classmethod
        def _patched_whisper_from_pretrained(cls, name_or_path=None, **kwargs):
            kwargs.setdefault("local_files_only", True)
            logger.info("Whisper: using local folder %s", whisper_dir)
            return _orig_whisper_from_pretrained.__func__(cls, whisper_dir, **kwargs)

        @classmethod
        def _patched_feat_extractor_from_pretrained(cls, name_or_path=None, **kwargs):
            kwargs.setdefault("local_files_only", True)
            # Whisper's feature extractor warns if sampling_rate isn't passed
            # explicitly, even though it's already correct in the checkpoint's
            # preprocessor_config.json. All Whisper checkpoints (including
            # whisper-small) use 16kHz — passing it explicitly just silences
            # the cosmetic warning, it doesn't change behavior.
            kwargs.setdefault("sampling_rate", 16000)
            return _orig_feat_extractor_from_pretrained.__func__(cls, whisper_dir, **kwargs)

        transformers.WhisperModel.from_pretrained = _patched_whisper_from_pretrained
        transformers.AutoFeatureExtractor.from_pretrained = _patched_feat_extractor_from_pretrained

        import app_vc
        app_vc.device = torch.device("cuda")
        args = Namespace(checkpoint=None, config=None, fp16=True, gpu=0)
        (
            app_vc.model,
            app_vc.semantic_fn,
            app_vc.vocoder_fn,
            app_vc.campplus_model,
            app_vc.to_mel,
            app_vc.mel_fn_args,
        ) = app_vc.load_models(args)

        app_vc.max_context_window = app_vc.sr // app_vc.hop_length * 30
        app_vc.overlap_wave_len = app_vc.overlap_frame_len * app_vc.hop_length

    finally:
        # Restore originals so a second Voice Clone Loader call (e.g. a
        # workflow re-running with different files) doesn't stack patches.
        try:
            hf_utils.load_custom_model_from_hf = _orig_load_custom_model_from_hf
            bigvgan_mod.BigVGAN.from_pretrained = _orig_bigvgan_from_pretrained
            transformers.WhisperModel.from_pretrained = _orig_whisper_from_pretrained
            transformers.AutoFeatureExtractor.from_pretrained = _orig_feat_extractor_from_pretrained
        except NameError:
            pass  # failed before one of the patches was even applied
        os.chdir(original_cwd)

    logger.info("Voice clone stack loaded: sr=%d", app_vc.sr)
    return app_vc


class ScenemaAudioVoiceCloneLoader:
    """Loads the SeedVC voice-conversion stack: CAMPPlus, BigVGAN, and
    Whisper from local files, plus DiT (checked locally first — see this
    module's docstring — HuggingFace is only ever contacted here, and
    only if DiT isn't already sitting in the local folder).
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_VOICE_CLONE",)
    RETURN_NAMES = ("voice_clone",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "campplus_file": (list_voice_clone_files(), {
                    "tooltip": (
                        "campplus_cn_common.bin — speaker embedding model "
                        "(funasr/campplus). Place in "
                        "<ComfyUI-ScenemaAudio package>/models/voice_clone/."
                    ),
                }),
                "bigvgan_dir": (list_voice_clone_dirs(), {
                    "tooltip": (
                        "Local folder with BigVGAN vocoder weights "
                        "(nvidia/bigvgan_v2_22khz_80band_256x — config.json + "
                        "weights). Place under "
                        "<ComfyUI-ScenemaAudio package>/models/voice_clone/<folder>/."
                    ),
                }),
                "whisper_dir": (list_voice_clone_dirs(), {
                    "tooltip": (
                        "Local folder with Whisper speech-tokenizer weights "
                        "(openai/whisper-small — config.json + weights). "
                        "Place under "
                        "<ComfyUI-ScenemaAudio package>/models/voice_clone/<folder>/."
                    ),
                }),
            },
        }

    def load(self, campplus_file, bigvgan_dir, whisper_dir):
        campplus_path = get_voice_clone_file_path(campplus_file)
        bigvgan_path = get_voice_clone_dir_path(bigvgan_dir)
        whisper_path = get_voice_clone_dir_path(whisper_dir)

        app_vc = _load_stack(campplus_path, bigvgan_path, whisper_path)
        return (app_vc,)
