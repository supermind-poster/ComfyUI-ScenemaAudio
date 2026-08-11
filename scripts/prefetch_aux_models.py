# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""One-time, manual-if-you-want-it prefetch for Scenema Audio Voice Clone
Loader's local helper files: CAMPPlus, BigVGAN, Whisper, and (optionally)
the DiT checkpoint. None of these are gated, so this script is a
convenience, not a requirement — grab them by hand instead with
`huggingface-cli download` (see README) if you'd rather.

Everything this script fetches goes directly inside this package's own
models/ folder, NOT into ComfyUI/models — these are helper checkpoints
specific to this node's SeedVC integration, not general-purpose models:

    <this package>/models/voice_clone/campplus_cn_common.bin
    <this package>/models/voice_clone/bigvgan/
    <this package>/models/voice_clone/whisper-small/
    <this package>/models/voice_clone/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth  (optional)
    <this package>/models/voice_clone/config_dit_mel_seed_uvit_whisper_small_wavenet.yml           (optional)

Scenema Audio Voice Clone Loader is the ONLY node in this whole package
that ever talks to HuggingFace, and even it only as a last resort: it
checks for DiT in that same local folder first, and only reaches for the
network if it's missing there. Running this script's DiT fetch ahead of
time means the very first Voice Clone Loader run never touches the
network at all — otherwise it'll fetch DiT itself, once, the first time
you use it.

This does NOT touch:
  - MelBandRoFormer. That one is deliberately manual-only — download it
    yourself (see README) and place it in ComfyUI/models/checkpoints/.
    Nothing in this package fetches it automatically, including this
    script.

Note: validate=True and the !INTERRUPTION! marker (Dialogue Generate)
reuse the whisper-small/ folder this script already fetches above for
Voice Clone Loader — nothing extra needed for them.

Run this ONCE, with network access, after installing the node:

    python scripts/prefetch_aux_models.py

After this, place the model files listed in the README (Scenema Audio
transformer/pipeline/VAE checkpoints, Gemma 3 12B IT, MelBandRoFormer)
and every loader node in the package works fully offline.
"""

import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOICE_CLONE_DIR = os.path.join(_PKG_ROOT, "models", "voice_clone")

DIT_FILENAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
DIT_CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DIT_REPO_ID = "Plachta/Seed-VC"


def _download(repo_id, filename, dest_dir):
    from huggingface_hub import hf_hub_download

    print(f"Fetching {filename} ({repo_id})...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        import shutil
        shutil.copy2(cached_path, dest_path)
    print(f"  -> {dest_path}")


def _download_folder(repo_id, dest_dir):
    from huggingface_hub import snapshot_download

    print(f"Fetching {repo_id} (full folder)...")
    os.makedirs(dest_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=dest_dir)
    print(f"  -> {dest_dir}")


if __name__ == "__main__":
    print(f"Downloading into: {_VOICE_CLONE_DIR}\n")

    # -> Scenema Audio Voice Clone Loader (campplus_file / bigvgan_dir / whisper_dir)
    _download("funasr/campplus", "campplus_cn_common.bin", _VOICE_CLONE_DIR)
    _download_folder("nvidia/bigvgan_v2_22khz_80band_256x",
                      os.path.join(_VOICE_CLONE_DIR, "bigvgan"))
    _download_folder("openai/whisper-small",
                      os.path.join(_VOICE_CLONE_DIR, "whisper-small"))

    # Optional: pre-fetch DiT into the same local-first location the
    # loader itself checks, so its very first run needs no network at all.
    _download(DIT_REPO_ID, DIT_FILENAME, _VOICE_CLONE_DIR)
    _download(DIT_REPO_ID, DIT_CONFIG_FILENAME, _VOICE_CLONE_DIR)

    print("\nDone. CAMPPlus, BigVGAN, Whisper, and DiT are now local, inside this package.")
    print("Voice Clone Loader will not need the network at all from here on.")
    print("MelBandRoFormer is NOT fetched by this script — download it yourself and")
    print("place it in ComfyUI/models/checkpoints/ (see README).")
    print("validate=True and !INTERRUPTION! reuse the whisper-small/ folder fetched")
    print("above — nothing extra needed for them.")
    print("See the README for the remaining Scenema Audio + Gemma 3 12B IT downloads.")

