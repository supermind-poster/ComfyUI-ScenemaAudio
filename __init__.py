# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""ComfyUI custom nodes for Scenema Audio.

Expressive text-to-speech with zero-shot voice cloning via
LTX 2.3 audio-only diffusion.

This build is fully local / offline: every checkpoint (transformer, VAE,
Gemma 3 12B IT text encoder) is loaded from ComfyUI's standard models/
folders via dropdown loader nodes, the same way native ComfyUI loaders
work. Nothing here calls out to HuggingFace at runtime. See README for
the one-time manual download instructions for each file.
"""

import hashlib
import os
import shutil
import sys

# Hard-enforce offline mode for huggingface_hub / transformers process-wide,
# before any node or vendored library gets a chance to import them. This is
# a safety net on top of removing all hf_hub_download/snapshot_download
# calls from our own code — some vendored third-party helpers (SeedVC,
# MelBandRoFormer) may still reference huggingface_hub internally, and this
# guarantees they fail fast with a clear offline error instead of silently
# reaching out to the network. Run scripts/prefetch_aux_models.py once
# (see README) to populate their local caches, after which they work fully
# offline like everything else.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Cosmetic: silences a huggingface_hub warning about degraded caching on
# Windows systems without symlink support (Developer Mode off / not admin).
# Harmless either way, just noisy — only relevant to the one legitimate
# HuggingFace fetch left in this package (see nodes/voice_clone_loader.py).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Add vendored packages to sys.path before any node imports.
# ltx_core, ltx_pipelines, seedvc, and mel_band_roformer are vendored
# to avoid external git dependencies and ensure one-click install.
_vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

# Point subprocess calls to "ffmpeg" at the imageio-ffmpeg bundled binary,
# so users don't need a system-wide ffmpeg install. imageio-ffmpeg is a
# pip-installable Python package that ships an ffmpeg binary, declared
# in requirements.txt.
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    if _ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    # pydub caches AudioSegment.converter at import time via shutil.which("ffmpeg").
    # Set it explicitly so it uses the bundled binary even if imported before us.
    try:
        from pydub import AudioSegment
        AudioSegment.converter = _ffmpeg_exe
        AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
except (ImportError, RuntimeError):
    pass

try:
    import folder_paths as _folder_paths

    # Register an extra, self-contained search path so all Scenema Audio
    # files can optionally live together under one folder instead of
    # being scattered across the standard diffusion_models/vae/
    # text_encoders/checkpoints directories. Both layouts work at the same
    # time — this is purely additive, native ComfyUI models in the
    # standard folders are unaffected.
    #
    #   ComfyUI/models/ScenemaAudio/diffusion_models/
    #   ComfyUI/models/ScenemaAudio/vae/
    #   ComfyUI/models/ScenemaAudio/text_encoders/
    #   ComfyUI/models/ScenemaAudio/checkpoints/   (MelBandRoFormer)
    _scenema_root = os.path.join(_folder_paths.models_dir, "ScenemaAudio")
    for _category, _subdir in (
        ("diffusion_models", "diffusion_models"),
        ("vae", "vae"),
        ("text_encoders", "text_encoders"),
        ("checkpoints", "checkpoints"),
    ):
        _path = os.path.join(_scenema_root, _subdir)
        os.makedirs(_path, exist_ok=True)
        _folder_paths.add_model_folder_path(_category, _path)
except Exception:
    # folder_paths not available (e.g. running outside ComfyUI) — the
    # standard-folder layout still works fine, this is just a convenience.
    pass

try:
    import comfy.model_management  # noqa: F401 — probe: are we running inside ComfyUI?
    _running_in_comfyui = True
except ImportError:
    _running_in_comfyui = False

if _running_in_comfyui:
    try:
        from .nodes.model_loader import ScenemaAudioModelLoader
        from .nodes.vae_loader import ScenemaAudioVAELoader
        from .nodes.text_encoder_loader import ScenemaAudioTextEncoderLoader
        from .nodes.text_encode import ScenemaAudioTextEncode
        from .nodes.melband_loader import ScenemaAudioMelBandLoader
        from .nodes.voice_clone_loader import ScenemaAudioVoiceCloneLoader
        from .nodes.vae_encode import ScenemaAudioVAEEncode
        from .nodes.load_audio_url import ScenemaAudioLoadAudioURL
        from .nodes.seedvc import ScenemaAudioVoiceClone
        from .nodes.generate import ScenemaAudioGenerate
        from .nodes.generate_xml import ScenemaAudioGenerateXML
        from .nodes.dialogue_generate import ScenemaAudioDialogueGenerate

        NODE_CLASS_MAPPINGS = {
            "ScenemaAudioModelLoader": ScenemaAudioModelLoader,
            "ScenemaAudioVAELoader": ScenemaAudioVAELoader,
            "ScenemaAudioTextEncoderLoader": ScenemaAudioTextEncoderLoader,
            "ScenemaAudioTextEncode": ScenemaAudioTextEncode,
            "ScenemaAudioMelBandLoader": ScenemaAudioMelBandLoader,
            "ScenemaAudioVoiceCloneLoader": ScenemaAudioVoiceCloneLoader,
            "ScenemaAudioVAEEncode": ScenemaAudioVAEEncode,
            "ScenemaAudioLoadAudioURL": ScenemaAudioLoadAudioURL,
            "ScenemaAudioVoiceClone": ScenemaAudioVoiceClone,
            "ScenemaAudioGenerate": ScenemaAudioGenerate,
            "ScenemaAudioGenerateXML": ScenemaAudioGenerateXML,
            "ScenemaAudioDialogueGenerate": ScenemaAudioDialogueGenerate,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            "ScenemaAudioModelLoader": "Scenema Audio Diffusion Loader",
            "ScenemaAudioVAELoader": "Scenema Audio VAE Loader",
            "ScenemaAudioTextEncoderLoader": "Scenema Audio Text Encoder Loader (Gemma 3)",
            "ScenemaAudioTextEncode": "Scenema Audio Text Encode",
            "ScenemaAudioMelBandLoader": "Scenema Audio MelBandRoFormer Loader",
            "ScenemaAudioVoiceCloneLoader": "Scenema Audio Voice Clone Loader",
            "ScenemaAudioVAEEncode": "Scenema Audio VAE Encode",
            "ScenemaAudioLoadAudioURL": "Scenema Audio Load Audio from URL",
            "ScenemaAudioVoiceClone": "Scenema Audio Voice Clone",
            "ScenemaAudioGenerate": "Scenema Audio Generate",
            "ScenemaAudioGenerateXML": "Scenema Audio Generate (XML)",
            "ScenemaAudioDialogueGenerate": "Scenema Audio Dialogue Generate",
        }

        # JS extensions live in web/. Powers the preset dropdown's auto-fill.
        WEB_DIRECTORY = "./web"

        # Copy shipped workflow files into ComfyUI's user workflows directory so
        # they appear in the Workflows sidebar without the user having to drag
        # JSON files onto the canvas. Overwrites when the shipped file's content
        # hash differs from the destination — this way package updates always
        # ship the latest workflow, but restarts with no shipped changes leave
        # the destination alone. Users who want to preserve customizations
        # should Save As under a different name.
        def _install_workflows():
            try:
                import folder_paths
                pkg_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(pkg_dir, "workflows")
                if not os.path.isdir(src_dir):
                    return
                user_dir = getattr(folder_paths, "user_directory", None) \
                    or folder_paths.get_user_directory()
                dest_dir = os.path.join(user_dir, "default", "workflows", "Scenema Audio")
                os.makedirs(dest_dir, exist_ok=True)
                for fn in os.listdir(src_dir):
                    if not fn.endswith(".json"):
                        continue
                    src = os.path.join(src_dir, fn)
                    dest = os.path.join(dest_dir, fn)
                    src_hash = hashlib.sha256(open(src, "rb").read()).hexdigest()
                    if os.path.exists(dest):
                        dest_hash = hashlib.sha256(open(dest, "rb").read()).hexdigest()
                        if src_hash == dest_hash:
                            continue
                    shutil.copy2(src, dest)
            except Exception:
                pass

        _install_workflows()

        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

    except Exception:
        # Import failed — print the full traceback so the real cause (a
        # missing dependency, a bug, etc.) is visible in the ComfyUI console
        # instead of just the generic "no NODE_CLASS_MAPPINGS" warning. Define
        # empty mappings so ComfyUI can still start up cleanly around us.
        import traceback
        print("[ComfyUI-ScenemaAudio] Failed to load nodes:")
        traceback.print_exc()
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # comfy isn't importable — we're running outside ComfyUI (e.g. pytest).
    # Nothing to register; tests import from nodes.* / audio_core.* directly.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
