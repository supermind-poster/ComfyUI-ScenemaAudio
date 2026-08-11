# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Shared utilities for Scenema Audio ComfyUI nodes.

INT8 linear layer, audio-only transformer patch, and local (offline)
model file resolution. No HuggingFace network calls happen anywhere in
this module — every checkpoint is expected to already exist on disk
under ComfyUI's standard `models/` folders, exactly like every other
native ComfyUI loader (Load Diffusion Model, Load VAE, etc).

Expected files (place manually, see README for download links):
    ComfyUI/models/diffusion_models/scenema-audio-transformer.safetensors
    ComfyUI/models/diffusion_models/scenema-audio-transformer-int8.safetensors
    ComfyUI/models/diffusion_models/scenema-audio-pipeline.safetensors
    ComfyUI/models/vae/scenema-audio-pipeline-audio.safetensors
    ComfyUI/models/vae/scenema-audio-vae-encoder.safetensors
    ComfyUI/models/text_encoders/gemma-3-12b-it/  (config.json, tokenizer*, *.safetensors)
"""

import gc
import json
import logging
import os

import torch
from ltx_core.batch_split import BatchedPerturbationConfig
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio
from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, rms_norm
from safetensors import safe_open
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

FPS = 24
MAX_REF_SECONDS = 5

# Expected filenames, kept only as documentation / sanity-check hints in
# the UI tooltips. Nothing in this module downloads them automatically.
TRANSFORMER_BF16 = "scenema-audio-transformer.safetensors"
TRANSFORMER_INT8 = "scenema-audio-transformer-int8.safetensors"
PIPELINE_CKPT = "scenema-audio-pipeline.safetensors"
PIPELINE_AUDIO_CKPT = "scenema-audio-pipeline-audio.safetensors"
VAE_ENCODER_CKPT = "scenema-audio-vae-encoder.safetensors"


# ── Local model discovery (folder_paths-based, like native Comfy loaders) ──


def _folder_paths():
    import folder_paths  # available inside ComfyUI's runtime only
    return folder_paths


def _base_dirs_for(category, fallback_subdir):
    """Return the list of directories ComfyUI knows about for `category`,
    falling back to `models/<fallback_subdir>` if the category was never
    registered (older ComfyUI versions).
    """
    fp = _folder_paths()
    try:
        dirs = fp.get_folder_paths(category)
        if dirs:
            return dirs
    except Exception:
        pass
    return [os.path.join(fp.models_dir, fallback_subdir)]


def list_diffusion_models():
    """Files under models/diffusion_models — transformer + pipeline ckpts."""
    fp = _folder_paths()
    try:
        files = fp.get_filename_list("diffusion_models")
    except Exception:
        files = []
    return files or ["<no files found in models/diffusion_models>"]


def get_diffusion_model_path(filename):
    fp = _folder_paths()
    path = fp.get_full_path("diffusion_models", filename)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{filename}' not found in any models/diffusion_models folder. "
            f"Place the .safetensors file there and reload the node list."
        )
    return path


def list_vae_files():
    """Files under models/vae — audio VAE decoder + encoder ckpts."""
    fp = _folder_paths()
    try:
        files = fp.get_filename_list("vae")
    except Exception:
        files = []
    return files or ["<no files found in models/vae>"]


def get_vae_path(filename):
    fp = _folder_paths()
    path = fp.get_full_path("vae", filename)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{filename}' not found in any models/vae folder. "
            f"Place the .safetensors file there and reload the node list."
        )
    return path


def list_checkpoint_files():
    """Files under models/checkpoints — used here for MelBandRoFormer.

    Reuses ComfyUI's native "checkpoints" category (same folder as full
    SD/etc. checkpoints), so this dropdown will list those too — just
    pick the MelBandRoFormer one.
    """
    fp = _folder_paths()
    try:
        files = fp.get_filename_list("checkpoints")
    except Exception:
        files = []
    return files or ["<no files found in models/checkpoints>"]


def get_checkpoint_path(filename):
    fp = _folder_paths()
    path = fp.get_full_path("checkpoints", filename)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{filename}' not found in any models/checkpoints folder. "
            f"Place the file there and reload the node list."
        )
    return path


def list_text_encoder_dirs():
    """Subdirectories of models/text_encoders that look like a local HF
    model folder (contain config.json). Gemma 3 12B IT ships this way.
    """
    dirs = []
    for base in _base_dirs_for("text_encoders", "text_encoders"):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                dirs.append(name)
    return dirs or ["<no folders found in models/text_encoders>"]


def get_text_encoder_path(name):
    for base in _base_dirs_for("text_encoders", "text_encoders"):
        full = os.path.join(base, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            return full
    raise FileNotFoundError(
        f"Text encoder folder '{name}' not found under models/text_encoders. "
        f"It must contain config.json, tokenizer files, and *.safetensors shards "
        f"for google/gemma-3-12b-it, downloaded once and placed there manually."
    )


def find_local_model(filename, categories=("vae", "diffusion_models")):
    """Search known Comfy model folders for an exact filename, across
    several categories. Used by internal (non user-facing) auxiliary
    loaders that need a single fixed checkpoint rather than a dropdown.
    """
    fp = _folder_paths()
    for cat in categories:
        try:
            path = fp.get_full_path(cat, filename)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            return path
    searched = ", ".join(f"models/{c}" for c in categories)
    raise FileNotFoundError(
        f"'{filename}' not found. Searched: {searched}. Place the file in one "
        f"of these folders."
    )


# ── Package-local model folders (NOT under ComfyUI/models) ─────────────
#
# The voice-clone helper checkpoints belong to this node specifically
# rather than to the shared ComfyUI model library — they aren't the kind
# of thing another custom node would ever want to pick up from a shared
# models/vae or models/text_encoders folder. They live directly inside
# this package instead:
#
#   <this package>/models/voice_clone/campplus_cn_common.bin
#   <this package>/models/voice_clone/bigvgan/          (folder)
#   <this package>/models/voice_clone/whisper-small/    (folder)
#
# MelBandRoFormer is different — it uses ComfyUI's shared
# models/checkpoints/ folder instead (see list_checkpoint_files above),
# since it's a standalone checkpoint someone might reasonably want to
# manage the same way as other checkpoints.
#
# Nothing here is auto-downloaded — see scripts/prefetch_aux_models.py
# for the one convenience exception (CAMPPlus/BigVGAN/Whisper only).


def _pkg_root():
    """Root of this package (parent of nodes/, sibling of vendor/, web/, etc.)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pkg_local_dir(*parts):
    path = os.path.join(_pkg_root(), "models", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def voice_clone_dir():
    """Public accessor for <this package>/models/voice_clone/ — used by
    Scenema Audio Voice Clone Loader to check for a locally-placed DiT
    checkpoint before ever considering HuggingFace as a fallback.
    """
    return _pkg_local_dir("voice_clone")


def list_voice_clone_files():
    """Single-file checkpoints under <this package>/models/voice_clone/
    (currently just CAMPPlus)."""
    folder = _pkg_local_dir("voice_clone")
    exts = (".safetensors", ".bin", ".pth", ".ckpt")
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    )
    return files or ["<no files found in custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone>"]


def get_voice_clone_file_path(filename):
    folder = _pkg_local_dir("voice_clone")
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{filename}' not found in {folder}.")
    return path


def list_voice_clone_dirs():
    """Subdirectories of <this package>/models/voice_clone/ that look like
    a local HF model folder (contain config.json) — BigVGAN and Whisper
    ship this way.
    """
    folder = _pkg_local_dir("voice_clone")
    dirs = sorted(
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
        and os.path.isfile(os.path.join(folder, name, "config.json"))
    )
    return dirs or ["<no folders found in custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone>"]


def get_voice_clone_dir_path(name):
    folder = _pkg_local_dir("voice_clone")
    path = os.path.join(folder, name)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return path
    raise FileNotFoundError(f"Folder '{name}' not found under {folder}.")


# ── INT8 Linear ────────────────────────────────────────────────────────


class Int8Linear(torch.nn.Module):
    """Linear layer with INT8 weights, dequantized to input dtype during forward."""

    def __init__(self, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_parameter("bias", torch.nn.Parameter(bias))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


# ── Audio-Only Forward Patch ───────────────────────────────────────────


def audio_only_forward(self, video, audio, perturbations=None):
    """Monkey-patched forward for audio-only transformer blocks.

    Skips all video computation and only runs audio self-attention,
    cross-attention, and feedforward.
    """
    if video is None and audio is None:
        raise ValueError("Need at least one modality")
    batch_size = (video or audio).x.shape[0]
    if perturbations is None:
        perturbations = BatchedPerturbationConfig.empty(batch_size)
    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None
    run_ax = audio is not None and audio.enabled and ax.numel() > 0
    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
        )
        norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        ax = (
            ax
            + self.audio_attn1(
                norm_ax, pe=audio.positional_embeddings, mask=audio.self_attention_mask
            )
            * agate_msa
        )
        del agate_msa, norm_ax
        ax = ax + self._apply_text_cross_attention(
            ax,
            audio.context,
            self.audio_attn2,
            self.audio_scale_shift_table,
            getattr(self, "audio_prompt_scale_shift_table", None),
            audio.timesteps,
            audio.prompt_timestep,
            audio.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        ashift_ff, ascale_ff, agate_ff = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
        )
        norm_ax_ff = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_ff) + ashift_ff
        del ashift_ff, ascale_ff
        ax = ax + self.audio_ff(norm_ax_ff) * agate_ff
        del agate_ff, norm_ax_ff
    if video is not None:
        object.__setattr__(video, "x", vx)
    if audio is not None:
        object.__setattr__(audio, "x", ax)
    return video, audio


# ── Meta Tensor Materialization ────────────────────────────────────────


def materialize_meta_tensors(module, device="cpu"):
    """Replace meta tensors with zeros on the specified device."""
    for name, param in list(module.named_parameters()):
        if param.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._parameters[parts[-1]] = torch.nn.Parameter(
                torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
            )
    for name, buf in list(module.named_buffers()):
        if buf.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = torch.zeros(
                buf.shape, dtype=torch.bfloat16, device=device
            )


# ── Model Loading (all local, no network) ──────────────────────────────


def _detect_int8_keys(state_dict):
    """Detect INT8 checkpoint format and return key mappings."""
    int8_map = {
        k.replace(".weight.int8", ""): k for k in state_dict if k.endswith(".weight.int8")
    }
    scale_map = {
        k.replace(".weight.scale", ""): k for k in state_dict if k.endswith(".weight.scale")
    }
    return int8_map, scale_map


def _load_int8_weights(mdl_wrapper, state_dict, int8_map, scale_map):
    """Load INT8 quantized weights into the model."""
    regular_sd = {
        k: v for k, v in state_dict.items()
        if not k.endswith(".int8") and not k.endswith(".scale")
    }
    mdl_wrapper.load_state_dict(regular_sd, strict=False, assign=True)

    n_replaced = 0
    for name in int8_map:
        w_int8 = state_dict[int8_map[name]]
        w_scale = state_dict[scale_map[name]]
        parts = name.split(".")
        parent = mdl_wrapper
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        bias_key = name + ".bias"
        bias = state_dict.get(bias_key)
        if bias is None and hasattr(old, "bias") and old.bias is not None:
            bias = old.bias.data
        setattr(parent, parts[-1], Int8Linear(w_int8, w_scale, bias))
        n_replaced += 1

    logger.info("INT8: replaced %d Linear layers", n_replaced)


def _nuke_video_paths(mdl):
    """Replace video computation paths with Identity modules."""
    for block in mdl.transformer_blocks:
        block.attn1 = torch.nn.Identity()
        block.attn2 = torch.nn.Identity()
        block.ff = torch.nn.Identity()
        block.audio_to_video_attn = torch.nn.Identity()
    gc.collect()


def load_transformer(checkpoint_path):
    """Load the audio-only transformer from a local safetensors checkpoint.

    Handles both bf16 and INT8 quantized checkpoints. Applies the
    audio-only forward patch and nukes video computation paths.
    Purely local — `checkpoint_path` must already exist on disk.

    Returns:
        Tuple of (X0Model wrapper, config dict).
    """
    with safe_open(checkpoint_path, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    with torch.device("meta"):
        mdl = LTXModelConfigurator.from_config(config)

    sd = load_file(checkpoint_path, device="cpu")
    int8_map, scale_map = _detect_int8_keys(sd)
    is_int8 = len(int8_map) > 0

    mdl_wrapper = X0Model(mdl)

    if is_int8:
        _load_int8_weights(mdl_wrapper, sd, int8_map, scale_map)
    else:
        mdl_wrapper.load_state_dict(sd, strict=False, assign=True)

    del sd
    gc.collect()

    _nuke_video_paths(mdl)
    materialize_meta_tensors(mdl_wrapper)

    cross_pe = max(
        mdl.positional_embedding_max_pos[0],
        mdl.audio_positional_embedding_max_pos[0],
    )
    mdl._init_preprocessors(cross_pe)

    BasicAVTransformerBlock.forward = audio_only_forward

    return mdl_wrapper.eval(), config


def load_vae_encoder(config, checkpoint_path):
    """Load the Audio VAE encoder from a standalone local checkpoint.

    Returns:
        Tuple of (encoder module, sample_rate).
    """
    avae_cfg = config["audio_vae"]
    preproc = avae_cfg["preprocessing"]
    vae_sr = preproc["audio"]["sampling_rate"]

    with torch.device("meta"):
        encoder = AudioEncoderConfigurator().from_config(avae_cfg)

    sd = load_file(checkpoint_path, device="cpu")
    encoder.load_state_dict(sd, strict=False, assign=True)

    pcs = encoder.per_channel_statistics
    if "per_channel_statistics.std-of-means" in sd:
        pcs._buffers["std-of-means"] = sd["per_channel_statistics.std-of-means"]
        pcs._buffers["mean-of-means"] = sd["per_channel_statistics.mean-of-means"]
    del sd

    dd = avae_cfg["model"]["params"]["ddconfig"]
    encoder.mel_bins = dd["mel_bins"]
    encoder.mid.attn_1 = torch.nn.Identity()

    materialize_meta_tensors(encoder, device="cpu")

    return encoder.eval().to(torch.bfloat16), vae_sr


def extract_wav(audio_obj):
    """Extract numpy waveform from an LTX Audio object."""
    w = audio_obj.waveform.cpu().float().numpy()
    if w.ndim == 3:
        w = w.squeeze(0)
    if w.ndim == 2:
        w = w.T
    return w, audio_obj.sampling_rate
