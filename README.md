# ComfyUI-ScenemaAudio (local / offline fork)

Native ComfyUI nodes for [Scenema Audio](https://scenema.ai/audio). Expressive text-to-speech with zero-shot voice cloning, built on the LTX 2.3 audio diffusion transformer.

This is not just TTS. It is real vocal performance. Laughs, whispers, voice cracks, breath, singing, foreign accents. All driven by prompt cues.

**This fork removes automatic HuggingFace downloading almost everywhere.** Every model-consuming node requires an explicit loader connection — transformer, VAE, Gemma 3 12B IT text encoder, MelBandRoFormer, and the CAMPPlus/BigVGAN/Whisper trio behind voice cloning all load from local files you place under ComfyUI's standard `models/` folders, picked from a dropdown — the same UX as ComfyUI's native "Load Diffusion Model" / "Load VAE" nodes. `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are forced process-wide the moment the package loads, as a hard guarantee. **Scenema Audio Voice Clone Loader is the only node in this package that ever talks to HuggingFace**, and even there only as a last resort — it checks locally first, and only fetches (once, caching the result locally) if the file genuinely isn't there. See "Local-first, HuggingFace-as-last-resort" below.

## Quick start

1. Install the node (see below).
2. Download the model files once, manually (see **Model files** below), and place them in the folders listed.
3. Load `workflows/scenema_audio.json`, connect the three loader nodes, pick a preset, hit Queue.

## Installation

### ComfyUI Manager (recommended)

Open the Manager tab. Choose **Install via Git URL**. Paste this URL, then restart ComfyUI.

### Manual

```
cd ComfyUI/custom_nodes
git clone <this-repo-url>.git ComfyUI-ScenemaAudio
pip install -r ComfyUI-ScenemaAudio/requirements.txt
```

Restart ComfyUI. **Nothing downloads automatically.** You need to fetch the model files yourself once — see below.

## Model files (download once, place manually)

Every model in this package loads through a dedicated loader node — dropdown from a local file/folder. There's a split in *where* files live, by design:

- **Core pipeline models** (transformer, VAE, Gemma 3 12B IT) and **MelBandRoFormer** go under ComfyUI's shared `models/` folders — same dropdown UX as native "Load Diffusion Model" / "Load VAE" / "Load CLIP", so they show up next to your other models. MelBandRoFormer specifically uses `models/checkpoints/`.
- **Voice-clone helper models** (CAMPPlus, BigVGAN, Whisper) are specific to this node's SeedVC integration, not general-purpose models another custom node would ever look for — so they live directly inside this package's own folder instead, not in `ComfyUI/models` at all.
- **One exception**: the SeedVC DiT checkpoint (~98M params, not gated, ~150MB) can be placed locally like everything else, or left alone — the Voice Clone Loader checks locally first and only fetches from HuggingFace if it's missing, caching it locally afterward. See "Local-first, HuggingFace-as-last-resort" below.

### Core pipeline models + MelBandRoFormer — under ComfyUI/models

Two layout options — pick whichever you prefer, both work at once:

**Option A — standard ComfyUI folders** (mixes in with your other models):

| File | Where to put it | Size | Source | Loader node |
|---|---|---|---|---|
| `scenema-audio-transformer-int8.safetensors` **or** `-bf16` | `ComfyUI/models/diffusion_models/` | ~4.9 GB / ~9.8 GB | [ScenemaAI/scenema-audio](https://huggingface.co/ScenemaAI/scenema-audio) | Diffusion Loader |
| `scenema-audio-pipeline.safetensors` | `ComfyUI/models/diffusion_models/` | — | same repo | Text Encoder Loader |
| `scenema-audio-pipeline-audio.safetensors` | `ComfyUI/models/vae/` | ~6.7 GB | same repo | VAE Loader |
| `scenema-audio-vae-encoder.safetensors` | `ComfyUI/models/vae/` | ~42.7 MB | same repo | VAE Loader |
| Gemma 3 12B IT (folder: `config.json`, tokenizer, `*.safetensors` shards) | `ComfyUI/models/text_encoders/gemma-3-12b-it/` | ~24 GB | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (gated) | Text Encoder Loader |
| `MelBandRoformer_fp16.safetensors` | `ComfyUI/models/checkpoints/` | ~280 MB | [Kijai/MelBandRoFormer_comfy](https://huggingface.co/Kijai/MelBandRoFormer_comfy) | MelBandRoFormer Loader — **manual only, see note below** |

**Option B — everything under one folder** (kept separate from the rest of your models): same files, just under `ComfyUI/models/ScenemaAudio/{diffusion_models,vae,text_encoders,checkpoints}/` instead of the top-level `models/` subfolders. The package registers that folder as an additional search path for all four categories at startup, so dropdowns see both layouts at once — mix and match freely.

```bash
# Scenema Audio checkpoints (not gated)
huggingface-cli download ScenemaAI/scenema-audio scenema-audio-transformer-int8.safetensors \
    --local-dir ComfyUI/models/diffusion_models
huggingface-cli download ScenemaAI/scenema-audio scenema-audio-pipeline.safetensors \
    --local-dir ComfyUI/models/diffusion_models
huggingface-cli download ScenemaAI/scenema-audio scenema-audio-pipeline-audio.safetensors \
    --local-dir ComfyUI/models/vae
huggingface-cli download ScenemaAI/scenema-audio scenema-audio-vae-encoder.safetensors \
    --local-dir ComfyUI/models/vae

# Gemma 3 12B IT — gated, requires accepting the license on the model page first,
# then a token: huggingface-cli login  (or) export HF_TOKEN=hf_...
huggingface-cli download google/gemma-3-12b-it \
    --local-dir ComfyUI/models/text_encoders/gemma-3-12b-it

# MelBandRoFormer — not gated, but deliberately not fetched by any script in
# this package (see note below) — grab it yourself:
huggingface-cli download Kijai/MelBandRoFormer_comfy MelBandRoformer_fp16.safetensors \
    --local-dir ComfyUI/models/checkpoints
```

**MelBandRoFormer is manual-only, deliberately.** It's not fetched by `scripts/prefetch_aux_models.py` or anything else in this package — download it yourself and drop it in `ComfyUI/models/checkpoints/` (or the Option B equivalent). This is the one model in the whole package without any auto-fetch convenience, by design. The dropdown reads the same `checkpoints` category ComfyUI's native checkpoint loaders use, so it'll list your other checkpoints too — just pick the MelBandRoFormer one.

### Voice-clone helper models — inside this package, NOT ComfyUI/models

CAMPPlus, BigVGAN, and Whisper are specific to this node's SeedVC integration, not general-purpose models — so they live under `ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone/` instead, a folder that ships with this package (empty, with a placeholder `.txt` explaining what goes where). They will never show up mixed in with your other ComfyUI models.

| File | Where to put it | Size | Source | Loader node |
|---|---|---|---|---|
| `campplus_cn_common.bin` | `<package>/models/voice_clone/` | ~28 MB | [funasr/campplus](https://huggingface.co/funasr/campplus) | Voice Clone Loader |
| BigVGAN folder | `<package>/models/voice_clone/bigvgan/` | ~0.5 GB | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) | Voice Clone Loader |
| Whisper-small folder | `<package>/models/voice_clone/whisper-small/` | ~1 GB | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Voice Clone Loader |
| `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + `config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | `<package>/models/voice_clone/` (optional — auto-fetched here if missing) | ~150 MB | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC) | Voice Clone Loader |

None of these three are gated, so `scripts/prefetch_aux_models.py` can fetch them for convenience:

```bash
python ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/scripts/prefetch_aux_models.py
```

...or grab them by hand:

```bash
huggingface-cli download funasr/campplus campplus_cn_common.bin \
    --local-dir ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone
huggingface-cli download nvidia/bigvgan_v2_22khz_80band_256x \
    --local-dir ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone/bigvgan
huggingface-cli download openai/whisper-small \
    --local-dir ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone/whisper-small
```

That's it — after all of the above, disconnect from the network entirely if you want; every loader node keeps working from what's on disk.

### About the Gemma 3 12B IT format

Gemma 3 12B IT ships from Google as a standard sharded-`safetensors` HF model folder. `safetensors` already memory-maps weights lazily (it doesn't read the whole file into RAM up front), and `transformers.Gemma3ForConditionalGeneration.from_pretrained(local_dir, local_files_only=True)` uses exactly that mechanism when pointed at a local folder — so "lazy safetensors loading" is already what happens here, no conversion or special handling needed. The only network dependency was ever the *first* download, because Google's repo is gated; once the folder exists locally, this node never touches HuggingFace for it again.

There is currently no way to load Gemma 3 12B IT through ComfyUI's native `CLIPLoader`/`DualCLIPLoader` — those expect a single-file text-encoder checkpoint with known key prefixes (T5, Gemma2 for Lumina2, etc.), while Gemma 3 12B IT here is loaded as the full `Gemma3ForConditionalGeneration` class from `transformers`. That's why this fork ships its own **Scenema Audio Text Encoder Loader** node instead of trying to force it through the native loader.

### Local-first, HuggingFace-as-last-resort (Voice Clone Loader only)

Every model-consuming node in this package requires an explicit loader connection and stays fully local — with exactly one exception, and even that one checks locally before ever considering the network.

**The SeedVC DiT checkpoint** (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + its config yml, ~98M parameters, not gated) is the only thing in this entire package that can reach HuggingFace, and only inside **Scenema Audio Voice Clone Loader**:

1. It checks `<this package>/models/voice_clone/` for both files first — same folder as CAMPPlus/BigVGAN/Whisper.
2. If they're already there (placed by hand, or by `scripts/prefetch_aux_models.py`), it uses them — no network touched at all.
3. Only if they're genuinely missing does it fetch from `Plachta/Seed-VC` — lifting the process-wide offline guard (`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`) for just that one call — then caches the result back into that same local folder, so every run after the first is local too.

Want zero network dependency even on the very first run? Place `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` and `config_dit_mel_seed_uvit_whisper_small_wavenet.yml` in `<this package>/models/voice_clone/` yourself (or run the prefetch script, which now grabs these two as well) — the loader will never have a reason to check HuggingFace at all.

### whisper-small for validate= / !INTERRUPTION! — reuses Voice Clone Loader's own whisper-small, no separate model

`validate=True` (word-match retry) and `<speak_interruption>`'s `!INTERRUPTION!` marker both need Whisper for transcription and word-level timing. Rather than bringing in a second copy of whisper-small in a different format, this reuses the exact same local checkpoint Voice Clone Loader already loads for SeedVC's speech tokenizer — `<this package>/models/voice_clone/whisper-small/` — through the same `transformers` library. If you've already set that up per the Model files section above, `validate=True` and `!INTERRUPTION!` just work; nothing extra to download. Since this isn't the Voice Clone Loader node, it follows the strict rule regardless: **fully local, no HuggingFace fallback of any kind**. If no local Whisper checkpoint is found anywhere under `models/voice_clone/`, both features fail gracefully with a clear message rather than reaching for the network — `validate=True` raises, and the `!INTERRUPTION!` marker just logs a warning and falls back to `dif_param`.
## Hardware requirements

Minimum is 8 GB VRAM. Tested end to end on RTX 3070 (8 GB) and RTX 4090 (24 GB). Generation runs up to 2x realtime.

Also needs around 32 GB system RAM for the pipeline components.

Gemma 3 12B IT encoding mode is picked automatically from VRAM (override via the loader's `precision` input):

| VRAM | Mode | Extra VRAM used |
|---|---|---|
| 40 GB+ | `bf16_gpu` | ~24 GB |
| 12–39 GB | `nf4` | ~8 GB |
| < 12 GB | `cpu_streaming` | minimal, slower |

## Nodes

Twelve user-facing nodes. Every model-consuming node requires an explicit loader connection now — no node loads or auto-downloads anything on its own. Scenema Audio Voice Clone Loader is the only node that can ever reach HuggingFace, and only as a last resort after checking locally first — see "Local-first, HuggingFace-as-last-resort" above.

| Node | Purpose |
|---|---|
| **Scenema Audio Diffusion Loader** | Loads the 3.3B audio transformer from a local file — dropdown reads `models/diffusion_models/`, same as native "Load Diffusion Model". |
| **Scenema Audio VAE Loader** | Loads the audio VAE encoder + decoder — dropdowns read `models/vae/`, same as native "Load VAE". |
| **Scenema Audio Text Encoder Loader (Gemma 3)** | Loads Gemma 3 12B IT + the pipeline checkpoint, fully local. Dropdown for the Gemma folder reads `models/text_encoders/`; dropdown for the pipeline file reads `models/diffusion_models/`. |
| **Scenema Audio MelBandRoFormer Loader** | Loads the vocal/background separator used for background-SFX stripping. Dropdown reads `models/checkpoints/` (same category as native checkpoint loaders) — manual-only, no auto-fetch anywhere, not even from the prefetch script. Connect its output to `separator` on any Generate node — required whenever `strip_background_sfx` actually strips (auto/yes); the node raises a clear error instead of silently loading its own copy if it's needed and missing. |
| **Scenema Audio Voice Clone Loader** | Loads the SeedVC stack (CAMPPlus, BigVGAN, Whisper, DiT — all read from `<package>/models/voice_clone/`, not `ComfyUI/models`) used for cross-chunk voice consistency and standalone voice conversion. The only node in this package that can ever reach HuggingFace — and only for DiT, and only if it's not already sitting in that local folder (see "Local-first, HuggingFace-as-last-resort" above). Connect its output to `voice_clone` on any Generate node — required whenever the cross-chunk voice-lock pass actually runs (multiple chunks, `skip_vc=False`). |
| **Scenema Audio Text Encode** | Standalone prompt → conditioning node, if you want to encode ahead of time or reuse conditioning across generations. |
| **Scenema Audio Generate** | The main node, friendly widget mode. Takes `model`, `vae`, and `text_encoder` inputs plus voice description, text, and scene. Includes a preset dropdown with 12 curated voices. |
| **Scenema Audio Generate (XML)** | Same engine as Generate, but takes a raw `<speak voice="..." scene="..." ...>...</speak>` prompt verbatim — the base model's native XML format, byte-for-byte, no widget layer in between. Use this if you already have prompts in Scenema's own format. |
| **Scenema Audio Dialogue Generate** | Takes a `<dialogue>` of `<speak>` / `<speak_overlap>` / `<speak_interruption>` turns and produces one continuous multi-speaker clip. Declare recurring characters once with `<speaker id="X" voice="..." .../>` and reference them with `<speak speaker="X">` — or write full inline `<speak voice="...">` per turn like before, both styles mix freely. `<speak_overlap>` layers several speakers' lines to start simultaneously (unison or different lines); `<speak_interruption>` has one party cut into the tail of another's line, splice point controlled by `dif_param="15"` (percent) or a precise `!INTERRUPTION!` marker inside the text. Each speaker's own voice-continuity reference tracks independently regardless of who else spoke in between. Optional `ref_speaker_1`, `ref_speaker_2`, ... inputs voice-clone each character from their very first line — connect one and the next slot appears automatically. |
| **Scenema Audio VAE Encode** | Encodes reference audio to a latent for voice cloning. Capped at 20 seconds. |
| **Scenema Audio Load Audio from URL** | Loads a reference audio file from a URL (mp3, wav, flac) for voice cloning. Alternative to the built-in LoadAudio. |
| **Scenema Audio Voice Clone** | Standalone SeedVC voice conversion. Requires a `voice_clone` input from Scenema Audio Voice Clone Loader. |

### The three generation nodes, and when to use which

- **Generate** — you want the guided experience: preset dropdown, separate voice/scene/action fields, `[bracketed]` inline cues. Builds the XML for you.
- **Generate (XML)** — you already have (or want to hand-write) a `<speak>` prompt in the exact format Scenema's own docs/API use. No translation layer, no auto-derived `shot`/`language` — you spell out every attribute.
- **Dialogue Generate** — multiple characters talking, and you want the whole scene as one clip. Two ways to write speakers, freely mixable: declare recurring characters once at the top with `<speaker id="Dad" voice="..." gender=".../>` and reference them with `<speak speaker="Dad">` (any attribute given directly on a `<speak>` overrides the declaration just for that line), or write the full inline form per turn for one-off characters. `<speak_overlap>`/`<speak_interruption>` require every speaker they reference to be declared first — see this node's source docstring for the full schema, including the two interruption-timing mechanisms (`dif_param` percentage vs. the `!INTERRUPTION!` text marker). Worth knowing: overlap/interruption are post-generation audio mixing of independently generated clean lines, not the model natively producing simultaneous speech — it reads as convincing dialogue but isn't genuine interruption-aware prosody. For voice-cloning specific characters, connect `ref_speaker_1` (first speaker to appear anywhere in the dialogue) — the node reveals `ref_speaker_2` once that's connected, and so on.

All three accept the same `SA_MODEL` / `SA_VAE` / `SA_TEXT_ENCODER` inputs from the three loader nodes, plus optional `separator` (MelBandRoFormer Loader) and `voice_clone` (Voice Clone Loader) — connect these two whenever background stripping or cross-chunk voice-lock might actually run; if one triggers at runtime and isn't connected, you get a clear error explaining which loader to add rather than a silent auto-load. They're three front ends onto one shared engine (`nodes/generate_core.py`).

### A note on VRAM now that Voice Clone is loader-based

Previously, the SeedVC stack loaded and freed itself around each use to protect smaller cards from OOM against the next transformer load. Now that it's owned by a loader node (like everything else), it stays resident across runs instead. On 8GB cards, keep an eye on total VRAM if you're stacking the transformer + Gemma + Voice Clone stack all at once — ComfyUI's own model management will offload as needed, but it's worth knowing the behavior changed.

## Writing prompts

Three fields drive expressiveness.

**Voice description.** Describes the speaker. Age, gender presentation, timbre, accent, delivery.

```
Male, mid 50s. Refined Central European accent with an Austrian tinge.
Warm baritone that turns cold in an instant. Cultured, articulate,
dangerously calm.
```

**Action tags.** Opening delivery cues, one per line. Sets the overall performance for the speech.

```
He smiles as he speaks, without warmth
```

**Speech text.** The words to say. Use inline `[bracketed cues]` for mid speech performance direction. Each bracket becomes a stage direction the model performs at that exact position in the text.

```
You know, [he lets out a soft, dry laugh] I've always found politeness
to be such a charming way of holding a knife. [His voice drops, suddenly
intimate] And all the while, their hands are already reaching for you.
```

Available bracket cues include laughs, whispers, voice cracks, gasps, sobs, sighs, pauses, mood shifts, emphasis, and breath sounds.

## Preset dropdown

Twelve production tested voices from the [official demos](https://scenema.ai/audio). Auto-fills all fields when selected. Custom lets you write your own from scratch.

## Voice cloning

1. **Load Audio** (built-in ComfyUI node) plus **Scenema Audio VAE Encode**. Connect to Generate's `ref_latent` input.
2. **Scenema Audio Load Audio from URL** (paste a link) plus **VAE Encode**. Connect to `ref_latent`.

Reference clip is capped at 20 seconds. Whenever the cross-chunk voice-lock pass runs (long text split into multiple chunks, `skip_vc=False`), connect a **Scenema Audio Voice Clone Loader** to Generate's `voice_clone` input — it's required for that pass, same as every other model connection in this package.

## Languages

Twelve tested: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi, Arabic, Swahili.

## Links

- [Demos and full article](https://scenema.ai/audio)
- [Model weights](https://huggingface.co/ScenemaAI/scenema-audio)
- [Standalone Docker server](https://github.com/ScenemaAI/scenema-audio)

## License

Node code is MIT. See [LICENSE](LICENSE).

Model weights are released under the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE). Gemma 3 12B IT is released under Google's [Gemma license](https://ai.google.dev/gemma/terms).
