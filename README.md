# ComfyUI-ScenemaAudio (local / offline fork)

Native ComfyUI nodes for [Scenema Audio](https://scenema.ai/audio). Expressive text-to-speech with zero-shot voice cloning, built on the LTX 2.3 audio diffusion transformer.

This is not just TTS. It is real vocal performance — laughs, whispers, voice cracks, breath, singing, foreign accents — all driven by prompt cues, plus multi-speaker dialogue with overlapping and interrupting speech.

> **This is a proof-of-concept release, not a maintained project.** It was built as an exploration of how far a fully local/offline ComfyUI integration and a richer prompt/dialogue format could be pushed on top of the original [ScenemaAI/ComfyUI-ScenemaAudio](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio). There is no roadmap, no support channel, and no guarantee anything here keeps working as ComfyUI, Scenema Audio, or any of the underlying libraries change. It has not been tested end-to-end on real hardware by the people who wrote it — treat it as a working sketch, not a polished product.
>
> **Do whatever you want with this code.** Fork it, cut it up, merge parts of it into something else, redistribute it, ignore it — no permission needed and no attribution required beyond what's already below. The one thing that carries real obligations is the **model weights**, which remain under their own original licenses (LTX-2 Community License for the Scenema Audio checkpoints, Google's Gemma license for Gemma 3 12B IT) — see [License](#license).

## What this fork changes, relative to the original

The upstream project downloads every model automatically from HuggingFace the first time each node runs. This fork replaces that with explicit, ComfyUI-native loader nodes — the same pattern as core ComfyUI's "Load Diffusion Model" / "Load VAE" / "Load CLIP" — and adds a few new capabilities on top:

- **Everything loads from local files through dedicated loader nodes.** No node downloads or auto-fetches a model on its own, with exactly one narrow exception (see below). Five new loader nodes: Diffusion Loader, VAE Loader, Text Encoder Loader (Gemma 3), MelBandRoFormer Loader, Voice Clone Loader.
- **One deliberate, narrow exception:** Scenema Audio Voice Clone Loader checks for the small SeedVC DiT checkpoint locally first, and only reaches HuggingFace if it's genuinely missing — caching the result locally afterward so it never needs the network again. Every other model-consuming node in the package has zero network code, period.
- **A raw XML prompt node** (`Scenema Audio Generate (XML)`) that takes the base model's native `<speak>` prompt format verbatim, for anyone who already has prompts in that format or wants full manual control without the friendly widget layer.
- **A multi-speaker dialogue node** (`Scenema Audio Dialogue Generate`) that didn't exist upstream at all: declared, reusable character voices; automatic per-speaker voice continuity across a whole conversation; layered/overlapping speech; one line interrupting another with either a percentage-based or a precise word-marker splice point; and progressively-revealed voice-clone reference inputs per character.
- **A storage split** between ComfyUI's shared `models/` tree (for models any other node might reasonably also want — transformer, VAE, Gemma, MelBandRoFormer) and a package-local `models/` folder (for SeedVC's helper checkpoints, which nothing else would ever look for).

None of this changes what the underlying model can generate — it's entirely about how models get loaded and how prompts get authored.

## Quick start

1. Install the node (see below).
2. Download the model files once, manually, and place them in the folders listed under [Model files](#model-files-download-once-place-manually).
3. Load `workflows/scenema_audio.json`, connect the loader nodes, pick a preset, hit Queue.

## Installation

### ComfyUI Manager

Open the Manager tab → **Install via Git URL** → paste this repository's URL → restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url>.git ComfyUI-ScenemaAudio
pip install -r ComfyUI-ScenemaAudio/requirements.txt
```

Restart ComfyUI. **Nothing downloads automatically.** Fetch the model files yourself once — see below.

## Model files (download once, place manually)

Every model loads through a dedicated loader node — a dropdown pointed at a local file or folder. Where files live is split by design:

- **Core pipeline models** (transformer, VAE, Gemma 3 12B IT) and **MelBandRoFormer** go under ComfyUI's shared `models/` folders, same as any native loader.
- **Voice-clone helper models** (CAMPPlus, BigVGAN, Whisper, DiT) live inside this package's own folder instead — not general-purpose models any other node would look for.

### Core pipeline models + MelBandRoFormer — under `ComfyUI/models`

| File | Where to put it | Size | Source | Loader node |
|---|---|---|---|---|
| `scenema-audio-transformer-int8.safetensors` **or** `-bf16` | `ComfyUI/models/diffusion_models/` | ~4.9 GB / ~9.8 GB | [ScenemaAI/scenema-audio](https://huggingface.co/ScenemaAI/scenema-audio) | Diffusion Loader |
| `scenema-audio-pipeline.safetensors` | `ComfyUI/models/diffusion_models/` | — | same repo | Text Encoder Loader |
| `scenema-audio-pipeline-audio.safetensors` | `ComfyUI/models/vae/` | ~6.7 GB | same repo | VAE Loader |
| `scenema-audio-vae-encoder.safetensors` | `ComfyUI/models/vae/` | ~42.7 MB | same repo | VAE Loader |
| Gemma 3 12B IT (folder: `config.json`, tokenizer, `*.safetensors` shards) | `ComfyUI/models/text_encoders/gemma-3-12b-it/` | ~24 GB | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (gated) | Text Encoder Loader |
| `MelBandRoformer_fp16.safetensors` | `ComfyUI/models/checkpoints/` | ~280 MB | [Kijai/MelBandRoFormer_comfy](https://huggingface.co/Kijai/MelBandRoFormer_comfy) | MelBandRoFormer Loader |

**Alternative layout:** the same files can instead go under `ComfyUI/models/ScenemaAudio/{diffusion_models,vae,text_encoders,checkpoints}/` — this package registers that folder as an additional search path for all four categories at startup, so both layouts work at once. Mix and match.

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

# Gemma 3 12B IT — gated: accept the license on the model page first, then
# authenticate (huggingface-cli login, or export HF_TOKEN=hf_...)
huggingface-cli download google/gemma-3-12b-it \
    --local-dir ComfyUI/models/text_encoders/gemma-3-12b-it

# MelBandRoFormer — not gated, but deliberately not fetched by any script
# in this package. Grab it yourself:
huggingface-cli download Kijai/MelBandRoFormer_comfy MelBandRoformer_fp16.safetensors \
    --local-dir ComfyUI/models/checkpoints
```

### Voice-clone helper models — inside this package, not `ComfyUI/models`

These live under `ComfyUI/custom_nodes/ComfyUI-ScenemaAudio/models/voice_clone/` (ships empty, with a placeholder `.txt` explaining what goes where):

| File | Where to put it | Size | Source |
|---|---|---|---|
| `campplus_cn_common.bin` | `models/voice_clone/` | ~28 MB | [funasr/campplus](https://huggingface.co/funasr/campplus) |
| BigVGAN folder | `models/voice_clone/bigvgan/` | ~0.5 GB | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) |
| Whisper-small folder | `models/voice_clone/whisper-small/` | ~1 GB | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + `config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | `models/voice_clone/` (optional — see below) | ~150 MB | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC) |

None of these are gated. Run the included convenience script:

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

That's it. After all of the above, you can disconnect from the network entirely — every loader node works from what's on disk.

### Local-first, HuggingFace-as-last-resort (the one exception)

Every model-consuming node stays fully local, with exactly one narrow exception: **the SeedVC DiT checkpoint**, and only inside **Scenema Audio Voice Clone Loader**.

1. It checks `models/voice_clone/` for `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + its config yml first.
2. If they're there (placed by hand, or by the prefetch script), it uses them — no network touched.
3. Only if they're genuinely missing does it fetch from `Plachta/Seed-VC`, then caches the result back into that same folder — every run after the first is local too.

Want zero network dependency even on the first run? Place those two files yourself, or run the prefetch script (it fetches these along with everything else above).

### Gemma 3 12B IT — already "lazy" safetensors loading, no conversion needed

Gemma ships as standard sharded `safetensors`, which already memory-maps weights lazily. `transformers.Gemma3ForConditionalGeneration.from_pretrained(local_dir, local_files_only=True)` uses exactly that mechanism when pointed at a local folder — nothing special to do. The only network dependency was ever the *first* download (Google's repo is gated); once the folder exists locally, this node never touches HuggingFace for it again. There's no way to load Gemma 3 12B IT through ComfyUI's native `CLIPLoader`/`DualCLIPLoader` (they expect single-file text-encoder checkpoints with known key prefixes), which is why this fork ships its own Text Encoder Loader instead.

### `validate=`/`!INTERRUPTION!` Whisper — reuses Voice Clone Loader's own whisper-small

Both features need Whisper for transcription and word-level timing. Rather than a second copy of whisper-small in a different format, this reuses the exact same local checkpoint Voice Clone Loader already loads (`models/voice_clone/whisper-small/`) through `transformers`. If that's already set up, both features just work. If no local Whisper checkpoint is found, both degrade gracefully rather than reaching for the network — `validate=True` raises with a clear message, and `!INTERRUPTION!` logs a warning and falls back to the percentage-based splice point.

## Hardware requirements

Minimum 8 GB VRAM. Generation runs up to ~2x realtime on higher-end cards. Roughly 32 GB system RAM recommended for the pipeline components.

Gemma 3 12B IT's encoding mode is picked automatically from VRAM (override via the Text Encoder Loader's `precision` input):

| VRAM | Mode | Extra VRAM used |
|---|---|---|
| 40 GB+ | `bf16_gpu` | ~24 GB |
| 12–39 GB | `nf4` | ~8 GB |
| < 12 GB | `cpu_streaming` | minimal, slower |

The SeedVC voice-clone stack, once loaded by its loader node, stays resident across runs (unlike the upstream node, which loaded and freed it around every use). On smaller cards, keep an eye on total VRAM if you're running the transformer + Gemma + Voice Clone stack all at once.

## Nodes

Twelve user-facing nodes.

| Node | Purpose |
|---|---|
| **Scenema Audio Diffusion Loader** | Loads the 3.3B audio transformer from a local file — dropdown reads `models/diffusion_models/`. |
| **Scenema Audio VAE Loader** | Loads the audio VAE encoder + decoder — dropdowns read `models/vae/`. |
| **Scenema Audio Text Encoder Loader (Gemma 3)** | Loads Gemma 3 12B IT + the pipeline checkpoint, fully local. |
| **Scenema Audio MelBandRoFormer Loader** | Loads the vocal/background separator used for background-SFX stripping. Manual-only — no auto-fetch anywhere. Connect its output to `separator` on any Generate node; required whenever `strip_background_sfx` actually strips. |
| **Scenema Audio Voice Clone Loader** | Loads the SeedVC stack (CAMPPlus, BigVGAN, Whisper, DiT) used for cross-chunk voice consistency and standalone voice conversion. The only node that can ever reach HuggingFace, and only as a last resort (see above). Connect its output to `voice_clone` on any Generate node; required whenever the cross-chunk voice-lock pass actually runs. |
| **Scenema Audio Text Encode** | Standalone prompt → conditioning node, for encoding ahead of time or reusing conditioning across generations. |
| **Scenema Audio Generate** | The main node, friendly widget mode: voice description, gender, speech text, scene, plus a preset dropdown with 12 curated voices. Builds the model's native XML prompt for you. |
| **Scenema Audio Generate (XML)** | Same engine as Generate, but takes a raw `<speak voice="..." scene="..." ...>...</speak>` prompt verbatim — the base model's native format, no widget layer in between. |
| **Scenema Audio Dialogue Generate** | Multi-speaker dialogue from a `<dialogue>` of turns — see [Dialogue Generate](#dialogue-generate) below. |
| **Scenema Audio VAE Encode** | Encodes reference audio to a latent for voice cloning. Capped at 20 seconds. |
| **Scenema Audio Load Audio from URL** | Loads a reference audio file from a URL (mp3, wav, flac) for voice cloning. |
| **Scenema Audio Voice Clone** | Standalone SeedVC voice conversion. Requires a `voice_clone` input from Voice Clone Loader. |

All three generation nodes (Generate, Generate XML, Dialogue Generate) accept the same `SA_MODEL` / `SA_VAE` / `SA_TEXT_ENCODER` inputs from the three core loaders, plus optional `separator` (MelBandRoFormer Loader) and `voice_clone` (Voice Clone Loader) — connect these whenever background stripping or cross-chunk voice-lock might actually run; if one triggers at runtime without being connected, you get a clear error naming which loader to add. They share one engine (`nodes/generate_core.py`).

## Writing prompts

Three fields drive expressiveness on the friendly Generate node.

**Voice description** — age, gender presentation, timbre, accent, delivery:

```
Male, mid 50s. Refined Central European accent with an Austrian tinge.
Warm baritone that turns cold in an instant. Cultured, articulate,
dangerously calm.
```

**Action tags** — opening delivery cues, one per line, set the overall performance:

```
He smiles as he speaks, without warmth
```

**Speech text** — the words to say, with inline `[bracketed cues]` for mid-speech performance direction. Each bracket becomes a stage direction performed at that exact position:

```
You know, [he lets out a soft, dry laugh] I've always found politeness
to be such a charming way of holding a knife. [His voice drops, suddenly
intimate] And all the while, their hands are already reaching for you.
```

Bracket cues include laughs, whispers, voice cracks, gasps, sobs, sighs, pauses, mood shifts, emphasis, and breath sounds.

For the raw `<speak>` XML format used by Generate (XML) and inside Dialogue Generate, see `audio_core/compiler.py` and the docstring at the top of `nodes/dialogue_generate.py`.

## Dialogue Generate

Takes a `<dialogue>` of turns and produces one continuous multi-speaker clip.

**Two ways to write speakers, freely mixable:**

```xml
<!-- Declared (reusable) -->
<speaker id="Man" voice="Deep male voice, gravelly" gender="male" scene="Dim room"/>
<speak speaker="Man">
  <action>He leans forward</action>
  Where were you last night?
</speak>

<!-- Inline (one-off, no declaration needed) -->
<speak voice="Gruff delivery driver, bored" gender="male">
  Package for the Connors.
</speak>
```

Any attribute given directly on a `<speak>` overrides its declaration just for that line. Each speaker's own voice-continuity reference tracks independently, so it doesn't matter who else spoke in between.

**Overlapping speech** — everyone listed starts at the same time:

```xml
<speak_overlap speakers="Dad,Mom">
  <action>Both shocked</action>
  Wait, what?!
</speak_overlap>
```

Use numbered `<action_1>`/`<action_2>` instead of a single `<action>` to give each speaker a different simultaneous line rather than the same one in unison.

**Interruption** — one party cuts into the tail of another's line:

```xml
<speak_interruption speakers="Man,Woman" dif_param="20">
  <action_1>He starts calmly, hardening</action_1>
  Because someone saw you, and I am done pretending I don't know where you were.
  <action_2>She cuts him off, sharp</action_2>
  You don't know anything.
</speak_interruption>
```

`dif_param` (percent, default 15) controls how far into party 1's line party 2 cuts in. For a precise splice point instead of a blind percentage, put a literal `!INTERRUPTION!` marker inside party 1's text — it's located via Whisper word-level alignment and always wins over `dif_param` when present. Either party can be a `[bracketed, group]` of several speakers overlapping each other instead of one name, e.g. `speakers="[Dad,Timmy],[Mom]"`.

Worth knowing: overlap and interruption are post-generation audio mixing of independently generated clean lines, not the model natively producing simultaneous speech. It reads as convincing dialogue but isn't genuine interruption-aware prosody — nobody actually reacts to being cut off.

**Voice cloning per character:** connect `ref_speaker_1` (the first speaker to appear anywhere in the dialogue) from a VAE-Encode'd reference clip; the node reveals `ref_speaker_2` once that's connected, and so on, up to 8.

## Preset dropdown

Twelve production-tested voices from the [official demos](https://scenema.ai/audio). Auto-fills all fields when selected; Custom lets you write your own from scratch.

## Voice cloning (single speaker)

1. **Load Audio** (built-in) + **Scenema Audio VAE Encode** → Generate's `ref_latent` input, or
2. **Scenema Audio Load Audio from URL** + **VAE Encode** → `ref_latent`.

Reference clip is capped at 20 seconds. Whenever the cross-chunk voice-lock pass runs (long text split into multiple chunks, `skip_vc=False`), connect a Voice Clone Loader to `voice_clone`.

## Languages

Twelve tested: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi, Arabic, Swahili.

## Links

- [Demos and full article](https://scenema.ai/audio)
- [Model weights](https://huggingface.co/ScenemaAI/scenema-audio)
- [Original upstream node](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio)
- [Standalone Docker server](https://github.com/ScenemaAI/scenema-audio)

## License

Node code in this repository is MIT — see [LICENSE](LICENSE). Use it, modify it, redistribute it, build something else out of it, for any purpose, with or without attribution.

**Model weights are not covered by that MIT license and remain under their own terms:**
- Scenema Audio checkpoints (transformer, VAE, pipeline) — [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)
- Gemma 3 12B IT — Google's [Gemma license](https://ai.google.dev/gemma/terms)
- CAMPPlus, BigVGAN, Whisper, SeedVC DiT, MelBandRoFormer — each under their own respective upstream licenses (see the linked HuggingFace repos above)

Check the model license before any commercial or redistribution use — the code here doesn't grant rights to the weights.
