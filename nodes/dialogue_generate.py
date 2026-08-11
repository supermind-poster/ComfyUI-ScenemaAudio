# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Dialogue Generate — automatic multi-speaker dialogue.

Takes a <dialogue> of <speak> turns (one per line, in speaking order) and
produces one continuous audio clip.

──────────────────────────────────────────────────────────────────────
SPEAKERS — two styles, freely mixable

1) Inline (original, fully backward compatible) — every <speak> carries
   its own full voice description:

    <speak voice="Deep male voice, gravelly" gender="male" scene="Dim room">
      <action>He leans forward</action>
      Where were you last night?
    </speak>

2) Declared — define each recurring character once with a
   <speaker id="..."> tag, then reference by id. No retyping the same
   voice description on every line, and any attribute given directly on
   a <speak> overrides its declaration just for that one turn:

    <speaker id="Man" voice="Deep male voice, gravelly" gender="male" scene="Dim room"/>
    <speak speaker="Man">
      <action>He leans forward</action>
      Where were you last night?
    </speak>

──────────────────────────────────────────────────────────────────────
OVERLAP AND INTERRUPTION

The underlying model is single-voice TTS — there's no such thing as it
natively generating two people talking over each other. <speak_overlap>
and <speak_interruption> are literal post-generation audio mixing: each
participant's line is generated independently and cleanly, then layered
in time. This is the same technique audio drama post-production uses to
build overlapping dialogue — it sounds convincing, but it isn't the
model reacting to another voice mid-line (no genuine interruption-aware
prosody, no one trailing off because they got cut off).

Both tags require every speaker they reference to already be declared
with <speaker id="...">  — there's no room on these tags for full inline
voice attributes per participant.

<speak_overlap speakers="A,B,...">  — everyone listed starts at the same
time (full overlap):
  - One shared <action> + one text body -> identical line, spoken by
    every listed speaker at once (e.g. a crowd shouting "Surprise!").
  - Numbered <action_1>, <action_2>, ... -> each listed speaker (by
    position in the speakers= list) gets their own line, all starting
    together (different simultaneous lines).

<speak_interruption speakers="A,B" dif_param="15">  — B cuts into the
tail of A's line rather than starting from scratch (partial overlap,
not a full unison start). Always exactly two parties; either can be a
[bracketed,group] of several speakers reciting/overlapping together
instead of a single name — e.g. speakers="[Dad,Timmy],[Mom]" for "Dad
and Timmy talking over each other, then Mom cuts through both of them".
<action_1>/<action_2> apply to party 1 / party 2 as a whole. The splice
point (how far into party 1's line party 2 cuts in) is controlled by:
  - dif_param="15" (percent, default 15) — party 2 starts 15% before
    party 1's own line ends. Doesn't know about word boundaries, so it
    can land mid-word.
  - a literal !INTERRUPTION! marker inside party 1's text — precisely
    locates the splice point via Whisper word-level alignment on party
    1's generated audio instead. Best-effort (depends on transcription
    accuracy), but far more natural than a blind percentage when it
    works. If present, this always wins over dif_param.

    <speak_interruption speakers="Dad,Mom">
      <action_1>Dad, mid-explanation</action_1>
      Look, if you would just let me finish explaining this one thing !INTERRUPTION! it would all make sense.
      <action_2>Mom, cutting in sharply</action_2>
      No time for that.
    </speak_interruption>

──────────────────────────────────────────────────────────────────────
Each <speak> is otherwise exactly the base model's native prompt format
(see audio_core/compiler.py / ScenemaAudioGenerateXML) — voice, scene,
gender, language, shot attributes, <action>/<sound> tags.

Turns are processed in dialogue (document) order. Each speaker's own
A2V continuity reference is tracked independently (keyed per speaker),
so processing chronologically doesn't cause any cross-speaker voice
bleed — a speaker's next line always conditions on THEIR OWN last line,
regardless of who else spoke (or overlapped, or interrupted) in between.

Optional ref_speaker_1..8 inputs seed that continuity reference with a
real voice-clone target from the start, instead of waiting for the
character's own first generated line — connect ref_speaker_1 for the
first speaker to appear anywhere in the dialogue, ref_speaker_2 for the
second, and so on (in ComfyUI's UI this reveals progressively: connect
ref_speaker_1 and ref_speaker_2 appears, etc).
"""

import logging
import re
import xml.etree.ElementTree as ET

import torch
import torchaudio

from .generate_core import run_generation, _encode_reference
from .dialogue_audio_ops import duration_seconds, mix_overlap, splice_interrupt

logger = logging.getLogger(__name__)

MAX_REF_SPEAKERS = 8
DEFAULT_DIF_PARAM = 15.0  # percent
INTERRUPTION_MARKER = "!INTERRUPTION!"

EXAMPLE_DIALOGUE_XML = """<dialogue>
  <speaker id="Man" voice="Male, 40s. Low, controlled, a little tired." gender="male" scene="Quiet indoor room"/>
  <speaker id="Woman" voice="Female, late 20s. Nervous, quick to fill silence." gender="female" scene="Quiet indoor room"/>

  <speak speaker="Man">
    <action>He leans back, arms crossed</action>
    Where were you last night?
  </speak>
  <speak speaker="Woman">
    <action>She hesitates</action>
    I... I was at home. Why does it matter?
  </speak>
  <speak_interruption speakers="Man,Woman" dif_param="20">
    <action_1>He starts calmly, then his voice hardens</action_1>
    Because someone saw you, and I am done pretending I don't !INTERRUPTION! know exactly where you were.
    <action_2>She cuts him off, sharp</action_2>
    You don't know anything.
  </speak_interruption>
</dialogue>"""


# ── XML escaping ────────────────────────────────────────────────────


def _esc_text(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(s):
    return _esc_text(s).replace('"', "&quot;")


def _build_speak_xml(attrs, action, text):
    """Build a standalone <speak ...>...</speak> string from a resolved
    attribute dict (voice/gender/scene/language/shot) plus one action
    cue and one body of spoken text.
    """
    attr_str = " ".join(f'{k}="{_esc_attr(v)}"' for k, v in attrs.items())
    body = f"<action>{_esc_text(action)}</action>\n" if action else ""
    body += _esc_text(text)
    return f"<speak {attr_str}>{body}</speak>"


# ── <speaker> declarations + <speak> resolution (inline vs. declared) ──


def _parse_declarations(root):
    """Collect <speaker id="..." voice="..." .../> declarations from the
    direct children of <dialogue>. Returns {id: {attrib dict without id}}.
    """
    declared = {}
    for child in root:
        if child.tag != "speaker":
            continue
        speaker_id = child.get("id", "").strip()
        if not speaker_id:
            raise ValueError(
                "<speaker> declaration is missing the required id attribute, "
                'e.g. <speaker id="Dad" voice="..." gender="male"/>.'
            )
        if speaker_id in declared:
            raise ValueError(f'<speaker id="{speaker_id}"> declared more than once.')
        attrs = {k: v for k, v in child.attrib.items() if k != "id"}
        if "voice" not in attrs:
            raise ValueError(f'<speaker id="{speaker_id}"> is missing the required voice attribute.')
        declared[speaker_id] = attrs
    return declared


def _resolve_speak_turn(child, declared, turn_index):
    """Resolve one <speak> turn's final attributes and speaker key.

    speaker="X" matching a declaration: declared attributes as a base,
    with any attribute present directly on this <speak> overriding it.
    Otherwise: today's original behavior — voice must be given directly
    on the tag; speaker key is the speaker attribute if given, else the
    voice text itself.

    Mutates and returns `child` with a final, complete attribute set and
    the `speaker` attribute removed (the base model's <speak> schema
    doesn't know that attribute).
    """
    speaker_attr = child.get("speaker", "").strip()

    if speaker_attr and speaker_attr in declared:
        merged = dict(declared[speaker_attr])
        merged.update({k: v for k, v in child.attrib.items() if k != "speaker"})
        child.attrib.clear()
        for k, v in merged.items():
            child.set(k, v)
        speaker_key = speaker_attr
    else:
        voice = child.get("voice", "").strip()
        if not voice:
            ref = f'speaker="{speaker_attr}"' if speaker_attr else "(no speaker attribute)"
            raise ValueError(
                f"Turn {turn_index}: {ref} doesn't match any <speaker id=\"...\"> "
                f"declaration, and this <speak> has no voice attribute of its own. "
                f'Either declare it at the top with <speaker id="{speaker_attr or "..."}" '
                f'voice="..." gender="..."/>, or give this line its own voice="..." directly.'
            )
        speaker_key = speaker_attr or voice

    if "speaker" in child.attrib:
        del child.attrib["speaker"]

    return child, speaker_key


def _require_declared(speaker_ids, declared, tag_name):
    missing = [s for s in speaker_ids if s not in declared]
    if missing:
        raise ValueError(
            f"<{tag_name}>: speaker(s) {missing} not declared. <{tag_name}> requires "
            f'every referenced speaker to have a <speaker id="..." voice="..." '
            f'gender="..."/> declaration at the top of <dialogue> — there is no room '
            f"for inline voice attributes on {tag_name} itself."
        )


# ── speakers="..." attribute parsing (flat lists and [bracket] groups) ──


def _split_top_level(value):
    """Split on top-level commas, respecting [...] grouping.
    'A,B' -> ['A','B'];  '[A,B],[C]' -> ['[A,B]','[C]']
    """
    parts, depth, current = [], 0, ""
    for ch in value:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_group_token(token):
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        return [s.strip() for s in token[1:-1].split(",") if s.strip()]
    return [token]


def _parse_overlap_speakers(value):
    if "[" in value or "]" in value:
        raise ValueError(
            f'speak_overlap speakers="{value}" should be a flat comma list — '
            f"[bracket] groups are only meaningful for speak_interruption."
        )
    return [s.strip() for s in value.split(",") if s.strip()]


def _parse_interruption_parties(value):
    tokens = _split_top_level(value)
    if len(tokens) != 2:
        raise ValueError(
            f'speak_interruption speakers="{value}" must resolve to exactly two '
            f"parties (each a single speaker or a [bracketed,list]), got {len(tokens)}."
        )
    return _parse_group_token(tokens[0]), _parse_group_token(tokens[1])


# ── action/action_N + text segmentation for overlap/interruption ──


def _extract_segments(elem, n_parties):
    """Split an overlap/interruption element's children into n_parties
    {"action": str|None, "text": str} segments. See module docstring for
    the unison vs. per-party rules.
    """
    generic_action = None
    numbered = {}  # n -> (action_text, spoken_text)

    for child in elem:
        if child.tag == "action":
            generic_action = (child.text or "").strip()
            continue
        m = re.fullmatch(r"action_(\d+)", child.tag)
        if m:
            n = int(m.group(1))
            numbered[n] = ((child.text or "").strip(), (child.tail or "").strip())

    if not numbered:
        parts = []
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        for child in elem:
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        shared_text = " ".join(parts).strip()
        if not shared_text:
            raise ValueError(f"<{elem.tag}> has no spoken text.")
        return [{"action": generic_action, "text": shared_text} for _ in range(n_parties)]

    segments = []
    for n in range(1, n_parties + 1):
        action_text, spoken_text = numbered.get(n, (None, ""))
        action = action_text if action_text else generic_action
        if not spoken_text:
            raise ValueError(
                f"<{elem.tag}>: party {n} has no spoken text — expected an "
                f"<action_{n}> tag immediately followed by that party's line."
            )
        segments.append({"action": action, "text": spoken_text})
    return segments


def _extract_marker(text):
    """Strip !INTERRUPTION! from text, if present, and return
    (cleaned_text, char_offset_in_cleaned_text) — or (text, None).
    """
    idx = text.find(INTERRUPTION_MARKER)
    if idx == -1:
        return text, None
    cleaned = (text[:idx] + text[idx + len(INTERRUPTION_MARKER):])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    offset = len(re.sub(r"\s+", " ", text[:idx]).strip())
    return cleaned, offset


# ── Top-level dialogue parsing -> ordered ops ──


def _parse_dialogue(dialogue_xml):
    """Parse <dialogue> into (declared, ops).

    ops is an ordered list of dicts, one of:
      {"kind": "solo", "index": i, "xml": str, "speakers": [id]}
      {"kind": "overlap", "index": i, "speakers": [ids...], "clip_xmls": [str...]}
      {"kind": "interruption", "index": i, "group1": [ids...], "group2": [ids...],
       "clip_xmls_1": [str...], "clip_xmls_2": [str...],
       "text1": str, "marker_offset": int|None, "language": str, "dif_param": float}
    """
    root = ET.fromstring(dialogue_xml.strip())
    if root.tag != "dialogue":
        raise ValueError(
            f"Root element must be <dialogue>...</dialogue>, got <{root.tag}>. "
            "Each line of dialogue goes inside as its own <speak> turn."
        )

    declared = _parse_declarations(root)
    ops = []
    idx = 0

    for child in root:
        if child.tag == "speaker":
            continue

        elif child.tag == "speak":
            resolved, speaker_key = _resolve_speak_turn(child, declared, idx)
            ops.append({
                "kind": "solo", "index": idx,
                "xml": ET.tostring(resolved, encoding="unicode"),
                "speakers": [speaker_key],
            })
            idx += 1

        elif child.tag == "speak_overlap":
            speakers = _parse_overlap_speakers(child.get("speakers", ""))
            if not speakers:
                raise ValueError(f'Turn {idx}: <speak_overlap> needs a speakers="A,B,..." attribute.')
            _require_declared(speakers, declared, "speak_overlap")
            segments = _extract_segments(child, len(speakers))
            clip_xmls = [
                _build_speak_xml(declared[spk], seg["action"], seg["text"])
                for spk, seg in zip(speakers, segments)
            ]
            ops.append({
                "kind": "overlap", "index": idx,
                "speakers": speakers, "clip_xmls": clip_xmls,
            })
            idx += 1

        elif child.tag == "speak_interruption":
            group1, group2 = _parse_interruption_parties(child.get("speakers", ""))
            _require_declared(group1 + group2, declared, "speak_interruption")
            segments = _extract_segments(child, 2)
            text1, marker_offset = _extract_marker(segments[0]["text"])
            try:
                dif_param = float(child.get("dif_param", DEFAULT_DIF_PARAM))
            except ValueError:
                raise ValueError(
                    f'Turn {idx}: dif_param="{child.get("dif_param")}" is not a number.'
                )
            if not (0.0 <= dif_param <= 100.0):
                raise ValueError(f"Turn {idx}: dif_param must be between 0 and 100, got {dif_param}.")

            clip_xmls_1 = [_build_speak_xml(declared[spk], segments[0]["action"], text1) for spk in group1]
            clip_xmls_2 = [_build_speak_xml(declared[spk], segments[1]["action"], segments[1]["text"]) for spk in group2]
            ops.append({
                "kind": "interruption", "index": idx,
                "group1": group1, "group2": group2,
                "clip_xmls_1": clip_xmls_1, "clip_xmls_2": clip_xmls_2,
                "text1": text1, "marker_offset": marker_offset,
                "language": declared[group1[0]].get("language", "en"),
                "dif_param": dif_param,
            })
            idx += 1

        else:
            raise ValueError(
                f"Unknown tag <{child.tag}> inside <dialogue> — expected <speaker>, "
                f"<speak>, <speak_overlap>, or <speak_interruption>."
            )

    if not ops:
        raise ValueError("No <speak>/<speak_overlap>/<speak_interruption> turns found inside <dialogue>.")

    return declared, ops


def _first_appearance_order(ops):
    order, seen = [], set()
    for op in ops:
        if op["kind"] in ("solo", "overlap"):
            ids = op["speakers"]
        else:
            ids = op["group1"] + op["group2"]
        for s in ids:
            if s not in seen:
                seen.add(s)
                order.append(s)
    return order


# ── Assembly ──


def _concat_with_silence(audio_list, gap_seconds):
    """Concatenate a list of {"waveform","sample_rate"} dicts in order,
    inserting gap_seconds of silence between each. Resamples to the first
    clip's sample rate if any differ.
    """
    sr = audio_list[0]["sample_rate"]
    gap_samples = max(0, int(gap_seconds * sr))

    parts = []
    for i, a in enumerate(audio_list):
        w = a["waveform"]
        if a["sample_rate"] != sr:
            w = torchaudio.functional.resample(w.float(), a["sample_rate"], sr)
        parts.append(w)
        if i < len(audio_list) - 1 and gap_samples > 0:
            silence = torch.zeros((w.shape[0], w.shape[1], gap_samples), dtype=w.dtype)
            parts.append(silence)

    combined = torch.cat(parts, dim=-1)
    return {"waveform": combined, "sample_rate": sr}


class ScenemaAudioDialogueGenerate:
    """Generates a multi-speaker dialogue from a <dialogue> of <speak>,
    <speak_overlap>, and <speak_interruption> turns. See this module's
    docstring for the full schema.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "pace": ("FLOAT", {
                "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
            }),
            "silence_between_turns": ("FLOAT", {
                "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1,
                "tooltip": "Gap inserted between consecutive dialogue turns, in seconds.",
            }),
            "strip_background_sfx": (["auto", "yes", "no"], {"default": "auto"}),
            "separator": ("SA_SEPARATOR", {
                "tooltip": "Pre-loaded MelBandRoFormer — required if strip_background_sfx ends up stripping any turn.",
            }),
            "voice_clone": ("SA_VOICE_CLONE", {
                "tooltip": "Pre-loaded SeedVC stack — required if any single line is long enough to split into multiple chunks and skip_vc=False.",
            }),
            "validate": ("BOOLEAN", {"default": False}),
            "min_match_ratio": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            "skip_vc": ("BOOLEAN", {"default": False}),
            "vc_steps": ("INT", {"default": 25, "min": 5, "max": 50, "step": 5}),
            "vc_cfg_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
        }
        for n in range(1, MAX_REF_SPEAKERS + 1):
            optional[f"ref_speaker_{n}"] = ("SA_LATENT", {
                "tooltip": (
                    f"Voice-clone reference for speaker #{n} (in first-appearance "
                    f"order anywhere in dialogue_xml, including inside "
                    f"speak_overlap/speak_interruption). Connect from VAE Encode "
                    f"fed by LoadAudio."
                ),
            })

        return {
            "required": {
                "model": ("SA_MODEL",),
                "vae": ("SA_VAE",),
                "text_encoder": ("SA_TEXT_ENCODER", {
                    "tooltip": "Connect from Scenema Audio Text Encoder Loader.",
                }),
                "dialogue_xml": ("STRING", {
                    "multiline": True,
                    "default": EXAMPLE_DIALOGUE_XML,
                    "tooltip": (
                        "<dialogue> of <speak>/<speak_overlap>/<speak_interruption> "
                        "turns, in order. Declare recurring characters once with "
                        '<speaker id="X" voice="..." gender="..."/> at the top, then '
                        'reference with <speak speaker="X"> (or write full inline '
                        "voice=\"...\" per turn like before — both styles mix freely). "
                        "See this field's default for an example, and this node's "
                        "source docstring for the full schema."
                    ),
                }),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": optional,
        }

    def _run_solo(self, model, vae, text_encoder, op, base_seed, last_ref_by_speaker,
                   pace, strip_background_sfx, separator, voice_clone,
                   validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate):
        speaker = op["speakers"][0]
        ref = last_ref_by_speaker.get(speaker)
        audio = run_generation(
            model, vae, text_encoder, op["xml"], base_seed + op["index"],
            separator, voice_clone, pace, strip_background_sfx,
            ref.cuda() if ref is not None else None,
            validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
        )
        last_ref_by_speaker[speaker] = _encode_reference(vae, audio["waveform"], audio["sample_rate"]).cpu()
        return audio

    def _run_party(self, model, vae, text_encoder, speaker_ids, xmls, seed_base,
                    last_ref_by_speaker, pace, strip_background_sfx, separator,
                    voice_clone, validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate):
        """Generate one clean line per speaker in a party (1 for a solo
        interruption party or an overlap participant, N for a bracketed
        interruption group), updating each speaker's own reference from
        their own clean line, then mix if there's more than one.
        """
        clips = []
        for i, (speaker, xml) in enumerate(zip(speaker_ids, xmls)):
            ref = last_ref_by_speaker.get(speaker)
            audio = run_generation(
                model, vae, text_encoder, xml, seed_base + i,
                separator, voice_clone, pace, strip_background_sfx,
                ref.cuda() if ref is not None else None,
                validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
            )
            last_ref_by_speaker[speaker] = _encode_reference(vae, audio["waveform"], audio["sample_rate"]).cpu()
            clips.append(audio)
        return mix_overlap(clips)

    def _run_interruption(self, model, vae, text_encoder, op, base_seed, last_ref_by_speaker,
                           pace, strip_background_sfx, separator, voice_clone,
                           validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate):
        seed1 = base_seed + op["index"] * 100
        seed2 = base_seed + op["index"] * 100 + 50

        party1_audio = self._run_party(
            model, vae, text_encoder, op["group1"], op["clip_xmls_1"], seed1,
            last_ref_by_speaker, pace, strip_background_sfx, separator, voice_clone,
            validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
        )
        party2_audio = self._run_party(
            model, vae, text_encoder, op["group2"], op["clip_xmls_2"], seed2,
            last_ref_by_speaker, pace, strip_background_sfx, separator, voice_clone,
            validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
        )

        splice_seconds = None
        if op["marker_offset"] is not None:
            try:
                from audio_core.whisper_aligner import find_marker_time
                wav = party1_audio["waveform"].squeeze(0).numpy()
                if wav.ndim == 2:
                    wav = wav.T
                splice_seconds = find_marker_time(
                    wav, party1_audio["sample_rate"], op["text1"],
                    op["marker_offset"], language=op["language"],
                )
            except Exception as e:
                logger.warning("  !INTERRUPTION! marker alignment failed (%s), "
                                "falling back to dif_param.", e)

        if splice_seconds is None:
            party1_duration = duration_seconds(party1_audio)
            splice_seconds = party1_duration * (1.0 - op["dif_param"] / 100.0)
            logger.info("  Interruption splice via dif_param=%.0f%% -> %.2fs",
                         op["dif_param"], splice_seconds)
        else:
            logger.info("  Interruption splice via !INTERRUPTION! marker -> %.2fs", splice_seconds)

        return splice_interrupt(party1_audio, party2_audio, splice_seconds)

    @torch.inference_mode()
    def generate(self, model, vae, text_encoder, dialogue_xml, base_seed,
                 pace=1.5, silence_between_turns=0.4,
                 strip_background_sfx="auto", separator=None, voice_clone=None,
                 validate=False, min_match_ratio=0.9,
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5,
                 **ref_speaker_kwargs):

        declared, ops = _parse_dialogue(dialogue_xml)
        speaker_order = _first_appearance_order(ops)
        logger.info("Dialogue: %d turn(s), %d speaker(s)", len(ops), len(speaker_order))

        last_ref_by_speaker = {}
        for n, speaker in enumerate(speaker_order, start=1):
            ref = ref_speaker_kwargs.get(f"ref_speaker_{n}")
            if ref is not None:
                logger.info("  ref_speaker_%d -> voice-cloning speaker '%s'", n, speaker)
                last_ref_by_speaker[speaker] = ref.cpu()

        results = []
        for op in ops:
            if op["kind"] == "solo":
                logger.info("Turn %d/%d: solo (%s)", op["index"] + 1, len(ops), op["speakers"][0])
                audio = self._run_solo(
                    model, vae, text_encoder, op, base_seed, last_ref_by_speaker,
                    pace, strip_background_sfx, separator, voice_clone,
                    validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
                )
            elif op["kind"] == "overlap":
                logger.info("Turn %d/%d: overlap (%s)", op["index"] + 1, len(ops), ", ".join(op["speakers"]))
                audio = self._run_party(
                    model, vae, text_encoder, op["speakers"], op["clip_xmls"],
                    base_seed + op["index"] * 100, last_ref_by_speaker,
                    pace, strip_background_sfx, separator, voice_clone,
                    validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
                )
            else:  # interruption
                logger.info("Turn %d/%d: interruption (%s -> %s)", op["index"] + 1, len(ops),
                             "+".join(op["group1"]), "+".join(op["group2"]))
                audio = self._run_interruption(
                    model, vae, text_encoder, op, base_seed, last_ref_by_speaker,
                    pace, strip_background_sfx, separator, voice_clone,
                    validate, min_match_ratio, skip_vc, vc_steps, vc_cfg_rate,
                )
            results.append(audio)

        combined_audio = _concat_with_silence(results, silence_between_turns)

        total_duration = duration_seconds(combined_audio)
        logger.info("Dialogue complete: %.1fs, %d turns, %d speakers",
                     total_duration, len(ops), len(speaker_order))

        return (combined_audio,)
