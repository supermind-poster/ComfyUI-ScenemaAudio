// Copyright (c) 2026 Scenema AI
// SPDX-License-Identifier: MIT
//
// Scenema Audio Dialogue Generate declares up to MAX_REF_SPEAKERS
// ref_speaker_N optional inputs in its Python schema (ComfyUI needs
// every possible socket name declared up front), but showing all of
// them at once would be clutter for the common case of 2-3 speakers.
//
// This extension makes them appear progressively: only ref_speaker_1 is
// visible on a fresh node. Connect it, and ref_speaker_2 appears.
// Connect that, and ref_speaker_3 appears, and so on up to the declared
// max. Disconnecting a slot removes any now-unconnected trailing slots
// beyond it, so the node always shows exactly one empty slot past the
// last connected one.
//
// Mapping is positional (see nodes/dialogue_generate.py): ref_speaker_1
// is the voice-clone reference for the first speaker to appear in
// dialogue_xml, ref_speaker_2 the second, etc.

import { app } from "../../../scripts/app.js";

const NODE_NAME = "ScenemaAudioDialogueGenerate";
const PREFIX = "ref_speaker_";
const MAX_REF_SPEAKERS = 8; // keep in sync with MAX_REF_SPEAKERS in dialogue_generate.py

function refSpeakerSlots(node) {
    const slots = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const name = node.inputs[i].name;
        if (name.startsWith(PREFIX)) {
            const n = parseInt(name.slice(PREFIX.length), 10);
            if (!Number.isNaN(n)) slots.push({ index: i, n, input: node.inputs[i] });
        }
    }
    return slots;
}

function findRefSpeakerInputType(node, n) {
    // Read the type straight off whatever slot currently exists for this
    // n, if any; otherwise fall back to the type of ref_speaker_1 (they
    // are all the same type, SA_LATENT).
    const existing = node.inputs.find((inp) => inp.name === `${PREFIX}${n}`);
    if (existing) return existing.type;
    const any = node.inputs.find((inp) => inp.name.startsWith(PREFIX));
    return any ? any.type : "SA_LATENT";
}

function ensureSlot(node, n) {
    if (n > MAX_REF_SPEAKERS) return;
    const name = `${PREFIX}${n}`;
    if (node.inputs.some((inp) => inp.name === name)) return;
    node.addInput(name, findRefSpeakerInputType(node, n));
}

function syncSlots(node) {
    const slots = refSpeakerSlots(node);
    if (slots.length === 0) return;

    let highestConnected = 0;
    for (const { n, input } of slots) {
        if (input.link != null) highestConnected = Math.max(highestConnected, n);
    }
    const wanted = Math.min(highestConnected + 1, MAX_REF_SPEAKERS);

    for (let n = 1; n <= wanted; n++) ensureSlot(node, n);

    // Remove trailing unconnected slots beyond `wanted`, highest index
    // first so removal doesn't shift the indices we still need to check.
    const toRemove = refSpeakerSlots(node)
        .filter(({ n, input }) => n > wanted && input.link == null)
        .sort((a, b) => b.index - a.index);
    for (const { index } of toRemove) {
        node.removeInput(index);
    }

    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ScenemaAudio.DialogueRefSpeakers",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            // Collapse to just ref_speaker_1 on a freshly placed node —
            // the Python schema declares all MAX_REF_SPEAKERS up front,
            // this trims the rest away immediately after creation.
            const toRemove = refSpeakerSlots(this)
                .filter(({ n, input }) => n > 1 && input.link == null)
                .sort((a, b) => b.index - a.index);
            for (const { index } of toRemove) this.removeInput(index);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
            const r = onConnectionsChange?.apply(this, arguments);
            // type === 1 is INPUT in LiteGraph's convention.
            if (type !== 1) return r;
            const slot = this.inputs[index];
            if (!slot || !slot.name.startsWith(PREFIX)) return r;
            syncSlots(this);
            return r;
        };
    },
});
