# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tiny shared helper for the one deliberate auto-fetch exception in this
package (the SeedVC DiT checkpoint, checked locally first — see
nodes/voice_clone_loader.py).

Everything else in ComfyUI-ScenemaAudio is local-only, enforced by setting
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 process-wide at import time (see
__init__.py). That guard needs to be lifted for the one spot that
legitimately needs to fetch something small and non-gated as a last
resort, or it'll fail with an "offline mode is enabled" error even
though fetching was the intended fallback behavior there.

IMPORTANT: just toggling the os.environ variable is NOT enough.
huggingface_hub reads HF_HUB_OFFLINE from the environment ONCE, at
import time, and caches it as a plain module-level boolean
(huggingface_hub.constants.HF_HUB_OFFLINE) — internal code checks that
cached boolean, not os.environ, on every call. By the time this context
manager ever runs, huggingface_hub has always already been imported
somewhere with HF_HUB_OFFLINE="1" already set (our __init__.py sets it
before any node code runs), so the cached boolean is already True and
popping the env var afterward has no effect on it whatsoever — this is
exactly what caused the "offline mode is enabled" failure even with this
context manager in place. The fix has to reach in and flip that cached
boolean directly, not just the environment variable that originally fed
it.
"""

import contextlib
import os

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# Everywhere HF_HUB_OFFLINE might exist as an already-cached module
# attribute, across huggingface_hub versions/internal reorganizations.
# Patched defensively — modules/attributes that don't exist in a given
# installed version are just skipped, not an error.
_POSSIBLE_OFFLINE_FLAG_MODULES = (
    "huggingface_hub.constants",
    "huggingface_hub.file_download",
    "huggingface_hub.utils._http",
    "huggingface_hub.utils._runtime",
)


@contextlib.contextmanager
def allow_network_for_one_fetch():
    """Temporarily lift the offline-mode guard for exactly one deliberate
    auto-fetch exception, restoring everything it touched on exit
    (success or failure) — both the environment variables and every
    cached HF_HUB_OFFLINE module attribute it found set to True.
    """
    saved_env = {k: os.environ.get(k) for k in _OFFLINE_ENV_VARS}
    for k in _OFFLINE_ENV_VARS:
        os.environ.pop(k, None)

    patched = []  # [(module, attr_name, old_value), ...]
    for mod_name in _POSSIBLE_OFFLINE_FLAG_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except ImportError:
            continue
        if getattr(mod, "HF_HUB_OFFLINE", None) is True:
            patched.append((mod, "HF_HUB_OFFLINE", True))
            setattr(mod, "HF_HUB_OFFLINE", False)

    try:
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for mod, attr, old_value in patched:
            setattr(mod, attr, old_value)

