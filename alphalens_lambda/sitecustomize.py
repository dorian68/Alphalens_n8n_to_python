"""Local runtime patches to improve robustness without changing app.py.

This module is auto-imported by Python if present on sys.path.
We use it to wrap Cerebras streaming calls so failures return an empty stream
instead of raising, which prevents UnboundLocalError in app.py.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
from typing import Any


def _wrap_cerebras() -> None:
    try:
        from cerebras.cloud.sdk import Cerebras  # type: ignore
    except Exception:
        return

    logger = logging.getLogger("cerebras_safe")

    orig_init = Cerebras.__init__

    def _safe_create(inner_create):
        def _wrapped(*args, **kwargs):
            try:
                return inner_create(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cerebras create failed; returning empty stream: %s", exc)
                return []
        return _wrapped

    def _init(self, *args, **kwargs):  # type: ignore[no-redef]
        orig_init(self, *args, **kwargs)
        try:
            chat = getattr(self, "chat", None)
            completions = getattr(chat, "completions", None)
            create = getattr(completions, "create", None)
        except Exception:
            return
        if not callable(create):
            return
        wrapped = _safe_create(create)
        try:
            completions.create = wrapped
            return
        except Exception:
            # Fallback if attribute is read-only: proxy completions.
            try:
                class _SafeCompletions:
                    def __init__(self, inner):
                        self._inner = inner

                    def create(self, *a, **kw):
                        return wrapped(*a, **kw)

                    def __getattr__(self, name):
                        return getattr(self._inner, name)

                chat.completions = _SafeCompletions(completions)
            except Exception:
                return

    Cerebras.__init__ = _init  # type: ignore[assignment]


_wrap_cerebras()


def _patch_app_module(module: Any) -> None:
    logger = logging.getLogger("app_safe")
    func = getattr(module, "data_collection_llm_agent", None)
    if func is None:
        return
    if not callable(func):
        return
    if getattr(func, "__app_safe_wrapped__", False):
        return

    async def _safe_data_collection(state):  # type: ignore[no-untyped-def]
        try:
            return await func(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("data_collection_llm_agent failed; returning empty articles: %s", exc)
            return {"articles": ""}

    _safe_data_collection.__app_safe_wrapped__ = True  # type: ignore[attr-defined]
    module.data_collection_llm_agent = _safe_data_collection


class _AppPatchLoader(importlib.abc.Loader):
    def __init__(self, base_loader: importlib.abc.Loader) -> None:
        self._base_loader = base_loader

    def create_module(self, spec):  # type: ignore[override]
        if hasattr(self._base_loader, "create_module"):
            return self._base_loader.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module):  # type: ignore[override]
        self._base_loader.exec_module(module)  # type: ignore[misc]
        _patch_app_module(module)


class _AppPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[override]
        if fullname != "app":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _AppPatchLoader(spec.loader)
        return spec


def _install_app_patch() -> None:
    if "app" in sys.modules:
        _patch_app_module(sys.modules["app"])
        return
    sys.meta_path.insert(0, _AppPatchFinder())


_install_app_patch()
