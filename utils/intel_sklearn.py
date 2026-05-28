"""Optional Intel Extension for Scikit-learn GPU/CPU acceleration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_USE_INTEL: bool | None = None


def intel_available() -> bool:
    global _USE_INTEL
    if _USE_INTEL is not None:
        return _USE_INTEL
    try:
        from sklearnex import patch_sklearn  # noqa: F401

        _USE_INTEL = True
    except ImportError:
        _USE_INTEL = False
    return _USE_INTEL


def patch_if_available() -> bool:
    if not intel_available():
        return False
    from sklearnex import patch_sklearn

    patch_sklearn()
    return True


@contextmanager
def sklearn_fit_context(*, use_gpu: bool = True, cpu_only: bool = False) -> Iterator[None]:
    """Patch sklearn and optionally offload fit/predict to Intel GPU."""
    if cpu_only or not patch_if_available():
        yield
        return
    if use_gpu:
        from sklearnex import config_context

        with config_context(target_offload="gpu:0"):
            yield
    else:
        yield
