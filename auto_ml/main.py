"""Backward compatible shim for the previous training entry point."""

from __future__ import annotations

import warnings

from .train import main as _train_main


def main() -> None:
    """Invoke the renamed training CLI, issuing a deprecation warning."""

    warnings.warn(
        "'auto_ml.main' is deprecated; use 'auto_ml.train' or the top-level 'train.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _train_main()


if __name__ == "__main__":
    main()
