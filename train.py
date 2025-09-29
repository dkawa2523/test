"""Project-level training entry point.

Thin wrapper so users can run ``python train.py --config config.yaml`` from the
project root while reusing the package CLI implementation.
"""

from __future__ import annotations

from auto_ml.train import main


if __name__ == "__main__":
    main()
