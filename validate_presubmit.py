# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local pre-submission validator for the smart traffic project."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], label: str) -> None:
    print(f"== {label} ==")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> None:
    required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        print(f"Missing environment variables for LLM-backed inference: {', '.join(missing)}")
        print("Continuing with local checks; inference.py will fall back to heuristic mode.")

    run([sys.executable, "-m", "pytest", "-q"], "pytest")
    run(["openenv", "validate", ".", "--verbose"], "openenv validate")
    run([sys.executable, "inference.py"], "inference")


if __name__ == "__main__":
    main()
