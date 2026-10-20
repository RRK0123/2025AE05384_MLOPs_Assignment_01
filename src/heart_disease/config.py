"""Paths and evaluation settings shared by command-line stages."""

import os
from pathlib import Path

# Installed commands may run from any working directory. For a wheel/container,
# set the data-workspace root explicitly; inference will use its own model path.
ROOT = Path(
    os.environ.get("HEART_DISEASE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
SEED = 42
TEST_SIZE = 0.20
