"""Canonical repository paths for the packaged Campus6 project."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
REPOSITORY_ROOT = Path(os.environ.get("DAHUA_CODE_ROOT", PROJECT_ROOT)).expanduser().resolve()
PROTOGCN_ROOT = REPOSITORY_ROOT / "third_party" / "ProtoGCN"
CONFIG_ROOT = PACKAGE_ROOT / "configs"
