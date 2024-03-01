from pathlib import Path

import ktp_classification

# Project Directories
PACKAGE_ROOT = Path(ktp_classification.__file__).resolve().parent

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
