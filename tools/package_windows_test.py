"""Package a prepared portable Windows directory with UTF-8 ZIP filenames.

Usage: python3 tools/package_windows_test.py STAGED_DIR OUTPUT.zip
The source directory itself is retained as the single top-level ZIP folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package(source: Path, output: Path) -> None:
    if not source.is_dir() or source.name != "Calibration-Windows-x64":
        raise ValueError("source must be the prepared Calibration-Windows-x64 directory")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6,
                 allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.name.startswith("._"):
                continue
            relative = (Path(source.name) / path.relative_to(source)).as_posix()
            if path.is_dir():
                archive.writestr(relative + "/", b"")
            elif path.is_file():
                archive.write(path, relative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.source, args.output)
