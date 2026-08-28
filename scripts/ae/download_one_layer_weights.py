#!/usr/bin/env python3
"""Download and verify the two small GGUFs used by Experiments 2 and 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "artifact/model-deposit-manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--client", default="ms-hub")
    args = parser.parse_args()

    if shutil.which(args.client) is None:
        raise SystemExit(
            f"{args.client!r} not found; install it with "
            "python3 -m pip install 'modelscope-hub==0.2.0'"
        )

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    repo = data["repository"]
    deposit = data["one_layer"]
    if repo["status"] != "public" or not repo["revision"]:
        raise SystemExit("the pinned ModelScope deposit is not public")

    endpoint = args.endpoint or repo["primary_endpoint"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in deposit["files"]:
        destination_dir = args.output_dir / item["local_subdirectory"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / item["filename"]
        if destination.is_file():
            if destination.stat().st_size != item["size_bytes"]:
                raise SystemExit(f"existing {destination} has the wrong size")
            if sha256(destination) != item["sha256"]:
                raise SystemExit(f"existing {destination} has the wrong SHA-256")
            print(f"[PASS] already verified: {destination}")
            continue

        temporary_dir = args.output_dir / ".modelscope-download"
        command = [
            args.client,
            "--endpoint",
            endpoint,
            "download",
            repo["repo_id"],
            item["filename"],
            "--revision",
            repo["revision"],
            "--local-dir",
            str(temporary_dir),
        ]
        print("[RUN]", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        downloaded = temporary_dir / item["filename"]
        if not downloaded.is_file():
            raise SystemExit(f"download client did not create {downloaded}")
        if downloaded.stat().st_size != item["size_bytes"]:
            raise SystemExit(f"wrong size for {downloaded}")
        if sha256(downloaded) != item["sha256"]:
            raise SystemExit(f"wrong SHA-256 for {downloaded}")
        os.replace(downloaded, destination)
        print(f"[PASS] downloaded and verified: {destination}")

    print(f"[PASS] Experiment 2/3 one-layer weights: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
