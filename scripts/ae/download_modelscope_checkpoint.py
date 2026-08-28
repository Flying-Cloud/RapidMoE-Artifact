#!/usr/bin/env python3
"""Download, verify and reconstruct the ModelScope RESplit checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    repo = data["repository"]
    checkpoint = data["checkpoint"]
    if repo["status"] != "public" or not repo["repo_id"] or not repo["revision"]:
        raise SystemExit(
            "ModelScope deposit is not public yet; fill repository.repo_id, "
            "repository.revision and set repository.status=public in "
            f"{path} after upload verification"
        )
    if not checkpoint["parts"] or checkpoint["sha256"].startswith("PENDING"):
        raise SystemExit(f"checkpoint part hashes are incomplete in {path}")
    return data


def download_part(
    client: str, endpoint: str, repo_id: str, revision: str, name: str, directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        client,
        "--endpoint",
        endpoint,
        "download",
        repo_id,
        name,
        "--revision",
        revision,
        "--local-dir",
        str(directory),
    ]
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    path = directory / name
    if not path.is_file():
        raise SystemExit(f"download client did not create {path}")
    return path


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
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    if shutil.which(args.client) is None:
        raise SystemExit(
            f"{args.client!r} not found; install the pinned client with "
            "python3 -m pip install 'modelscope-hub==0.2.0'"
        )

    data = load_manifest(args.manifest)
    repo = data["repository"]
    checkpoint = data["checkpoint"]
    endpoint = args.endpoint or repo["primary_endpoint"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / checkpoint["logical_filename"]
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    staging = args.output_dir / ".rapidmoe-modelscope-parts"

    if final_path.exists():
        if final_path.stat().st_size != checkpoint["size_bytes"]:
            raise SystemExit(f"existing {final_path} has the wrong size")
        if sha256(final_path) != checkpoint["sha256"]:
            raise SystemExit(f"existing {final_path} has the wrong SHA-256")
        print(f"[PASS] checkpoint already verified: {final_path}")
        return 0

    cumulative = 0
    current_size = partial_path.stat().st_size if partial_path.exists() else 0
    boundaries = {0}
    for part in checkpoint["parts"]:
        cumulative += part["size_bytes"]
        boundaries.add(cumulative)
    if current_size not in boundaries:
        raise SystemExit(
            f"partial output size {current_size} is not a verified part boundary; "
            f"move {partial_path} aside and restart"
        )

    cumulative = 0
    for part in checkpoint["parts"]:
        previous = cumulative
        cumulative += part["size_bytes"]
        if current_size >= cumulative:
            print(f"[SKIP] already assembled: {part['filename']}")
            continue
        if current_size != previous:
            raise SystemExit("partial output does not end at the expected boundary")
        part_path = download_part(
            args.client,
            endpoint,
            repo["repo_id"],
            repo["revision"],
            part["filename"],
            staging,
        )
        if part_path.stat().st_size != part["size_bytes"]:
            raise SystemExit(f"wrong size for {part_path}")
        if sha256(part_path) != part["sha256"]:
            raise SystemExit(f"wrong SHA-256 for {part_path}")
        with partial_path.open("ab") as output, part_path.open("rb") as source:
            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        current_size = partial_path.stat().st_size
        if current_size != cumulative:
            raise SystemExit("assembled file ended at an unexpected byte offset")
        print(f"[PASS] verified and assembled: {part['filename']}")
        if not args.keep_parts:
            part_path.unlink()

    if partial_path.stat().st_size != checkpoint["size_bytes"]:
        raise SystemExit("reconstructed checkpoint has the wrong size")
    if sha256(partial_path) != checkpoint["sha256"]:
        raise SystemExit("reconstructed checkpoint has the wrong SHA-256")
    os.replace(partial_path, final_path)
    print(f"[PASS] reconstructed checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
