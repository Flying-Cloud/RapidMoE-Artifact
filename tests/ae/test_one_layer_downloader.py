#!/usr/bin/env python3
"""Dependency-free regression for the one-layer ModelScope downloader."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


root = Path(__file__).resolve().parents[2]
downloader = root / "scripts/ae/download_one_layer_weights.py"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


with tempfile.TemporaryDirectory(prefix="rapidmoe-one-layer-download-") as temporary:
    work = Path(temporary)
    remote = work / "remote"
    remote.mkdir()
    payloads = {
        "layer-res.gguf": b"small-resplit-layer\n",
        "layer-q4.gguf": b"small-q4-layer\n",
    }
    for name, payload in payloads.items():
        (remote / name).write_bytes(payload)

    manifest = {
        "repository": {
            "status": "public",
            "repo_id": "example/one-layer",
            "revision": "2" * 40,
            "primary_endpoint": "https://modelscope.cn",
        },
        "one_layer": {
            "files": [
                {
                    "filename": name,
                    "local_subdirectory": directory,
                    "size_bytes": len(payloads[name]),
                    "sha256": digest(payloads[name]),
                }
                for name, directory in (
                    ("layer-res.gguf", "RES"),
                    ("layer-q4.gguf", "Q4_K_M"),
                )
            ]
        },
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fake_client = work / "ms-hub-fake"
    fake_client.write_text(
        """#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[0:3] == ["--endpoint", "https://modelscope.cn", "download"], args
assert args[3] == "example/one-layer", args
name = args[4]
assert args[5:7] == ["--revision", "2222222222222222222222222222222222222222"], args
assert args[7] == "--local-dir", args
destination = Path(args[8])
destination.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(os.environ["RAPIDMOE_FAKE_REMOTE"]) / name, destination / name)
""",
        encoding="utf-8",
    )
    fake_client.chmod(0o755)
    environment = os.environ.copy()
    environment["RAPIDMOE_FAKE_REMOTE"] = str(remote)
    output = work / "download"
    command = [
        sys.executable,
        str(downloader),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output),
        "--client",
        str(fake_client),
    ]
    subprocess.run(command, check=True, env=environment, stdout=subprocess.DEVNULL)
    assert (output / "RES/layer-res.gguf").read_bytes() == payloads["layer-res.gguf"]
    assert (output / "Q4_K_M/layer-q4.gguf").read_bytes() == payloads["layer-q4.gguf"]
    # The already-verified path must also succeed without re-downloading.
    subprocess.run(command, check=True, env=environment, stdout=subprocess.DEVNULL)

print("[PASS] one-layer downloader fresh and already-verified paths")
