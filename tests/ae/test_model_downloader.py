#!/usr/bin/env python3
"""Small end-to-end regression for the public checkpoint downloader."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


root = Path(__file__).resolve().parents[2]
downloader = root / "scripts/ae/download_modelscope_checkpoint.py"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


with tempfile.TemporaryDirectory(prefix="rapidmoe-downloader-test-") as temporary:
    work = Path(temporary)
    remote = work / "remote"
    remote.mkdir()
    payloads = {
        "tiny.gguf.part-00000": b"RapidMoE-public-deposit-",
        "tiny.gguf.part-00001": b"downloader-contract\n",
    }
    for name, payload in payloads.items():
        (remote / name).write_bytes(payload)

    complete = b"".join(payloads.values())
    manifest = {
        "repository": {
            "status": "public",
            "repo_id": "example/tiny",
            "revision": "1" * 40,
            "primary_endpoint": "https://modelscope.cn",
        },
        "checkpoint": {
            "logical_filename": "tiny.gguf",
            "size_bytes": len(complete),
            "sha256": digest(complete),
            "parts": [
                {"filename": name, "size_bytes": len(payload), "sha256": digest(payload)}
                for name, payload in payloads.items()
            ],
        },
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    log = work / "download.log"
    fake_client = work / "ms-hub-fake"
    fake_client.write_text(
        """#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[0:3] == ["--endpoint", "https://modelscope.cn", "download"], args
assert args[3] == "example/tiny", args
name = args[4]
assert args[5:7] == ["--revision", "1111111111111111111111111111111111111111"], args
assert args[7] == "--local-dir", args
destination = Path(args[8])
destination.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(os.environ["RAPIDMOE_FAKE_REMOTE"]) / name, destination / name)
with Path(os.environ["RAPIDMOE_FAKE_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(name + "\\n")
""",
        encoding="utf-8",
    )
    fake_client.chmod(0o755)

    environment = os.environ.copy()
    environment["RAPIDMOE_FAKE_REMOTE"] = str(remote)
    environment["RAPIDMOE_FAKE_LOG"] = str(log)

    output = work / "fresh"
    subprocess.run(
        [
            sys.executable,
            str(downloader),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--client",
            str(fake_client),
        ],
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    assert (output / "tiny.gguf").read_bytes() == complete
    assert not (output / "tiny.gguf.partial").exists()
    assert log.read_text(encoding="utf-8").splitlines() == list(payloads)

    # A partial file ending at a verified part boundary resumes at the next
    # part and does not download the completed prefix again.
    log.write_text("", encoding="utf-8")
    resumed = work / "resumed"
    resumed.mkdir()
    (resumed / "tiny.gguf.partial").write_bytes(payloads["tiny.gguf.part-00000"])
    subprocess.run(
        [
            sys.executable,
            str(downloader),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(resumed),
            "--client",
            str(fake_client),
        ],
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    assert (resumed / "tiny.gguf").read_bytes() == complete
    assert log.read_text(encoding="utf-8").splitlines() == ["tiny.gguf.part-00001"]

print("[PASS] checkpoint downloader fresh and boundary-resume paths")
