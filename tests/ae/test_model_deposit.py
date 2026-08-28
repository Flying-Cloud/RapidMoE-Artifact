#!/usr/bin/env python3
"""Dependency-free validation of the external model deposit contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


root = Path(__file__).resolve().parents[2]
path = root / "artifact/model-deposit-manifest.json"
data = json.loads(path.read_text(encoding="utf-8"))
repo = data["repository"]
checkpoint = data["checkpoint"]
one_layer = data["one_layer"]

assert data["schema_version"] == 1
assert repo["provider"] == "ModelScope"
assert repo["status"] in {"pending-public-upload", "public"}
assert checkpoint["logical_filename"] == "DeepSeek-V3-0324-RES.gguf"
assert checkpoint["size_bytes"] == 427535921888
assert checkpoint["transport_chunk_size_bytes"] <= 100 * 1024**3
assert one_layer["layer"] == 38
assert len(one_layer["files"]) == 2
assert {item["role"] for item in one_layer["files"]} == {
    "rapidmoe_resplit",
    "kexpertscpu_q4_k_m",
}
for item in one_layer["files"]:
    assert 0 < item["size_bytes"] < 8 * 1024**3
    assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
    assert item["local_subdirectory"] in {"RES", "Q4_K_M"}

if repo["status"] == "public":
    assert repo["repo_id"] and repo["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", repo["revision"])
    assert repo["access_validation"]["primary_anonymous"] == "passed"
    assert checkpoint["parts"]
    assert sum(part["size_bytes"] for part in checkpoint["parts"]) == checkpoint["size_bytes"]
    assert re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
    for index, part in enumerate(checkpoint["parts"]):
        assert part["filename"].endswith(f"part-{index:05d}")
        assert 0 < part["size_bytes"] <= 100 * 1024**3
        assert re.fullmatch(r"[0-9a-f]{64}", part["sha256"])
    print("[PASS] public ModelScope deposit is immutable and checksum-complete")
else:
    assert not repo["repo_id"] and not repo["revision"]
    print("[INFO] ModelScope deposit is pending; real-weight tiers are not yet downloadable")

assert data["licenses"]["artifact_code"] == "Apache-2.0"
assert data["licenses"]["deepseek_derived_checkpoint"] == "MIT"
print("[PASS] external-model schema and license boundary")
print("[PASS] Experiment 2/3 one-layer weight manifest")
