#!/usr/bin/env python3
"""Inspect a GGUF header without importing torch or mapping tensor payloads."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d", 13: "<B",
}
TYPE_NAMES = {
    0: "F32", 8: "Q8_0", 350: "IQ1_M_R4_RES_Q2_K",
    351: "IQ1_M_PLUS_Q4_K_R4",
}
CANONICAL_CUSTOM_COUNTS = {350: 116, 351: 58}


def read_value(handle, value_type: int):
    if value_type == 8:
        length = struct.unpack("<Q", handle.read(8))[0]
        return handle.read(length).decode("utf-8")
    if value_type == 9:
        element_type, count = struct.unpack("<IQ", handle.read(12))
        return [read_value(handle, element_type) for _ in range(count)]
    fmt = SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]


def inspect(path: Path) -> dict:
    counts: collections.Counter[int] = collections.Counter()
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", handle.read(20))
        metadata = {}
        for _ in range(metadata_count):
            key = read_value(handle, 8)
            metadata[key] = read_value(handle, struct.unpack("<I", handle.read(4))[0])
        for _ in range(tensor_count):
            read_value(handle, 8)
            dimensions = read_value(handle, 4)
            for _ in range(dimensions):
                read_value(handle, 10)
            tensor_type = read_value(handle, 4)
            read_value(handle, 10)
            counts[tensor_type] += 1
    return {
        "schema_version": 1,
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "architecture": metadata.get("general.architecture"),
        "quantization_version": metadata.get("general.quantization_version"),
        "tensor_type_counts": {
            f"{key}:{TYPE_NAMES.get(key, 'UNKNOWN')}": value for key, value in sorted(counts.items())
        },
        "canonical_resplit_layout": all(counts[key] == value for key, value in CANONICAL_CUSTOM_COUNTS.items()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sha256", action="store_true", help="hash the entire file (slow for the 398 GiB checkpoint)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.checkpoint)
    if args.sha256:
        report["sha256"] = sha256(args.checkpoint)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["canonical_resplit_layout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
