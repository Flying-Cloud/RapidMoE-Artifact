#!/usr/bin/env python3
"""Capture the machine-readable environment for an evaluator check."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--check", required=True)
args = parser.parse_args()
packages = {}
for name in (
    "torch", "transformers", "triton", "flashinfer-python", "fastapi",
    "uvicorn", "numpy", "pydantic", "accelerate", "sentencepiece", "openai",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = "MISSING"
record = {
    "schema_version": 1,
    "status": "CAPTURED",
    "check": args.check,
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": packages,
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "nvcc": command("nvcc", "--version"),
    "nvidia_smi": command(
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader",
    ),
    "gcc": command("gcc", "--version").splitlines()[0],
    "cmake": command("cmake", "--version").splitlines()[0],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
