#!/usr/bin/env python3
"""Standard-library smoke client for the sole public AE endpoint."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:10002")
parser.add_argument("--model", required=True, choices=("deepseek-v3",))
parser.add_argument("--output", type=Path)
args = parser.parse_args()
body = json.dumps({
    "model": args.model,
    "messages": [{"role": "user", "content": "Reply with exactly: RapidMoE AE OK"}],
    "max_tokens": 32,
    "temperature": 0,
}).encode()
request = urllib.request.Request(
    args.base_url.rstrip("/") + "/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
started = time.perf_counter()
# The AE endpoint is local. Do not let host-wide HTTP_PROXY settings route a
# loopback validation request through an unrelated proxy service.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=600) as response:
    payload = json.load(response)
assert payload.get("choices"), payload
content = payload["choices"][0]["message"]["content"].strip()
assert content == "RapidMoE AE OK", content
result = {
    "status": "PASS",
    "endpoint": "/v1/chat/completions",
    "model": args.model,
    "elapsed_seconds": time.perf_counter() - started,
    "response": payload,
}
text = json.dumps(result, indent=2) + "\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
print(text, end="")
