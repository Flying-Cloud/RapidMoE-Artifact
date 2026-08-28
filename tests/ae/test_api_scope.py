#!/usr/bin/env python3
"""Verify that the AE server exposes only scoped OpenAI discovery/chat routes."""

import asyncio

from ktransformers.server.api import router
from ktransformers.server.api.openai.endpoints.chat import chat_completion
from ktransformers.server.config.config import Config
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate


paths = sorted(route.path for route in router.routes)
assert paths == ["/v1/chat/completions", "/v1/models"], paths

Config().rapidmoe_mode = "dynamic"
invalid = ChatCompletionCreate(
    model="qwen3",
    messages=[{"role": "user", "content": "not in AE scope"}],
)
response = asyncio.run(chat_completion(None, invalid))
assert response.status_code == 400
assert b"deepseek-v3" in response.body

print(f"[PASS] scoped API routes: {paths}")
print("[PASS] non-V3 model rejected by the dynamic public endpoint")
