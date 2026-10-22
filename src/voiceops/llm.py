"""OpenAI-compatible chat and embedding clients."""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from .settings import Settings


class ModelGateway:
    """One small adapter for hosted or OpenAI-compatible model endpoints."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise RuntimeError(
                "Set LLM_API_KEY in .env before calling the LLM. "
                "For a local OpenAI-compatible server, a non-empty placeholder key is accepted."
            )
        self.settings = settings
        self.chat_client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.embedding_client = OpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or settings.llm_base_url,
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None):
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 700,
        }
        local_ollama = any(
            host in self.settings.llm_base_url.lower()
            for host in ("localhost:11434", "127.0.0.1:11434")
        )
        if local_ollama and self.settings.llm_model.lower().startswith("qwen3"):
            # Ollama uses its `think` request field; reasoning_effort is ignored by its
            # OpenAI-compatible endpoint and Qwen3 may otherwise expose long <think> blocks.
            request["extra_body"] = {"think": False}
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        response = self.chat_client.chat.completions.create(**request)
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            safe_content = re.sub(
                r"<think\b[^>]*>.*?(?:</think\s*>|$)",
                "",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            if safe_content != content:
                message = message.model_copy(update={"content": safe_content})
        return message

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [list(row.embedding) for row in response.data]
