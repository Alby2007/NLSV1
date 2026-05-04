"""
Shared Ollama client — JSON mode, retry loop, consistent model config.

All phases import this instead of calling ollama directly.
"""

import json
import time
from pathlib import Path
from typing import Any

import ollama as _ollama

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT
from utils.logging import get_logger

logger = get_logger(__name__)

_STRICT_JSON_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON. No preamble, no explanation, "
    "no markdown fences. The first character of your response must be { or [."
)


class OllamaClient:
    """Wrapper around the Ollama Python client with JSON mode and retry logic."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = OLLAMA_TIMEOUT,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = _ollama.Client(host=host, timeout=timeout)

    def chat_json(self, prompt: str, context_label: str = "") -> dict | list | None:
        """
        Send a prompt expecting a JSON response.
        Retries up to max_retries times, escalating prompt strictness on failure.
        Returns parsed JSON (dict or list) or None if all retries fail.
        """
        current_prompt = prompt
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": current_prompt}],
                    format="json",
                )
                raw = response["message"]["content"].strip()
                parsed = json.loads(raw)
                if attempt > 1:
                    logger.info(f"[{context_label}] JSON parse succeeded on attempt {attempt}")
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{context_label}] Attempt {attempt}/{self.max_retries}: "
                    f"JSON decode failed — {e}. Raw: {raw[:120]!r}"
                )
                current_prompt = prompt + _STRICT_JSON_SUFFIX
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

            except Exception as e:
                logger.warning(
                    f"[{context_label}] Attempt {attempt}/{self.max_retries}: "
                    f"Ollama call failed — {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error(f"[{context_label}] All {self.max_retries} attempts failed. Returning None.")
        return None

    def chat_text(self, prompt: str, context_label: str = "") -> str | None:
        """
        Send a prompt expecting a plain-text response (no JSON mode).
        Returns response string or None on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response["message"]["content"].strip()
            except Exception as e:
                logger.warning(
                    f"[{context_label}] Attempt {attempt}/{self.max_retries}: "
                    f"Ollama call failed — {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error(f"[{context_label}] All {self.max_retries} attempts failed. Returning None.")
        return None


def get_client(**kwargs) -> OllamaClient:
    """Return a shared OllamaClient instance with optional overrides."""
    return OllamaClient(**kwargs)
