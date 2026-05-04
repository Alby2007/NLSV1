"""
Smoke test: verify Ollama is reachable and returns valid JSON.
Run this before starting any phase that calls Ollama.

Usage:
    python smoke_test_ollama.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.ollama_client import get_client
from utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Ollama smoke test starting...")
    client = get_client()

    result = client.chat_json(
        'Return this exact JSON: {"symbol": "TEST", "type": "Entity", "gloss": "a test", "confidence": 0.9}',
        context_label="smoke_test",
    )

    if result is None:
        logger.error("FAIL — chat_json returned None. Is Ollama running? Is the model pulled?")
        logger.error("Run: ollama serve  (in a separate terminal)")
        logger.error("Run: ollama pull qwen2.5:14b")
        sys.exit(1)

    expected_keys = {"symbol", "type", "gloss", "confidence"}
    missing = expected_keys - set(result.keys())
    if missing:
        logger.error(f"FAIL — response missing keys: {missing}")
        logger.error(f"Got: {result}")
        sys.exit(1)

    if result.get("type") != "Entity":
        logger.warning(f"Model didn't echo back 'Entity' exactly — got '{result.get('type')}'. "
                       "Minor instruction-following drift, acceptable for labelling tasks.")

    logger.info(f"PASS — Ollama responded correctly: {result}")

    logger.info("\nTesting chat_text...")
    text = client.chat_text("Reply with exactly: hello", context_label="smoke_test_text")
    if text is None:
        logger.error("FAIL — chat_text returned None.")
        sys.exit(1)
    logger.info(f"PASS — chat_text responded: {text!r}")

    logger.info("\nOllama smoke test PASSED. Ready to run pipeline phases.")


if __name__ == "__main__":
    main()
