#!/usr/bin/env python3
"""Anthropic Claude proxy script for Oracle Cloud.

Reads JSON from stdin, calls Claude Haiku 4.5 via anthropic SDK,
writes corrected text to stdout.

Usage:
    echo '{"system_prompt": "...", "user_input": "..."}' | python3 anthropic_proxy.py
    python3 anthropic_proxy.py --help
    python3 anthropic_proxy.py --test

Exit codes:
    0 = success (stdout = corrected text)
    1 = failure (stderr = error message)

Self-contained: no imports from voice_input project.
Requires: anthropic SDK (pip install anthropic)
"""

import json
import sys
import time

# Suppress Python warnings before any third-party imports to keep stderr clean
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path


ANTHROPIC_KEY_PATH: Path = Path("~/.config/claude.secret").expanduser()
DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS: int = 1024


def _trace(msg: str) -> None:
    """Write trace timing to stderr (captured by caller for diagnosis)."""
    print(f"[TRACE] {msg}", file=sys.stderr)


def _import_anthropic():
    """Lazy import anthropic SDK.

    Returns:
        (anthropic module, Anthropic class)

    Raises:
        ImportError: If anthropic SDK is not installed.
    """
    import anthropic
    from anthropic import Anthropic
    return anthropic, Anthropic


def _read_api_key() -> str:
    """Read Anthropic API key from ANTHROPIC_KEY_PATH.

    Returns:
        Stripped key string, guaranteed non-empty.

    Raises:
        FileNotFoundError: If key file does not exist.
        PermissionError: If key file cannot be read.
        ValueError: If key file is empty after stripping.
    """
    key = ANTHROPIC_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {ANTHROPIC_KEY_PATH}")
    return key


def print_help() -> None:
    """Print usage information."""
    print(
        "anthropic_proxy.py — Anthropic Claude proxy\n"
        "\n"
        "Usage:\n"
        "  echo '{\"system_prompt\": \"...\", \"user_input\": \"...\"}' | python3 anthropic_proxy.py\n"
        "  python3 anthropic_proxy.py --help    Show this help\n"
        "  python3 anthropic_proxy.py --test    Verify SDK import + API key file\n"
        "\n"
        "Stdin JSON fields:\n"
        "  system_prompt  System instruction for Claude\n"
        "  user_input     User text to process (REQUIRED, non-empty)\n"
        "  model          Claude model name (default: claude-haiku-4-5-20251001)\n"
        "  max_tokens     Max output tokens (default: 1024)\n"
        "\n"
        "Stdout: corrected text (plain text, no JSON wrapping)\n"
        "Exit 0 = success, exit 1 = failure (stderr has error message)"
    )


def run_test() -> None:
    """Verify SDK import and API key file. Calls sys.exit itself."""
    try:
        _import_anthropic()
    except ImportError as e:
        print(f"FAIL: anthropic SDK not installed: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        _read_api_key()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: SDK import + API key file readable.")
    sys.exit(0)


def main() -> None:
    """Read JSON from stdin, call Claude, write result to stdout."""
    # Parse flags
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--help", "-h"):
            print_help()
            sys.exit(0)
        elif arg == "--test":
            run_test()
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)

    # Read JSON from stdin
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    system_prompt = data.get("system_prompt", "")
    user_input = data.get("user_input", "")
    model = data.get("model", DEFAULT_MODEL)
    max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)

    if not user_input:
        print("Missing 'user_input' in JSON", file=sys.stderr)
        sys.exit(1)

    # Import SDK + read API key
    t_sdk = time.time()
    try:
        anthropic_module, Anthropic = _import_anthropic()
    except ImportError as e:
        print(f"anthropic SDK not installed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        api_key = _read_api_key()
    except Exception as e:
        print(f"API key error: {e}", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    _trace(f"sdk_init: {time.time() - t_sdk:.2f}s")

    # Call Anthropic Messages API
    t_api = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
    except Exception as e:
        print(f"Anthropic API error: {e}", file=sys.stderr)
        sys.exit(1)
    _trace(f"anthropic_api: {time.time() - t_api:.2f}s")

    # Extract text from response.content[0].text
    content = getattr(response, "content", None) or []
    if not content:
        print("Anthropic returned empty response", file=sys.stderr)
        sys.exit(1)

    text = getattr(content[0], "text", None)
    if not text:
        print("Anthropic returned empty response", file=sys.stderr)
        sys.exit(1)

    print(text.strip())
    sys.exit(0)


if __name__ == "__main__":
    main()
