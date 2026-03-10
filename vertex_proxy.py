#!/usr/bin/env python3
"""Vertex AI Gemini proxy script for Oracle Cloud.

Reads JSON from stdin, calls Gemini 2.5 Flash via google-genai SDK,
writes corrected text to stdout.

Usage:
    echo '{"system_prompt": "...", "user_input": "...", "model": "gemini-2.5-flash", "region": "us-central1"}' | python3 vertex_proxy.py
    python3 vertex_proxy.py --help
    python3 vertex_proxy.py --test

Exit codes:
    0 = success (stdout = corrected text)
    1 = failure (stderr = error message)

Self-contained: no imports from voice_input project.
Requires: google-genai SDK (pip install google-genai)
"""

import json
import sys

# Suppress Python warnings (e.g., RequestsDependencyWarning) before any
# google imports to prevent stderr pollution hiding real errors (CRITIC-R7-M3)
import warnings
warnings.filterwarnings('ignore')

# GCP project ID (not sensitive — public project identifier)
GCP_PROJECT = "project-9cd34e60-a50d-406f-ae4"


def _import_genai():
    """Lazy import google-genai SDK.

    Returns:
        (genai module, GenerateContentConfig class)

    Raises:
        ImportError: If google-genai is not installed.
    """
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig
    return genai, GenerateContentConfig, ThinkingConfig


def print_help():
    """Print usage information."""
    print(
        "vertex_proxy.py — Vertex AI Gemini proxy\n"
        "\n"
        "Usage:\n"
        "  echo '{\"system_prompt\": \"...\", \"user_input\": \"...\"}' | python3 vertex_proxy.py\n"
        "  python3 vertex_proxy.py --help    Show this help\n"
        "  python3 vertex_proxy.py --test    Verify SDK import + ADC credentials\n"
        "\n"
        "Stdin JSON fields:\n"
        "  system_prompt  System instruction for Gemini\n"
        "  user_input     User text to process\n"
        "  model          Gemini model name (default: gemini-2.5-flash)\n"
        "  region         Vertex AI region (default: us-central1)\n"
        "\n"
        "Stdout: corrected text (plain text, no JSON wrapping)\n"
        "Exit 0 = success, exit 1 = failure (stderr has error message)"
    )


def run_test():
    """Verify SDK import and ADC credentials.

    Uses client.models.list() for true ADC validation —
    genai.Client() init is lazy and succeeds even with expired credentials (CRITIC-R8-L1).
    """
    try:
        genai, _ = _import_genai()
        client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location="us-central1",
        )
        # Actually validate credentials by making a lightweight API call
        models = list(client.models.list(config={"page_size": 1}))
        print(f"OK: SDK import + ADC auth verified. Found {len(models)} model(s).")
        sys.exit(0)
    except ImportError as e:
        print(f"FAIL: google-genai SDK not installed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Read JSON from stdin, call Gemini, write result to stdout."""
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
    model = data.get("model", "gemini-2.5-flash")
    region = data.get("region", "us-central1")

    if not user_input:
        print("Missing 'user_input' in JSON", file=sys.stderr)
        sys.exit(1)

    # Import SDK (lazy to allow --help without SDK installed)
    try:
        genai, GenerateContentConfig, ThinkingConfig = _import_genai()
    except ImportError as e:
        print(f"google-genai SDK not installed: {e}", file=sys.stderr)
        sys.exit(1)

    # Create Vertex AI client (CRITIC-R6-C1, R6-C2)
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=region,
    )

    # Call Gemini (thinking disabled via thinking_budget=0 for low latency)
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        thinking_config=ThinkingConfig(thinking_budget=0),
        temperature=0.3,
        max_output_tokens=512,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_input,
            config=config,
        )
    except Exception as e:
        print(f"Gemini API error: {e}", file=sys.stderr)
        sys.exit(1)

    if response.text is None:
        print("Gemini returned empty response (response.text is None)", file=sys.stderr)
        sys.exit(1)

    print(response.text.strip())


if __name__ == "__main__":
    main()
