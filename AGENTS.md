# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Votype is a single-package Python application providing global voice typing for Linux via offline ASR models. There are no databases, sub-services, or monorepo packages. The core daemon requires X11, audio hardware (arecord), GTK3, and system clipboard tools (xdotool/xclip) — none of which are available in headless Cloud Agent VMs. However, the **full test suite runs without any of these** because everything is mocked via pytest fixtures in `tests/conftest.py`.

### Running tests

```bash
pytest                          # all tests (real_model tests will timeout without models)
pytest -m "not real_model"      # skip tests requiring real ASR model downloads (~recommended)
```

- 120+ tests pass. 7 tests have **pre-existing failures** in the codebase (as of initial setup); these are not caused by the environment.
- 3 `real_model` tests timeout/error because they attempt to download ~1GB ASR models — skip them with `-m "not real_model"`.
- Timeout is 60s per test (configured in `pytest.ini`).

### Linting

No linter is configured in the repo. `ruff check .` can be used for basic Python linting (33 pre-existing issues as of initial setup).

### CLI commands (no daemon)

These commands work without X11 or audio hardware:

```bash
python voice_input.py models    # list available ASR models
python voice_input.py status    # show daemon/recording status
```

The `daemon`, `start`, `stop`, `toggle` commands require X11, GTK, and audio hardware and will not function in headless environments.

### Key files

- `voice_input.py` — main entry point (CLI, daemon, recording logic)
- `model_presets.py` — model configuration constants
- `model_configs.py` — model loading and inference classes (`ModelLoader`, `ModelInference`)
- `settings_dialog.py` — GTK settings dialog
- `tests/conftest.py` — all pytest fixtures (mocking, isolation, cleanup)
