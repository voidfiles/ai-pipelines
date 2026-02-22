default:
    @just --list

test *args:
    uv run pytest {{args}}

test-v *args:
    uv run pytest -v {{args}}

lint:
    uv run python -m py_compile src/ai_pipelines/__init__.py

sync:
    uv sync --all-extras
