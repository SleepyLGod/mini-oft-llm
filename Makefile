.PHONY: check fmt lint setup sync

# ---------- Environment (uv) ----------
setup:
	uv venv .venv --python 3.11
	uv sync --extra dev

sync:
	uv sync --extra dev

# ---------- Code quality ----------
check:
	python -m compileall src scripts

fmt:
	ruff format src scripts

lint:
	ruff check src scripts
