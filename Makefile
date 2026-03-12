.PHONY: check fmt lint

check:
	python -m compileall src scripts

fmt:
	python -m pip install ruff >/dev/null 2>&1 || true
	ruff format src scripts

lint:
	python -m pip install ruff >/dev/null 2>&1 || true
	ruff check src scripts
