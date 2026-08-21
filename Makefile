.PHONY: install test lint typecheck check notebooks validate zip

install:
	poetry install --with dev,quantization

test:
	poetry run pytest --cov=qwen_text2sql --cov-report=term-missing

lint:
	poetry run ruff check src tests scripts

typecheck:
	poetry run mypy src/qwen_text2sql

check: lint typecheck test

notebooks:
	poetry run python scripts/validate_notebooks.py

validate:
	poetry run python scripts/validate_results.py --results-dir results
