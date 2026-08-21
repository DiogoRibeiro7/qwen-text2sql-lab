.DEFAULT_GOAL := help
.PHONY: help install install-cpu hooks hooks-push format format-check lint typecheck test test-fast check notebooks validate build clean

POETRY ?= poetry
RUN    := $(POETRY) run
SOURCES := src tests scripts notebooks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies, including 4-bit quantization support
	$(POETRY) install --with dev,notebooks,quantization

install-cpu: ## Install without the quantization group (no CUDA GPU)
	$(POETRY) install --with dev,notebooks

hooks: ## Install the fast pre-commit git hooks
	$(RUN) pre-commit install

hooks-push: hooks ## Also gate `git push` on mypy, tests and notebook validation
	$(RUN) pre-commit install --hook-type pre-push

format: ## Apply Ruff formatting and safe lint fixes
	$(RUN) ruff format $(SOURCES)
	$(RUN) ruff check --fix $(SOURCES)

format-check: ## Verify formatting without modifying files
	$(RUN) ruff format --check --diff $(SOURCES)

lint: ## Run Ruff lint rules
	$(RUN) ruff check $(SOURCES)

typecheck: ## Run mypy in strict mode over the package and the entry-point scripts
	$(RUN) mypy src/qwen_text2sql scripts

test: ## Run the test suite with branch coverage
	$(RUN) pytest --cov=qwen_text2sql --cov-report=term-missing

test-fast: ## Run the test suite, skipping slow and GPU tests
	$(RUN) pytest -m "not slow and not gpu"

check: lint format-check typecheck test ## Run every quality gate CI runs

notebooks: ## Statically validate the notebook sequence
	$(RUN) python scripts/validate_notebooks.py

validate: ## Validate the contents of the results directory
	$(RUN) python scripts/validate_results.py --results-dir results

build: ## Build the sdist and wheel
	$(POETRY) build

clean: ## Remove build, cache and coverage artifacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
