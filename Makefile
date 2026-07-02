# Makefile for withoutbg

.PHONY: help install dev-install test test-unit test-integration test-performance test-coverage lint format type-check clean build docs version-show version-patch version-minor version-major publish publish-test publish-check

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install the package"
	@echo "  dev-install      Install package in development mode with dev dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-api         Run API tests"
	@echo "  test-fast        Run fast tests (skip integration & performance)"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-verbose     Run tests with verbose output"
	@echo "  test-quiet       Run tests with minimal output"
	@echo "  test-parallel    Run tests in parallel"
	@echo "  lint             Run linting (ruff)"
	@echo "  format           Format code (black)"
	@echo "  format-check     Check code formatting"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  quality          Run all quality checks (lint + format + type)"
	@echo "  fix              Auto-fix linting issues and format code"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  debug-env        Show environment information"
	@echo "  security         Run security checks"
	@echo "  deps-check       Check for outdated dependencies"
	@echo "  version-show     Show current version"
	@echo "  version-patch    Increment patch version (0.1.0 -> 0.1.1)"
	@echo "  version-minor    Increment minor version (0.1.0 -> 0.2.0)"
	@echo "  version-major    Increment major version (0.1.0 -> 1.0.0)"
	@echo "  publish-check    Check package before publishing (quality + build)"
	@echo "  publish-test     Publish to TestPyPI"
	@echo "  publish          Publish to PyPI (production)"
	@echo "  dev              Quick development setup and checks"
	@echo "  ci               Simulate CI pipeline"

# Installation targets
install:
	uv sync

dev-install:
	uv sync --dev

# Testing targets
test:
	uv run pytest --run-real-processing

test-unit:
	uv run pytest -m "unit"

test-integration:
	uv run pytest -m "integration" --run-real-processing

test-performance:
	uv run pytest -m "performance" --run-performance

test-api:
	uv run pytest -m "api" --run-api

test-coverage:
	uv run pytest --cov=src/withoutbg --cov-report=html --cov-report=term-missing --run-real-processing

test-fast:
	uv run pytest -m "not (performance or integration)" -x

test-verbose:
	uv run pytest -v

test-quiet:
	uv run pytest -q

test-parallel:
	uv run pytest -n auto

# Code quality targets
lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/

format-check:
	uv run black --check src/ tests/

type-check:
	uv run mypy src/

quality: lint format-check type-check
	@echo "✓ All quality checks passed!"

# Fix common issues
fix:
	uv run ruff check --fix src/ tests/
	uv run black src/ tests/

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Development workflow
dev: dev-install quality test-fast
	@echo "✓ Development setup complete!"

# CI/CD simulation
ci: dev-install quality test-coverage
	@echo "✓ CI checks completed!"

# Documentation
docs:
	@echo "See README.md and the docs/ directory for documentation."
	@echo "Online: https://withoutbg.com/documentation/integrations/python-sdk"

# Performance benchmarking
benchmark:
	uv run pytest tests/performance/ --run-performance -v

# Debug helpers
debug-env:
	@echo "Python version: $$(python --version)"
	@echo "UV version: $$(uv --version)"
	@echo "Package info:"
	@uv run python -c "import withoutbg; print(f'withoutbg version: {withoutbg.__version__}')" || echo "Package not installed"

# Security and dependency checks
security:
	uv run pip-audit

deps-check:
	uv run pip list --outdated

# Version management
version-show:
	@echo "Current version: $$(uv run hatch version)"

version-patch:
	uv run hatch version patch
	@echo "Version bumped to: $$(uv run hatch version)"

version-minor:
	uv run hatch version minor
	@echo "Version bumped to: $$(uv run hatch version)"

version-major:
	uv run hatch version major
	@echo "Version bumped to: $$(uv run hatch version)"

# Release helpers
release-patch: version-patch
	@echo "Patch release created: $$(uv run hatch version)"

release-minor: version-minor
	@echo "Minor release created: $$(uv run hatch version)"

release-major: version-major
	@echo "Major release created: $$(uv run hatch version)"

# Publishing targets (using modern tools)
publish-check: quality build
	@echo "✓ Pre-publish checks passed!"
	@echo "📦 Package built and ready for publishing"
	@echo ""
	@echo "Contents of dist/:"
	@ls -lh dist/
	@echo ""
	@echo "To publish to TestPyPI: make publish-test"
	@echo "To publish to PyPI:     make publish"

publish-test: publish-check
	@echo "📤 Publishing to TestPyPI..."
	@echo ""
	@echo "⚠️  You'll need to authenticate with TestPyPI"
	@echo "Set up your API token at: https://test.pypi.org/manage/account/token/"
	@echo ""
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo ""
	@echo "✓ Published to TestPyPI!"
	@echo "Test installation with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ withoutbg"

publish: publish-check
	@echo "📤 Publishing to PyPI..."
	@echo ""
	@echo "⚠️  PRODUCTION RELEASE - Are you sure? (Ctrl+C to cancel)"
	@echo "Version: $$(uv run hatch version)"
	@sleep 3
	@echo ""
	@echo "⚠️  You'll need to authenticate with PyPI"
	@echo "Set up your API token at: https://pypi.org/manage/account/token/"
	@echo ""
	uv publish
	@echo ""
	@echo "✓ Successfully published to PyPI!"
	@echo "Install with: pip install withoutbg"
	@echo "View at: https://pypi.org/project/withoutbg/"

