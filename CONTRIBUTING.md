# Contributing to withoutbg

Contributions are welcome. This document covers everything you need to get from zero to a merged PR.

## Quick setup

```bash
git clone https://github.com/withoutbg/withoutbg
cd withoutbg

# Recommended: install with uv
uv sync --extra dev

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

Verify everything works:

```bash
make test-fast    # fast unit tests only
make quality      # lint + format check + type check
```

## Development workflow

1. Fork and clone the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make changes with tests
4. Run `make quality` and `make test-fast` — both must pass
5. Commit with a [conventional commit](#commit-messages) message
6. Open a pull request against `main`

## Commit messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>[optional scope]: <description>
```

Types that matter for releases:

| Type | Effect | Example |
|------|--------|---------|
| `feat` | minor version bump | `feat: add GPU support` |
| `fix` | patch version bump | `fix: handle RGBA inputs correctly` |
| `perf` | patch version bump | `perf: reduce model load time` |
| `refactor` | patch version bump | `refactor: extract preprocessing step` |
| `docs` | no release | `docs: update CLI examples` |
| `test` | no release | `test: add edge case for empty image` |
| `chore` | no release | `chore: update dev dependencies` |
| `feat!` or `BREAKING CHANGE:` | major version bump | |

See [`.github/COMMIT_CONVENTION.md`](.github/COMMIT_CONVENTION.md) for the full reference.

## Coding standards

- **Formatting:** [Black](https://black.readthedocs.io/) at 88 characters — run `make format`
- **Linting:** [Ruff](https://docs.astral.sh/ruff/) — run `make lint`
- **Types:** All new public functions need type annotations
- **Docstrings:** Google-style for all public symbols

Example:

```python
def remove_background(
    input_image: Union[str, Path, Image.Image],
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Image.Image:
    """Remove the background from an image.

    Args:
        input_image: File path or PIL Image.
        progress_callback: Optional function called with progress in [0, 1].

    Returns:
        PIL Image in RGBA mode with the background removed.

    Raises:
        WithoutBGError: If processing fails.
    """
```

## Tests

```bash
make test           # all tests (requires model download on first run)
make test-fast      # unit tests only, no model download needed
make test-unit      # same as test-fast
make test-coverage  # full suite with HTML coverage report
```

Tests are marked with pytest markers:

| Marker | When it runs | Use for |
|--------|-------------|---------|
| `unit` | always | fast, isolated, mocked |
| `integration` | `--run-real-processing` | multi-component, real images |
| `performance` | `--run-performance` | benchmarks, memory checks |

Write unit tests for all new code. Mock file I/O and model calls in unit tests:

```python
@pytest.mark.unit
class TestMyFeature:
    def test_does_the_thing(self):
        # Arrange
        model = MagicMock(spec=OpenWeightsModel)
        model.remove_background.return_value = Image.new("RGBA", (100, 100))

        # Act / Assert
        ...
```

Coverage gate is 80%. New code should not lower it.

## Pull requests

Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). Keep PRs focused — one logical change per PR makes review easier and faster.

## Reporting bugs

Open an [issue](https://github.com/withoutbg/withoutbg/issues) using the bug report template. Include:

- Python version and OS
- Whether you are using **Local** (`open_weights()`) or **Cloud** (`api()`)
- Minimal repro snippet

For Docker / self-hosted inference issues, open an issue in [withoutbg-inference](https://github.com/withoutbg/withoutbg-inference) instead.

## Security issues

Do not create public issues for security vulnerabilities. Email [security@withoutbg.com](mailto:security@withoutbg.com) instead.

## License

By contributing, you agree your contributions are licensed under the [Apache License 2.0](LICENSE).
