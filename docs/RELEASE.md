# Release Process

Releases are fully automated via [semantic-release](https://semantic-release.gitbook.io/) triggered by pushes to `main`. You do not need to manually bump versions or publish to PyPI.

## How it works

The [release workflow](.github/workflows/release.yml) runs on every push to `main` and:

1. Analyzes commit messages since the last release
2. Determines the next version (patch / minor / major) based on commit types
3. Updates `src/withoutbg/__version__.py`
4. Writes `CHANGELOG.md`
5. Creates a GitHub release and tag
6. Publishes to PyPI via trusted publisher (OIDC, no API token stored in secrets)

## Commit types and their effect on versioning

| Commit type | Version bump | Example |
|---|---|---|
| `feat:` | minor (1.0.x → 1.1.0) | `feat: add GPU acceleration` |
| `fix:` | patch (1.0.0 → 1.0.1) | `fix: handle rotated JPEG correctly` |
| `perf:` | patch | `perf: reduce inference memory` |
| `refactor:` | patch | `refactor: simplify model loading` |
| `feat!:` or `BREAKING CHANGE:` | major (1.x.x → 2.0.0) | `feat!: remove deprecated aliases` |
| `docs:`, `test:`, `chore:`, `ci:`, `build:` | no release | `docs: add batch example` |

If no release-triggering commits are present since the last tag, the release job exits cleanly with no publish.

## Manual fallback

If you need to publish manually (e.g. CI is unavailable):

```bash
# 1. Run quality checks and build
make publish-check

# 2. Test on TestPyPI first
make publish-test
pip install --index-url https://test.pypi.org/simple/ withoutbg

# 3. Publish to PyPI
make publish
```

Authentication uses `uv publish`. Set credentials via:

```bash
export UV_PUBLISH_USERNAME="__token__"
export UV_PUBLISH_PASSWORD="pypi-..."
```

## Deprecation policy

Deprecated public API symbols (e.g. `WithoutBG.opensource()`, `ProAPI`) emit `DeprecationWarning` and are documented in [docs/MIGRATION.md](MIGRATION.md). They are removed only in the next **major** release.

## Checking the current version

```bash
make version-show
# or:
python -c "import withoutbg; print(withoutbg.__version__)"
```
