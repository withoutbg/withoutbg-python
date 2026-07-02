# Commit Convention

This repository uses [Conventional Commits](https://www.conventionalcommits.org/).

## Format

```
<type>[optional scope]: <short description>

[optional body]

[optional footer(s)]
```

The short description must be ≤72 characters and written in the imperative mood ("add", "fix", "remove" — not "added", "fixes", "removed").

## Types

| Type | Release effect | Use for |
|------|---------------|---------|
| `feat` | minor bump | New user-facing feature |
| `fix` | patch bump | Bug fix |
| `perf` | patch bump | Performance improvement |
| `refactor` | patch bump | Code restructure, no behavior change |
| `revert` | patch bump | Reverting a previous commit |
| `docs` | none | Documentation only |
| `test` | none | Adding or fixing tests |
| `style` | none | Whitespace, formatting (no logic change) |
| `chore` | none | Maintenance, dependencies, tooling |
| `build` | none | Build system changes |
| `ci` | none | CI/CD changes |

## Breaking changes

Append `!` to the type or include `BREAKING CHANGE:` in the footer to trigger a major version bump:

```
feat!: remove deprecated opensource() alias

BREAKING CHANGE: WithoutBG.opensource() has been removed.
Use WithoutBG.open_weights() instead.
```

## Examples

```
feat: add progress_callback to remove_background_batch
fix: apply EXIF orientation before inference
perf: cache ONNX session across batch calls
docs: add examples for batch processing
test: add unit test for RGBA image input
chore: update onnxruntime to 1.18.0
refactor: extract preprocessing to dedicated function
```

## Scope (optional)

Scope narrows the context:

```
feat(cli): add --format webp output option
fix(api): retry on 429 rate limit response
```

Valid scopes: `cli`, `api`, `models`, `core`, `docs`, `ci`.

## Pre-commit hook

The conventional-pre-commit hook validates commit messages automatically when you run `pre-commit install`. It will reject commits that don't match the format.
