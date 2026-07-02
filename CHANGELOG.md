# Changelog

All notable changes to this project will be documented in this file. See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## Unreleased

### Breaking Changes (with deprecation)

The two product variants now have clear canonical names: **withoutBG Open Weights Model** (local ONNX) and **withoutBG API** (cloud).

**New primary API:**
- `WithoutBG.open_weights()` replaces `WithoutBG.opensource()`
- `OpenWeightsModel` replaces `OpenSourceModel`
- `WithoutBGAPIClient` replaces `ProAPI`
- `WithoutBGOpenWeights` replaces `WithoutBGOpenSource`
- CLI `--model open-weights` replaces `--model opensource`

**Deprecated (removed in next major release):**
- `WithoutBG.opensource()` — emits `DeprecationWarning`, delegates to `open_weights()`
- `OpenSourceModel` — alias for `OpenWeightsModel`
- `ProAPI` — alias for `WithoutBGAPIClient`
- `WithoutBGOpenSource` — alias for `WithoutBGOpenWeights`
- CLI `--model opensource` — maps to `open-weights` with a deprecation notice
