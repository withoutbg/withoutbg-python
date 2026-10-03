# Changelog

All notable changes to this project will be documented in this file. See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [1.2.1](https://github.com/withoutbg/withoutbg-python/compare/v1.2.0...v1.2.1) (2026-10-03)

### Bug Fixes

* reject rooted and drive bundle paths on every OS ([78c520f](https://github.com/withoutbg/withoutbg-python/commit/78c520fdc4b2b741349e6588948922213862fc03))

## [1.2.0](https://github.com/withoutbg/withoutbg-python/compare/v1.1.1...v1.2.0) (2026-10-03)

### Features

* route community inference through matting and BiRefNet ([27bf3bc](https://github.com/withoutbg/withoutbg-python/commit/27bf3bc06607494c4bca813d928e516e000d8ca2))
* run the routed open-weights bundle (router → matting or BiRefNet) ([9a1dc92](https://github.com/withoutbg/withoutbg-python/commit/9a1dc9279eb70a526aa520abcf3c24a949e3de37))

### Documentation

* embed AlphaMate video via GitHub CDN ([1c441c7](https://github.com/withoutbg/withoutbg-python/commit/1c441c785951d5375ac3b4dc5812a93236a89765))
* rewrite README for clarity and conviction ([8469a16](https://github.com/withoutbg/withoutbg-python/commit/8469a166de1203ca7a7ea2ccdee84b399cc30fcf))
* swap README hero to resized revealed.webp ([d922fa7](https://github.com/withoutbg/withoutbg-python/commit/d922fa7d834ec07a78f73883f3fa8932f7fd63c2))
* trim uncertain README claims and em dashes ([169979c](https://github.com/withoutbg/withoutbg-python/commit/169979cac64f8f0dc4fe5623bc31305682cc8a72))
* use animated WebP for AlphaMate README reveal ([6b4830d](https://github.com/withoutbg/withoutbg-python/commit/6b4830d531129f940de439af7894f5102ac860d0))

## [1.1.1](https://github.com/withoutbg/withoutbg-python/compare/v1.1.0...v1.1.1) (2026-07-19)

### Bug Fixes

* **ci:** upgrade setuptools before pip-audit ([c081cae](https://github.com/withoutbg/withoutbg-python/commit/c081caed022eca08b1c2e44d31667d15dda292dd))

### Documentation

* lead README with AlphaMate reveal video ([55ecde1](https://github.com/withoutbg/withoutbg-python/commit/55ecde1d5a4d46295e9f324665f553218325396c))

## [1.1.0](https://github.com/withoutbg/withoutbg-python/compare/v1.0.6...v1.1.0) (2026-07-17)

### Features

* **models:** support v10 open-weights (448 canvas) ([ee61482](https://github.com/withoutbg/withoutbg-python/commit/ee61482b3babec0262b77ddc37d24d7ebb5c6bf7))

## Unreleased

### Features

* Support open-weights model **v10.0.0** (unified ONNX; 448×448 canvas)

### Bug Fixes

* Default letterbox canvas to 448 when sidecar metadata is missing

## [1.0.6](https://github.com/withoutbg/withoutbg/compare/v1.0.5...v1.0.6) (2026-07-02)

### Bug Fixes

* **ci:** use Python 3.12 as mypy target for numpy 2.x stubs ([f5cfa68](https://github.com/withoutbg/withoutbg/commit/f5cfa68725fe1b9f210f006ea6bd07c0f139dfe1))

## [1.0.5](https://github.com/withoutbg/withoutbg/compare/v1.0.4...v1.0.5) (2026-07-02)

### Bug Fixes

* **ci:** skip numpy stubs in mypy to fix CI on numpy 2.x ([1b5a42d](https://github.com/withoutbg/withoutbg/commit/1b5a42db2498e0f98ac0f0fa03b50d3f39bdbfa3))

## [1.0.4](https://github.com/withoutbg/withoutbg/compare/v1.0.3...v1.0.4) (2026-07-02)

### Bug Fixes

* **ci:** update mypy target to Python 3.10 ([08941b1](https://github.com/withoutbg/withoutbg/commit/08941b1ac40b222984c3a5349671fa4b476d56a3))

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
