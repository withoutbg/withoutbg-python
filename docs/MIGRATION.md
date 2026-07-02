# Migration Guide

## From the withoutbg monorepo

The Python SDK (`pip install withoutbg`) has moved from a sub-package inside the monorepo to its own standalone repository.

| Old location | New location |
|---|---|
| `withoutbg/withoutbg` repo, `packages/python/` sub-directory | This repo (root level) |
| `apps/web/` — Docker + FastAPI + web UI | [withoutbg/withoutbg-inference](https://github.com/withoutbg/withoutbg-inference) |

If you were cloning the monorepo to use the Python package, you can now simply `pip install withoutbg` or clone this repo instead.

If you were running the self-hosted Docker application from `apps/web/`, see [withoutbg-inference](https://github.com/withoutbg/withoutbg-inference).

---

## API renames (v1.0.3)

The two product variants now have clear canonical names. The old names are deprecated aliases that still work but will be removed in the next major release.

### Python API

| Old (deprecated) | New (canonical) | Notes |
|---|---|---|
| `WithoutBG.opensource()` | `WithoutBG.open_weights()` | Emits `DeprecationWarning` |
| `OpenSourceModel` | `OpenWeightsModel` | Alias, no warning yet |
| `ProAPI` | `WithoutBGAPIClient` | Alias, no warning yet |
| `WithoutBGOpenSource` | `WithoutBGOpenWeights` | Alias, no warning yet |

**Before:**

```python
from withoutbg import WithoutBG

model = WithoutBG.opensource()
result = model.remove_background("photo.jpg")
```

**After:**

```python
from withoutbg import WithoutBG

model = WithoutBG.open_weights()
result = model.remove_background("photo.jpg")
```

For the cloud API:

**Before:**

```python
from withoutbg import ProAPI

api = ProAPI(api_key="sk_...")
result = api.remove_background("photo.jpg")
```

**After:**

```python
from withoutbg import WithoutBG

model = WithoutBG.api(api_key="sk_...")
result = model.remove_background("photo.jpg")
```

### CLI

| Old (deprecated) | New |
|---|---|
| `withoutbg photo.jpg --model opensource` | `withoutbg photo.jpg --model open-weights` |

The old flag still works but prints a deprecation notice.

---

## Model changes

The older multi-file model (four separate ONNX files: depth, ISNet, matting, refiner) has been superseded by a single unified ONNX graph. The four-file environment variables (`WITHOUTBG_DEPTH_MODEL_PATH`, `WITHOUTBG_ISNET_MODEL_PATH`, etc.) are no longer used.

Set `WITHOUTBG_MODEL_PATH` to point at the unified `.onnx` file if you want to use a local copy instead of the Hugging Face download.

The sidecar metadata file (`withoutbg-open-weights.onnx.json`) must be in the same directory as the ONNX file.
