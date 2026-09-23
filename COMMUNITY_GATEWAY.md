# Community gateway

The next community bundle reuses `router_10.1.0_best.pth` and its frozen
DINOv3 ConvNeXt-Base backbone. withoutBG trains and maintains the matting
branch; BiRefNet supplies segmentation. There is no random or heuristic fallback.

| Router category | Branch |
| --- | --- |
| fine_strand, soft_detail, transparency | withoutBG matting |
| hard_opaque, flat_scene, vehicle | BiRefNet |

## Runtime contract

`withoutbg-open-weights.onnx.json` contains a `gateway` object with
`schema_version: 1`, checkpoint-ordered `categories`, and `router`, `matting`,
`birefnet` specifications. Each specification has `file`, `sha256`, `canvas_size`,
`resize`, `interpolation`, `input_name`, and `output_name`.

All three graphs accept NCHW float32 RGB in [0, 1], with normalization inside
the exported graph. The router takes a 448-square bicubic stretch and returns
six logits. BiRefNet takes a 1024-square bilinear stretch and returns sigmoid
alpha. Matting retains its released letterbox preprocessing. Branch output
dimensions may differ from input dimensions; crop coordinates are scaled before
restoring the original image size. Only the selected branch executes.

The SDK loads graphs lazily; `preload()` loads and verifies all three. Docker
and the HF Space load all three before declaring readiness. Every graph is
verified against its manifest checksum. Corrupt or incomplete gateway bundles
fail rather than silently switching to matting. Old sidecars without `gateway`
retain the existing single-model behavior.

The Python implementation in `src/withoutbg/gateway.py` is vendored unchanged
into Docker and the HF Space. Run `scripts/sync_community_gateway.py --check`
from this repository to check the sibling copies, or omit `--check` to update them.
The Swift implementation uses the same manifest and category mapping; Core ML
package checksums hash sorted relative file paths followed by their contents.

## Local use

Point the existing SDK interface at the generated bundle:

```python
from withoutbg import WithoutBG

model = WithoutBG.open_weights(model_path="/path/to/community-onnx/withoutbg-open-weights.onnx")
result = model.remove_background("input.jpg")
```

For Docker, mount the whole bundle directory and set `WITHOUTBG_MODEL_PATH`
to the matting ONNX file inside it. GIMP keeps using `/v1/remove-background`
on its local Docker or Mac server. No plugin protocol change is needed.

## Rebuild and release

`model-pro/scripts/export_community_gateway.py` exports the trained router and
BiRefNet, numerically checks both against PyTorch, copies the existing community
matting artifact, and writes the manifest plus enriched sidecar. Example from
`model-pro`:

```sh
uv run python scripts/export_community_gateway.py \
  --router-checkpoint checkpoints/router_10.1.0_best.pth \
  --convnext-checkpoint checkpoints/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth \
  --birefnet-checkpoint ../wbgapi/models/checkpoints/BiRefNet \
  --matting compiled_models/wbgnet_oss.onnx \
  --output compiled_models/community-onnx
```

For Mac, use `--format coreml`, supply the existing `wbgnet_oss.mlpackage`, and
choose a separate output directory. Core ML validation requires macOS. Copy the
router, BiRefNet, manifest, and sidecar into `WithoutBGCore` resources. The existing
matting package must match the manifest checksum. Python and CoreGraphics image
resampling can differ slightly; tensor-level export parity does not establish
pixel-identical output across platforms.

BiRefNet's deformable convolution is represented with grid sampling and
convolutions during export; its weights are retained. The exporter checks this
operation against torchvision before exporting the full segmentation graph.

Release the ONNX assets and sidecar together on Hugging Face, retain all upstream
notices, then release the updated SDK, Space, Docker images, and Mac app. The SDK
pins downloads within a session to one Hub snapshot. Docker's downloader includes
all gateway assets and verifies hashes. The fp32 ONNX bundle is approximately
1.6 GiB, larger than the previous single model. CUDA execution still needs a
separate hardware validation pass before GPU release.

The current website labels the gateway as the next community release. Change
that copy only when the distribution releases are available. Existing published
comparison images and quality claims have not been recomputed for this pipeline.
