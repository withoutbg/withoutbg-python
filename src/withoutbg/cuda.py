"""Make the published ONNX graphs loadable by ``CUDAExecutionProvider``.

The bundle is offline-optimized for CPU, so it contains ``com.microsoft``
``FusedConv`` nodes with a ``Sigmoid`` activation. The CUDA provider only
supports ReLU there and fails at session creation. Splitting those nodes back
into ``Conv`` + ``Sigmoid`` gives identical results (CPU output unchanged,
CUDA within 1/255) and ONNX Runtime re-fuses them on CPU at load time.

Needs the ``onnx`` package (``pip install onnx``). The result is built in memory,
nothing is written to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CUDA_PROVIDER = "CUDAExecutionProvider"
_UNSUPPORTED = frozenset({"Sigmoid"})


def _activation(node: Any) -> str | None:
    for attr in node.attribute:
        if attr.name == "activation":
            value = attr.s
            return value.decode() if isinstance(value, bytes) else value
    return None


def defuse_for_cuda(path: Path) -> bytes | None:
    """Serialized graph with CUDA-unsupported ``FusedConv`` split, or ``None``.

    ``None`` means the graph needs no change; load it from ``path`` as usual.
    """
    try:
        import onnx
        from onnx import helper
    except ImportError as e:  # pragma: no cover - exercised via message only
        raise RuntimeError(
            "CUDAExecutionProvider needs the 'onnx' package to prepare the "
            "model graphs. Install it with: pip install onnx"
        ) from e

    model = onnx.load(str(path))
    nodes, split = [], 0
    for node in model.graph.node:
        act = _activation(node) if node.op_type == "FusedConv" else None
        if act not in _UNSUPPORTED:
            nodes.append(node)
            continue
        if len(node.input) > 3 or any(
            a.name == "activation_params" for a in node.attribute
        ):
            raise RuntimeError(
                f"Cannot prepare {path.name} for CUDA: FusedConv {node.name!r} "
                "has a residual input or activation parameters"
            )
        attrs = {
            a.name: helper.get_attribute_value(a)
            for a in node.attribute
            if a.name != "activation"
        }
        mid = f"{node.output[0]}__preact"
        label = node.name or node.output[0]
        nodes.append(
            helper.make_node("Conv", list(node.input), [mid], f"{label}__conv", **attrs)
        )
        nodes.append(
            helper.make_node(act, [mid], list(node.output), f"{label}__{act.lower()}")
        )
        split += 1
    if split == 0:
        return None
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())
