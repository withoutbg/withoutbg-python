# ruff: noqa: E402
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from withoutbg import routed
from withoutbg.cuda import defuse_for_cuda


def _model(tmp_path, activation):
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "w")
    attrs = {"kernel_shape": [1, 1]}
    if activation:
        node = helper.make_node(
            "FusedConv",
            ["x", "w"],
            ["y"],
            "fc",
            domain="com.microsoft",
            activation=activation,
            **attrs,
        )
    else:
        node = helper.make_node("Conv", ["x", "w"], ["y"], "c", **attrs)
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        [w],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    return path


def test_sigmoid_fused_conv_is_split(tmp_path):
    data = defuse_for_cuda(_model(tmp_path, "Sigmoid"))
    ops = [n.op_type for n in onnx.load_from_string(data).graph.node]
    assert ops == ["Conv", "Sigmoid"]


def test_relu_fused_conv_and_plain_conv_untouched(tmp_path):
    assert defuse_for_cuda(_model(tmp_path, "Relu")) is None
    assert defuse_for_cuda(_model(tmp_path, None)) is None


def test_split_graph_matches_math(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    data = defuse_for_cuda(_model(tmp_path, "Sigmoid"))
    x = np.random.rand(1, 1, 2, 2).astype(np.float32)
    out = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    assert np.allclose(out, 1 / (1 + np.exp(-x)), atol=1e-6)


def test_cuda_session_gets_prepared_bytes(monkeypatch):
    import hashlib

    from withoutbg.routed import RoutedPipeline  # noqa: F401

    calls = []
    monkeypatch.setattr(routed, "defuse_for_cuda", lambda p: b"prepared")
    p = routed.RoutedPipeline.__new__(routed.RoutedPipeline)
    import pathlib
    import tempfile
    import threading

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "g.onnx").write_bytes(b"g")
    p.sidecar = {
        "router": {"file": "g.onnx", "sha256": hashlib.sha256(b"g").hexdigest()}
    }
    p.resolve_file = lambda n: d / n
    p.providers = ["CUDAExecutionProvider"]
    p.sessions = {}
    p.lock = threading.RLock()

    class S:
        def __init__(self, source, providers):
            calls.append(source)

        def get_providers(self):
            return ["CUDAExecutionProvider"]

    p.session_factory = S
    p._session("router")
    assert calls == [b"prepared"]
