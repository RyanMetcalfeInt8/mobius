"""Window-index export retains vector rank and the reference ordering."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius.components._qwen25_vl_vision import _window_index_one_image
from mobius.components._scan_utils import create_body_graph


@pytest.mark.parametrize("thw", [(1, 2, 2), (1, 4, 4), (2, 6, 10)])
def test_window_index_explicit_axis_preserves_order(thw):
    inputs = [
        ir.Value(name=n, shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT64))
        for n in ("t", "h", "w")
    ]
    graph, builder = create_body_graph([], inputs, name="window_index")
    outputs = _window_index_one_image(builder.op, *inputs, ms=2, ws=2, smu=4)
    for i, output in enumerate(outputs):
        output.name = f"out{i}"
    graph.outputs.extend(outputs)
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    compress = next(n for n in proto.graph.node if n.op_type == "Compress")
    assert next(a.i for a in compress.attribute if a.name == "axis") == 0
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None, {n: np.array(v, np.int64) for n, v in zip(("t", "h", "w"), thw)}
    )

    t, h, w = thw
    h, w = h // 2, w // 2
    indices = np.arange(t * h * w).reshape(t, h, w)
    padded = np.pad(indices, ((0, 0), (0, 2 - h % 2), (0, 2 - w % 2)), constant_values=-100)
    windows = (
        padded.reshape(t, padded.shape[1] // 2, 2, padded.shape[2] // 2, 2)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1, 4)
    )
    valid = windows != -100
    expected = [
        windows[valid],
        np.cumsum(valid.sum(axis=1) * 4),
        np.array(t * h * w),
        np.array(len(windows)),
    ]
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
