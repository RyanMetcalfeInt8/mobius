"""OpenVINO profiles must export portable recurrent operator bodies."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx import helper, inliner

from mobius._optimizations import optimize_model
from mobius.functions import causal_conv_nd_with_state, linear_attention


@pytest.mark.parametrize("ep", ["openvino", "openvino-sdpa", "openvino-npu"])
@pytest.mark.parametrize("kind", ["conv", "attention"])
def test_recurrent_function_export_preserves_outputs(ep, kind):
    if kind == "conv":
        function = causal_conv_nd_with_state(kernel_size=3, channels=2)
        shapes = [[1, 2, 3], [2, 1, 3], [2], [1, 2, 2]]
        outputs = [[1, 2, 3], [1, 2, 2]]
    else:
        function = linear_attention(q_num_heads=1, kv_num_heads=1)
        shapes = [[1, 3, 2]] * 3 + [[1, 1, 2, 2], [1, 3, 2], [1, 3, 1]]
        outputs = [[1, 3, 2], [1, 1, 2, 2]]
    inputs = [helper.make_tensor_value_info(f"x{i}", 1, s) for i, s in enumerate(shapes)]
    results = [helper.make_tensor_value_info(f"y{i}", 1, s) for i, s in enumerate(outputs)]
    graph = helper.make_graph(
        [
            helper.make_node(
                function.name,
                [x.name for x in inputs],
                [y.name for y in results],
                domain=function.domain,
            )
        ],
        "recurrent",
        inputs,
        results,
    )
    proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid(function.domain, 1)],
        ir_version=11,
    )
    model = ir.from_proto(proto)
    model.functions[function.identifier()] = function
    reference = inliner.inline_local_functions(ir.to_proto(model))
    optimize_model(model, ep=ep)
    exported = ir.to_proto(model)
    assert not any(
        n.op_type in ("CausalConvWithState", "LinearAttention") for n in exported.graph.node
    )
    rng = np.random.default_rng(0)
    feeds = {x.name: rng.normal(0, 0.1, s).astype(np.float32) for x, s in zip(inputs, shapes)}
    expected = ort.InferenceSession(
        reference.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    actual = ort.InferenceSession(
        exported.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
