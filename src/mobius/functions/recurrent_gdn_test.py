# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable native GDN contract and nonzero-state recurrence checks."""

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx.inliner import inline_local_functions

from mobius.functions.linear_attention import recurrent_gdn


def test_openvino_optimizer_preserves_native_boundary():
    from mobius._optimizations import optimize_model

    function = recurrent_gdn()
    model = ir.Model(function.graph.clone(), ir_version=11)
    # The function body is portable; its named call must remain native for OV.
    node = ir.Node(function.domain, function.name, model.graph.inputs, num_outputs=2)
    model.graph.extend([node])
    model.graph.outputs.clear()
    model.graph.outputs.extend(node.outputs)
    model.functions[function.identifier()] = function
    model.graph.opset_imports[function.domain] = 1
    optimize_model(model, ep="openvino-sdpa")
    assert [(n.domain, n.op_type) for n in model.graph] == [(function.domain, function.name)]
    assert function.identifier() in model.functions
    assert model.graph.opset_imports[function.domain] == 1


@pytest.mark.parametrize("length", [1, 7])
def test_portable_recurrence_with_nonzero_state(length):
    function = recurrent_gdn()
    shapes = [
        [1, 2, length, 3],
        [1, 2, length, 3],
        [1, 2, length, 5],
        [1, 2, length],
        [1, 2, length],
        [1, 2, 3, 5],
    ]
    inputs = [
        onnx.helper.make_tensor_value_info(v.name, onnx.TensorProto.FLOAT, shape)
        for v, shape in zip(function.graph.inputs, shapes)
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("final", onnx.TensorProto.FLOAT, shapes[-1]),
        onnx.helper.make_tensor_value_info("sequence", onnx.TensorProto.FLOAT, shapes[2]),
    ]
    node = onnx.helper.make_node(
        function.name,
        [v.name for v in inputs],
        [v.name for v in outputs],
        domain=function.domain,
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph([node], "cell", inputs, outputs),
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid(function.domain, 1),
        ],
        functions=[ir.serde.serialize_function(function)],
        ir_version=11,
    )
    onnx.checker.check_model(model)
    # ORT must expand the registered local function without a native kernel.
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    fallback = inline_local_functions(model)
    assert all(n.domain in ("", "ai.onnx") for n in fallback.graph.node)
    rng = np.random.default_rng(42)
    feeds = {
        v.name: rng.normal(0, 0.1, shape).astype(np.float32)
        for v, shape in zip(inputs, shapes)
    }
    feeds["g"].fill(-0.2)
    feeds["beta"].fill(0.5)
    state = feeds["state"].copy()
    expected = np.empty(shapes[2], dtype=np.float32)
    for t in range(length):
        state *= np.exp(feeds["g"][:, :, t, None, None])
        key = feeds["k"][:, :, t, :, None]
        retrieved = (state * key).sum(axis=-2)
        delta = (feeds["v"][:, :, t] - retrieved) * feeds["beta"][:, :, t, None]
        state += key * delta[:, :, None, :]
        expected[:, :, t] = (state * feeds["q"][:, :, t, :, None]).sum(axis=-2)
    final, sequence = session.run(None, feeds)
    np.testing.assert_allclose(final, state, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(sequence, expected, atol=2e-7, rtol=2e-6)
