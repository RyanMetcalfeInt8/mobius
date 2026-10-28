"""Computed OpenVINO RoPE matches cached embeddings and remains portable."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._build_context import build_context
from mobius._execution_providers import get_ep
from mobius._optimizations import optimize_model
from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.components._rotary_embedding import (
    ChunkedMRope,
    DefaultRope,
    DynamicNTKRope,
    InterleavedMRope,
    LinearRope,
    Llama3Rope,
    LongRope,
    ProportionalRope,
    YarnRope,
)


def build_rope(rope_class, config, ep, rank=2):
    builder, op, graph = create_test_builder()
    positions = create_test_input(builder, "position_ids", [None] * rank, ir.DataType.INT64)
    with build_context(get_ep(ep), config.dtype):
        outputs = rope_class(config)(op, positions)
    for i, value in enumerate(outputs):
        value.name = f"out{i}"
    graph.outputs.extend(outputs)
    model = ir.Model(graph, ir_version=11)
    optimize_model(model, ep=ep, dtype=config.dtype, model_role="embedding")
    return ir.to_proto(model)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize(
    "rope_class,scaling",
    [
        (DefaultRope, {}),
        (ProportionalRope, {}),
        (LinearRope, {"factor": 2.0}),
        (DynamicNTKRope, {"factor": 2.0}),
        (Llama3Rope, {"factor": 4.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
        (
            LongRope,
            {"short_factor": [1.0, 1.2, 1.4, 1.6], "long_factor": [2.0, 3.0, 4.0, 5.0]},
        ),
        (YarnRope, {"factor": 2.0}),
    ],
)
def test_computed_rope_matches_cached(rope_class, scaling, dtype):
    config = make_config(
        head_dim=16,
        partial_rotary_factor=0.5,
        max_position_embeddings=256,
        original_max_position_embeddings=32,
        rope_scaling=scaling,
        dtype=dtype,
    )
    cached = build_rope(rope_class, config, "default")
    dynamic = build_rope(rope_class, config, "openvino-sdpa")
    assert any(n.op_type == "Cos" for n in dynamic.graph.node)
    assert not any(
        "cos_cache" in t.name or "sin_cache" in t.name for t in dynamic.graph.initializer
    )
    assert not any(n.op_type == "Cos" for n in cached.graph.node)
    sessions = [
        ort.InferenceSession(p.SerializeToString(), providers=["CPUExecutionProvider"])
        for p in (cached, dynamic)
    ]
    # Switch to long frequencies on a mixed batch, then return to short frequencies.
    for positions in (
        np.array([[0, 1, 31], [4, 7, 30]]),
        np.array([[31, 32, 33], [2, 4, 6]]),
        np.array([[0, 1, 2]]),
    ):
        want, got = [
            s.run(None, {"position_ids": positions.astype(np.int64)}) for s in sessions
        ]
        for actual, expected in zip(got, want):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2e-3 if dtype == ir.DataType.FLOAT16 else 2e-6,
                atol=2e-3 if dtype == ir.DataType.FLOAT16 else 2e-6,
            )


@pytest.mark.parametrize("ep", ["openvino", "openvino-sdpa", "openvino-npu"])
def test_openvino_profiles_select_computed_rope(ep):
    proto = build_rope(DefaultRope, make_config(head_dim=16), ep)
    assert any(n.op_type == "Cos" for n in proto.graph.node)
    assert not any(
        "cos_cache" in t.name or "sin_cache" in t.name for t in proto.graph.initializer
    )


@pytest.mark.parametrize("rope_class", [ChunkedMRope, InterleavedMRope])
def test_computed_mrope_matches_cached(rope_class):
    config = make_config(head_dim=16, max_position_embeddings=64, mrope_section=[2, 2, 4])
    protos = [
        build_rope(rope_class, config, ep, rank=3) for ep in ("default", "openvino-sdpa")
    ]
    positions = np.array([[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]], np.int64)
    want, got = [
        ort.InferenceSession(p.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"position_ids": positions}
        )
        for p in protos
    ]
    for actual, expected in zip(got, want):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_longrope_4096_boundary_and_long_context():
    config = make_config(
        head_dim=8,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_scaling={
            "short_factor": [1.0, 1.2, 1.4, 1.6],
            "long_factor": [2.0, 3.0, 4.0, 5.0],
        },
    )
    protos = [build_rope(LongRope, config, ep) for ep in ("default", "openvino-sdpa")]
    sessions = [
        ort.InferenceSession(p.SerializeToString(), providers=["CPUExecutionProvider"])
        for p in protos
    ]
    for positions in ([[4094, 4095]], [[4095, 4096]], [[4096, 131071]], [[0, 4095]]):
        want, got = [
            s.run(None, {"position_ids": np.array(positions, np.int64)}) for s in sessions
        ]
        for actual, expected in zip(got, want):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_bfloat16_cast_is_after_float32_trigonometry():
    proto = build_rope(
        DefaultRope, make_config(head_dim=16, dtype=ir.DataType.BFLOAT16), "openvino-sdpa"
    )
    assert all(
        o.type.tensor_type.elem_type == int(ir.DataType.BFLOAT16) for o in proto.graph.output
    )
    producers = {name: node for node in proto.graph.node for name in node.output}
    for output in proto.graph.output:
        assert producers[output.name].op_type == "Cast"
    assert any(n.op_type == "Cos" for n in proto.graph.node)
