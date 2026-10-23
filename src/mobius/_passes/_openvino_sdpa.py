# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared additive masks for the standard full-length dynamic-cache ABI.

The caller must establish that all layers have the same past sequence length
and attention_mask covers past plus current tokens. This is an export profile,
not a rewrite for arbitrary ONNX Attention-24 graphs with shortened masks.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_ir as ir

logger = logging.getLogger(__name__)

# These revisions add FLOAT8E8M0 support. Do not lower graphs using that type,
# or any other type outside the explicitly supported set below.
_TYPE_ONLY_REVISIONS = {
    "Cast",
    "Constant",
    "DequantizeLinear",
    "Reshape",
    "Shape",
    "Transpose",
    "Unsqueeze",
}
_SUPPORTED_TYPES = {
    ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.DOUBLE,
    ir.DataType.BOOL, ir.DataType.INT8, ir.DataType.UINT8, ir.DataType.INT16,
    ir.DataType.UINT16, ir.DataType.INT32, ir.DataType.UINT32, ir.DataType.INT64,
    ir.DataType.UINT64, ir.DataType.INT4, ir.DataType.UINT4,
}
_CONTRACT_KEY = "mobius.openvino_sdpa.mask_contract"


def _can_lower(model: ir.Model) -> bool:
    """Preflight the entire graph before making any edits or changing imports."""
    for value in (*model.graph.inputs, *model.graph.initializers.values()):
        if value.dtype is not None and value.dtype not in _SUPPORTED_TYPES:
            return False
    for node in model.graph:
        if node.domain not in ("", "ai.onnx"):
            return False
        for value in node.outputs:
            if value.dtype is not None and value.dtype not in _SUPPORTED_TYPES:
                return False
        for attr in node.attributes.values():
            if attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS):
                return False
            if (
                attr.type == ir.AttributeType.TENSOR
                and attr.value.dtype not in _SUPPORTED_TYPES
            ):
                return False
        dtype_attribute = {"Cast": "to", "DequantizeLinear": "output_dtype"}.get(node.op_type)
        if dtype_attribute in node.attributes:
            dtype = node.attributes[dtype_attribute].as_int()
            if dtype != 0 and ir.DataType(dtype) not in _SUPPORTED_TYPES:
                return False
        if node.op_type == "Swish":
            if node.attributes.get("alpha", ir.AttrFloat32("alpha", 1.0)).as_float() != 1.0:
                return False
            continue
        try:
            old = onnx.defs.get_schema(node.op_type, 23, node.domain)
            current = onnx.defs.get_schema(node.op_type, 24, node.domain)
        except onnx.defs.SchemaError:
            return False
        if (
            old.since_version != current.since_version
            and node.op_type not in _TYPE_ONLY_REVISIONS | {"Attention"}
        ):
            return False
        if any(name not in old.attributes for name in node.attributes):
            return False
    return True


def apply_openvino_shared_sdpa(model: ir.Model, *, num_layers: int) -> bool:
    """Apply after build/optimization, only with the standard decoder ABI.

    Attention-23 removes runtime mask padding, while the shared additive mask
    removes repeated causal shape calculations. Other ops are retained at their
    existing semantics; alpha=1 Swish is represented as x * Sigmoid(x).
    Returns False without edits if the graph does not meet the profile.
    """
    graph = model.graph
    if graph.opset_imports.get("") != 24 or num_layers <= 0:
        return False
    inputs = {v.name: v for v in graph.inputs}
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        return False
    for name in ("input_ids", "attention_mask"):
        value = inputs[name]
        if value.dtype != ir.DataType.INT64 or value.shape is None or len(value.shape) != 2:
            return False
    attentions = [n for n in graph if n.op_type == "Attention" and n.domain == ""]
    if len(attentions) != num_layers or not _can_lower(model):
        return False
    padding = attentions[0].inputs[3] if len(attentions[0].inputs) > 3 else None
    if padding is None or padding.dtype != ir.DataType.BOOL:
        return False
    producer = padding.producer()
    if producer is None or producer.op_type != "Unsqueeze":
        return False
    # Recognize create_padding_mask(), rather than changing arbitrary masks.
    current = producer.inputs[0]
    for kind in ("Expand", "Unsqueeze", "Cast"):
        source = current.producer()
        if source is None or source.op_type != kind:
            return False
        current = source.inputs[0]
    if current is not inputs["attention_mask"]:
        return False
    past_dimension = None
    for index, node in enumerate(attentions):
        causal = node.attributes.get("is_causal", ir.AttrInt64("is_causal", 0)).as_int()
        softcap = node.attributes.get("softcap", ir.AttrFloat32("softcap", 0.0)).as_float()
        if (
            len(node.inputs) != 6
            or node.inputs[3] is not padding
            or causal != 1
            or softcap != 0.0
            or len(node.outputs) != 3
        ):
            return False
        if any(value is None or value.dtype != ir.DataType.FLOAT for value in node.inputs[:3]):
            return False
        for slot, suffix in ((4, "key"), (5, "value")):
            cache = inputs.get(f"past_key_values.{index}.{suffix}")
            if (
                cache is None
                or node.inputs[slot] is not cache
                or cache.dtype != ir.DataType.FLOAT
                or cache.shape is None
                or len(cache.shape) != 4
            ):
                return False
            if past_dimension is None:
                past_dimension = cache.shape[2]
            elif cache.shape[2] != past_dimension:
                return False

    nodes = []
    prefix = "mobius_shared_sdpa/"
    names = {v.name for n in graph for v in n.outputs} | set(inputs) | set(graph.initializers)
    while any(name and name.startswith(prefix) for name in names):
        prefix = "_" + prefix

    def emit(kind, values, name, **attrs):
        out = ir.Value(name=prefix + name)
        node = ir.node(kind, values, attributes=attrs, outputs=[out], name=prefix + name)
        nodes.append(node)
        return out

    def constant(name, value, dtype=np.int64):
        return emit("Constant", [], name, value=ir.tensor(np.asarray(value, dtype=dtype)))

    zero = constant("zero", 0)
    one = constant("one", 1)
    axis = constant("sequence_axis", 1)
    q_shape = emit("Shape", [inputs["input_ids"]], "query_shape")
    k_shape = emit("Shape", [inputs["attention_mask"]], "mask_shape")
    q_len = emit("Gather", [q_shape, axis], "query_length", axis=0)
    total = emit("Gather", [k_shape, axis], "total_length", axis=0)
    past = emit("Sub", [total, q_len], "past_length")
    rows = emit("Range", [zero, q_len, one], "rows")
    rows = emit("Add", [rows, past], "offset_rows")
    cols = emit("Range", [zero, total, one], "columns")
    rows = emit("Reshape", [rows, constant("row_shape", [1, 1, -1, 1])], "rows4d")
    cols = emit("Reshape", [cols, constant("column_shape", [1, 1, 1, -1])], "columns4d")
    allowed = emit("LessOrEqual", [cols, rows], "causal_allowed")
    allowed = emit("And", [padding, allowed], "allowed")
    yes = constant("allowed_value", 0.0, np.float32)
    no = constant("blocked_value", -np.inf, np.float32)
    mask = emit("Where", [allowed, yes, no], "additive_mask")
    mask.type = ir.TensorType(ir.DataType.FLOAT)
    graph.insert_before(attentions[0], nodes)
    for node in attentions:
        node.replace_input_with(3, mask)
        node.attributes["is_causal"] = ir.AttrInt64("is_causal", 0)
    for node in list(graph):
        if node.op_type != "Swish":
            continue
        sigmoid = ir.Node("", "Sigmoid", [node.inputs[0]], num_outputs=1)
        multiply = ir.Node("", "Mul", [node.inputs[0], sigmoid.outputs[0]], num_outputs=1)
        graph.insert_before(node, [sigmoid, multiply])
        multiply.outputs[0].name = node.outputs[0].name
        multiply.outputs[0].type = node.outputs[0].type
        multiply.outputs[0].shape = node.outputs[0].shape
        node.outputs[0].replace_all_uses_with(multiply.outputs[0], replace_graph_outputs=True)
        graph.remove(node, safe=True)
    graph.opset_imports[""] = 23
    model.metadata_props[_CONTRACT_KEY] = "full_length_mask_equal_layer_cache_lengths"
    logger.info(
        "OpenVINO SDPA: shared additive mask and Attention-23 for %d full-attention layers",
        num_layers,
    )
    return True
