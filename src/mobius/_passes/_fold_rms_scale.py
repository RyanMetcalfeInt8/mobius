# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Materialize small, loaded constant scale expressions for ONNX RMS norms."""

import numpy as np
import onnx_ir as ir


class FoldRmsScalePass(ir.passes.InPlacePass):
    """Fold Add/Cast scale branches without touching projection weights.

    Evaluate each operation in its original dtype (in particular, Gemma's
    FP32 offset addition). Constant scales expose RMS fusion/precision patterns
    to backends. Runtime scales and unsupported expressions remain unchanged.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        folded = {}
        modified = False

        def evaluate(value, depth=0):
            if value is None or depth > 16 or value.is_graph_input():
                return None
            if value.const_value is not None:
                tensor = value.const_value
                if tensor.size > 8192 or tensor.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
                    return None
                return tensor.numpy()
            node = value.producer()
            if node is None or node.domain not in ('', 'ai.onnx'):
                return None
            if node.op_type not in ('Add', 'Cast'):
                return None
            arrays = [evaluate(v, depth + 1) for v in node.inputs]
            if any(a is None for a in arrays):
                return None
            if node.op_type == 'Cast' and len(arrays) == 1:
                to = node.attributes.get('to')
                types = {int(ir.DataType.FLOAT): np.float32, int(ir.DataType.FLOAT16): np.float16}
                if to is None or to.value not in types:
                    return None
                result = arrays[0].astype(types[to.value])
            elif node.op_type == 'Add' and len(arrays) == 2:
                if arrays[0].dtype != arrays[1].dtype:
                    return None
                try:
                    shape = np.broadcast_shapes(arrays[0].shape, arrays[1].shape)
                except ValueError:
                    return None
                if np.prod(shape) > 8192:
                    return None
                result = arrays[0] + arrays[1]
            else:
                return None
            return result if np.isfinite(result).all() else None

        for node in model.graph.all_nodes():
            if node.domain not in ('', 'ai.onnx') or node.op_type != 'RMSNormalization' or len(node.inputs) < 2:
                continue
            scale = node.inputs[1]
            if scale is None or scale.const_value is not None:
                continue
            if scale not in folded:
                array = evaluate(scale)
                if array is None:
                    continue
                tensor = ir.tensor(array)
                replacement = ir.Value(
                    name=f'{scale.name}_folded_rms_scale',
                    shape=ir.Shape(array.shape), type=ir.TensorType(tensor.dtype), const_value=tensor,
                )
                node.graph.register_initializer(replacement)
                folded[scale] = replacement
            node.replace_input_with(1, folded[scale])
            modified = True
        return ir.passes.PassResult(model, modified=modified)
