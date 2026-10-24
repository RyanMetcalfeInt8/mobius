# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import numpy as np
import onnx
import onnx_ir as ir
from onnx import helper, numpy_helper
from mobius._optimizations import fold_initializers_after_weights


def model(dynamic=False, size=4, op="Add", shared=False):
    w = np.full(size, 0.0003, dtype=np.float16)
    nodes = [helper.make_node("Cast", ["w"], ["wf"], to=onnx.TensorProto.FLOAT),
             helper.make_node(op, ["wf", "one"], ["scale"]),
             helper.make_node("RMSNormalization", ["x", "scale"], ["y"])]
    outputs = [helper.make_tensor_value_info("y", 1, [1, size])]
    if shared:
        nodes.append(helper.make_node("RMSNormalization", ["x", "scale"], ["z"]))
        outputs.append(helper.make_tensor_value_info("z", 1, [1, size]))
    inputs = [helper.make_tensor_value_info("x", 1, [1, size])]
    initializers = [numpy_helper.from_array(np.array(1, np.float32), "one")]
    if dynamic:
        inputs.append(helper.make_tensor_value_info("w", 10, [size]))
    else:
        initializers.append(numpy_helper.from_array(w, "w"))
    proto = helper.make_model(helper.make_graph(nodes, "test", inputs, outputs, initializers),
                              opset_imports=[helper.make_opsetid("", 23)])
    return ir.serde.deserialize_model(proto), w


class FoldRmsScaleTests(unittest.TestCase):
    def test_fp32_offset_shared_and_idempotent(self):
        m, w = model(shared=True)
        fold_initializers_after_weights(m)
        norms = list(m.graph)
        self.assertEqual([n.op_type for n in norms], ["RMSNormalization"] * 2)
        self.assertIs(norms[0].inputs[1], norms[1].inputs[1])
        np.testing.assert_array_equal(norms[0].inputs[1].const_value.numpy(), w.astype(np.float32) + np.float32(1))
        self.assertEqual(norms[0].inputs[1].dtype, ir.DataType.FLOAT)
        fold_initializers_after_weights(m)
        self.assertEqual(len(list(m.graph)), 2)
        onnx.checker.check_model(ir.serde.serialize_model(m))

    def test_skip_runtime_large_and_unsupported(self):
        for kwargs in ({"dynamic": True}, {"size": 8193}, {"op": "Mul"}):
            with self.subTest(kwargs=kwargs):
                m, _ = model(**kwargs)
                fold_initializers_after_weights(m)
                norm = list(m.graph)[-1]
                self.assertIsNone(norm.inputs[1].const_value)

    def test_skip_unloaded(self):
        m, _ = model()
        m.graph.initializers["w"].const_value = None
        fold_initializers_after_weights(m)
        self.assertIsNone(list(m.graph)[-1].inputs[1].const_value)

if __name__ == "__main__":
    unittest.main()
