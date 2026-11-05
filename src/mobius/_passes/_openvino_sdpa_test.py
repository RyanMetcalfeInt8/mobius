# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export-profile contracts and shared-mask numerical checks (no weights/network)."""

import unittest

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from mobius._builder import _maybe_apply_openvino_shared_sdpa, build_from_module
from mobius._passes._openvino_sdpa import apply_openvino_shared_sdpa
from mobius._testing import make_config
from mobius.models.base import CausalLMModel
from mobius.tasks._causal_lm import CausalLMTask


def build(ep="default", **overrides):
    config = make_config(**overrides)
    module = CausalLMModel(config)
    package = build_from_module(module, config, execution_provider=ep)
    return module, package


class OpenVINOSharedSDPATest(unittest.TestCase):
    def test_profile_is_selected_by_real_builder(self):
        _, package = build("openvino-sdpa")
        model = package["model"]
        self.assertEqual(model.graph.opset_imports[""], 23)
        attentions = [n for n in model.graph if n.op_type == "Attention"]
        self.assertEqual(len(attentions), 2)
        self.assertIs(attentions[0].inputs[3], attentions[1].inputs[3])
        self.assertEqual(attentions[0].inputs[3].dtype, ir.DataType.FLOAT)
        self.assertTrue(all(n.attributes["is_causal"].as_int() == 0 for n in attentions))
        self.assertEqual(sum(n.op_type == "Range" for n in model.graph), 2)
        self.assertFalse(any(n.op_type == "Swish" for n in model.graph))
        self.assertFalse(apply_openvino_shared_sdpa(model, num_layers=2))

    def test_other_profiles_and_attention_schedules_stay_at_24(self):
        for ep, overrides in [
            ("default", {}), ("openvino", {}),
            ("openvino-sdpa", {"dtype": ir.DataType.FLOAT16}),
            ("openvino-sdpa", {"sliding_window": 8}),
            ("openvino-sdpa", {"layer_types": ["full_attention", "sliding_attention"]}),
        ]:
            with self.subTest(ep=ep, overrides=overrides):
                _, package = build(ep, **overrides)
                self.assertEqual(package["model"].graph.opset_imports[""], 24)

    def test_static_paged_and_custom_tasks_are_not_selected(self):
        class CustomTask(CausalLMTask):
            pass
        tasks = [CausalLMTask(static_cache=True), CausalLMTask(paged_cache=True), CustomTask()]
        for task in tasks:
            module, package = build()
            _maybe_apply_openvino_shared_sdpa(package, module, task, "openvino-sdpa")
            self.assertEqual(package["model"].graph.opset_imports[""], 24)

    def test_preflight_rejection_does_not_edit_graph(self):
        for kind in [
            "nonpad", "custom", "new_dtype", "softcap", "different_mask",
            "swish_alpha", "different_cache_length", "new_output_dtype",
        ]:
            with self.subTest(kind=kind):
                _, package = build()
                model = package["model"]
                attention = next(n for n in model.graph if n.op_type == "Attention")
                if kind == "nonpad":
                    attention.resize_inputs(7)
                    attention.replace_input_with(6, model.graph.inputs[0])
                elif kind == "custom":
                    model.graph.append(ir.node(
                        "Unrecognized", [attention.outputs[0]], domain="custom", num_outputs=1,
                    ))
                elif kind == "new_dtype":
                    model.graph.inputs[0].type = ir.TensorType(ir.DataType.FLOAT8E8M0)
                elif kind == "softcap":
                    attention.attributes["softcap"] = ir.AttrFloat32("softcap", 30.0)
                elif kind == "different_mask":
                    attention.replace_input_with(3, model.graph.inputs[0])
                elif kind == "different_cache_length":
                    cache = next(
                        v for v in model.graph.inputs if v.name == "past_key_values.1.key"
                    )
                    cache.shape = ir.Shape(["batch", 2, "different_past_length", 16])
                elif kind == "new_output_dtype":
                    extra = ir.node(
                        "DequantizeLinear", [attention.inputs[4], attention.inputs[4]],
                        attributes={"output_dtype": int(ir.DataType.FLOAT8E8M0)},
                        num_outputs=1,
                    )
                    model.graph.append(extra)
                else:
                    swish = next(n for n in model.graph if n.op_type == "Swish")
                    swish.attributes["alpha"] = ir.AttrFloat32("alpha", 2.0)
                before = ir.serde.serialize_model(model).SerializeToString()
                self.assertFalse(apply_openvino_shared_sdpa(model, num_layers=2))
                self.assertEqual(ir.serde.serialize_model(model).SerializeToString(), before)

    def test_shared_mask_matches_bottom_right_oracle(self):
        _, package = build("openvino-sdpa")
        model = package["model"]
        mask = next(n.inputs[3] for n in model.graph if n.op_type == "Attention")
        # Evaluate only the shared mask, independent of decoder weights.
        pending = [mask.producer()]
        required = set()
        while pending:
            node = pending.pop()
            if node in required:
                continue
            required.add(node)
            pending.extend(
                v.producer() for v in node.inputs
                if v is not None and v.producer() is not None
            )
        used = {v for n in required for v in n.inputs if v is not None}
        initializers = []
        for value in model.graph.initializers.values():
            if value in used:
                tensor = ir.serde.serialize_tensor(value.const_value)
                tensor.name = value.name
                initializers.append(tensor)
        graph = onnx.helper.make_graph(
            [ir.serde.serialize_node(n) for n in model.graph if n in required],
            "shared_mask", [
                onnx.helper.make_tensor_value_info(
                    "input_ids", onnx.TensorProto.INT64, ["B", "Q"],
                ),
                onnx.helper.make_tensor_value_info(
                    "attention_mask", onnx.TensorProto.INT64, ["B", "T"],
                ),
            ], [onnx.helper.make_tensor_value_info(
                mask.name, onnx.TensorProto.FLOAT, ["B", 1, "Q", "T"],
            )],
            initializer=initializers,
        )
        proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 23)], ir_version=10,
        )
        onnx.checker.check_model(proto)
        evaluator = ReferenceEvaluator(proto)
        for batch, query, past, kind in [(1,3,0,"full"), (1,1,5,"full"),
                                          (2,2,3,"padding"), (2,2,3,"empty")]:
            with self.subTest(batch=batch, query=query, past=past, kind=kind):
                total = query + past
                padding = np.ones((batch, total), np.int64)
                if kind == "padding":
                    padding[0, :2] = 0
                    padding[1, -2:] = 0
                elif kind == "empty":
                    padding[:] = 0
                actual = evaluator.run(None, {"input_ids": np.zeros((batch, query), np.int64),
                                              "attention_mask": padding})[0]
                causal = np.arange(total)[None, :] <= (past + np.arange(query))[:, None]
                allowed = padding[:, None, None, :].astype(bool) & causal[None, None, :, :]
                np.testing.assert_array_equal(
                    actual, np.where(allowed, np.float32(0), np.float32(-np.inf)),
                )

    def test_custom_model_is_not_selected(self):
        class CustomModel(CausalLMModel):
            pass

        config = make_config()
        package = build_from_module(
            CustomModel(config), config, execution_provider="openvino-sdpa",
        )
        self.assertEqual(package["model"].graph.opset_imports[""], 24)


if __name__ == "__main__":
    unittest.main()
