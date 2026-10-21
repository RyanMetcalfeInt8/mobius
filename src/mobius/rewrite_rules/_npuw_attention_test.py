# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from mobius.rewrite_rules._decompose_attention import NpuwAttentionPass
from mobius.rewrite_rules._decompose_attention_test import _build_attention_model, _feeds


@pytest.mark.parametrize('seq,past', [(4, 0), (1, 5), (3, 5)])
@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'float', 'fully_masked', 'short'])
def test_npuw_attention_parity(tmp_path, seq, past, mask_kind):
    ov = pytest.importorskip('openvino')
    model = _build_attention_model(2, seq, past, 4, 2, 8, with_mask=mask_kind != 'none')
    for output, shape in zip(model.graph.outputs, ([2, seq, 32], [2, 2, past + seq, 8], [2, 2, past + seq, 8])):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape(shape)
    feeds = _feeds(2, seq, past, 4, 2, 8, with_mask=mask_kind != 'none')
    if mask_kind in ('bool', 'fully_masked'):
        mask = model.graph.inputs[3]
        mask.type = ir.TensorType(ir.DataType.BOOL)
        feeds['mask'] = np.ones((2, 1, seq, past + seq), dtype=bool)
        feeds['mask'][0, :, :, 0] = False
        if mask_kind == 'fully_masked':
            feeds['mask'][0] = False
    if mask_kind == 'short':
        model.graph.inputs[3].shape = ir.Shape([2, 1, seq, past + seq - 1])
        feeds['mask'] = feeds['mask'][..., :-1]
    source = tmp_path / 'source.onnx'
    target = tmp_path / 'target.onnx'
    ir.save(model, source)
    NpuwAttentionPass()(model)
    ir.save(model, target)
    onnx.checker.check_model(onnx.load(target))
    core = ov.Core()
    config = {'INFERENCE_PRECISION_HINT': 'f32'}
    a = core.compile_model(core.read_model(source), 'CPU', config)(feeds)
    converted = core.read_model(target)
    b = core.compile_model(converted, 'CPU', config)(feeds)
    for i in range(3):
        np.testing.assert_allclose(a[i], b[i], atol=2e-6, rtol=2e-5)
    softmax = next(n for n in converted.get_ops() if n.get_type_name() == 'Softmax')
    add = softmax.input_value(0).get_node()
    assert add.get_type_name() == 'Add'
    assert add.input_value(0).get_node().get_type_name() == 'MatMul'
    final = converted.output(0).get_node().input_value(0).get_node()
    assert final.get_type_name() == 'Select'
    assert final.input_value(2).get_node().get_type_name() == 'Reshape'


def test_npuw_rejects_softcap_before_mutation():
    model = _build_attention_model(1, 2, 0, 4, 2, 8, softcap=1.0)
    with pytest.raises(ValueError, match='openvino-npu requires'):
        NpuwAttentionPass()(model)
    assert any(n.op_type == 'Attention' for n in model.graph)
