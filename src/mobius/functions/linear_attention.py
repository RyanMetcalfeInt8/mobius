# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reference ir.Function for the proposed LinearAttention operator.

Implements the gated delta-rule linear attention recurrence as a
self-contained function: 3D→4D reshape + GQA head expansion + query
scaling + sequential Scan over the time dimension.  The component
(GatedDeltaNet) simply calls
``op.LinearAttention(q, k, v, state, decay, beta)`` — all complexity
lives here.

The gated-delta variant (used by Qwen3.5 GatedDeltaNet) computes:

    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)
    o_t = q_t^T S_t

All activations are 3D ``[B, T, H*D]`` (matching the ONNX Attention op
convention).  ``q_num_heads`` and ``kv_num_heads`` attributes tell
the function how to reshape to 4D internally.

The function has 6 required inputs:
  ``(query, key, value, past_state, decay, beta)``

GQA support: when Q/K have fewer heads (q_num_heads) than
V/state (kv_num_heads), the function expands Q/K heads internally
via Tile+Reshape.

Op spec: https://github.com/onnx/onnx/issues/7689
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION
from mobius.components._scan_utils import (
    create_body_graph,
    rename_subgraph_values,
)

DOMAIN = "com.microsoft"


def linear_attention(
    *,
    q_num_heads: int,
    kv_num_heads: int,
    update_rule: str = "gated_delta",
    scale: float = 1.0,
    stash_type: ir.DataType = ir.DataType.FLOAT,
    openvino_gdn: bool = False,
) -> ir.Function:
    """Build an ir.Function for LinearAttention.

    Inputs (conditional):
        query:      (B, T, q_num_heads * d_k)
        key:        (B, T, q_num_heads * d_k)
        value:      (B, T, kv_num_heads * d_v)
        past_state: (B, kv_num_heads, d_k, d_v) — recurrent state
        decay:      (B, T, kv_num_heads * d_k), or (B, T, kv_num_heads)
                    with scalar_decay — required for "gated" and "gated_delta"
        beta:       (B, T, kv_num_heads) — required when update_rule in ("delta", "gated_delta")

    Outputs:
        output:        (B, T, kv_num_heads * d_v) — attention output (3D)
        present_state: (B, kv_num_heads, d_k, d_v) — updated state

    Args:
        q_num_heads: Number of heads for Q/K (may be fewer than V
            for GQA).  Matches ``kv_num_heads`` in the standard
            Attention op (Q/K are the "key-query" pair here).
        kv_num_heads: Number of heads for V and the output.  Matches
            ``q_num_heads`` in the standard Attention op (V determines
            the output head count).

    .. note:: **Naming convention vs standard Attention**

       In standard GQA (``Attention`` op), Q has *more* heads than K/V,
       so ``q_num_heads >= kv_num_heads``.  In LinearAttention the roles
       are flipped: Q/K are the *key-query* pair (fewer heads) while V
       determines the *output* head count (more heads).  The attribute
       names ``q_num_heads`` and ``kv_num_heads`` therefore map to the
       *opposite* head counts compared to the standard Attention op.
       This matches the ONNX LinearAttention op proposal
       (onnx/onnx#7689) which uses the same naming convention.
        update_rule: One of "linear", "gated", "delta", "gated_delta".
        scale: Scalar multiplier applied to query before the recurrence.
            Per the ONNX LinearAttention op spec, should be set to
            ``1/sqrt(head_dim)`` for correct scaling.
        stash_type: Element type for the Scan body's internal
            computation.  Must match the precision of the inputs
            passed at the call site.  Defaults to ``FLOAT``.
        openvino_gdn: Preserve a native-layout RecurrentGDN function boundary
            and derive query scaling from sqrt(head_dim).
            This mode replaces the supplied scale with 1/sqrt(head_dim).
            Requires gated_delta with FLOAT inputs; decay is scalar per head.

    The function body handles 3D→4D reshape, GQA expansion, query
    scaling, and uses an ONNX Scan op for the sequential recurrence.
    Output is reshaped back to 3D.  A fused kernel can replace the
    entire function for 10-50x speedup.
    """
    valid_rules = ("linear", "gated", "delta", "gated_delta")
    if update_rule not in valid_rules:
        raise ValueError(
            f"Unknown update_rule: {update_rule!r}. Expected one of {valid_rules}."
        )
    if q_num_heads <= 0:
        raise ValueError(f"q_num_heads must be > 0; got {q_num_heads}")
    if kv_num_heads <= 0:
        raise ValueError(f"kv_num_heads must be > 0; got {kv_num_heads}")

    uses_decay = update_rule in ("gated", "gated_delta")
    uses_beta = update_rule in ("delta", "gated_delta")
    scalar_decay = openvino_gdn
    if openvino_gdn and (update_rule != "gated_delta" or stash_type != ir.DataType.FLOAT):
        raise ValueError("openvino_gdn requires gated_delta with FLOAT inputs")

    # --- Define function inputs ---
    # Always declare the full 6-input signature:
    # (query, key, value, past_state, decay, beta).
    # Update-rule-specific paths inside the function body ignore unused
    # trailing inputs so every variant exposes the same ir.Function
    # interface.  Call sites that don't need trailing inputs (decay, beta)
    # simply omit them — onnx-shape-inference >=0.1.9 accepts trailing
    # optional inputs (matching ONNX C++ behavior).
    inputs: list[ir.Value] = [
        ir.Value(name="query"),  # (B, T, q_num_heads * d_k)
        ir.Value(name="key"),  # (B, T, q_num_heads * d_k)
        ir.Value(name="value"),  # (B, T, kv_num_heads * d_v)
        ir.Value(name="past_state"),
        ir.Value(name="decay"),
        ir.Value(name="beta"),
    ]

    def body(op, query_v, key_v, value_v, past_state_v, decay_v, beta_v):

        # --- Reshape 3D → 4D using head counts ---
        b_dim = op.Shape(query_v, start=0, end=1)
        t_dim = op.Shape(query_v, start=1, end=2)

        # Q/K: [B, T, q_num_heads*d_k] → [B, T, q_num_heads, d_k]
        #     → transpose to [B, q_num_heads, T, d_k]
        qk_4d_shape = op.Concat(
            b_dim,
            t_dim,
            op.Constant(value_ints=[q_num_heads, -1]),
            axis=0,
        )
        query_4d = op.Transpose(
            op.Reshape(query_v, qk_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, q_num_heads, T, d_k]
        key_4d = op.Transpose(
            op.Reshape(key_v, qk_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, q_num_heads, T, d_k]

        # V: [B, T, kv_num_heads*d_v] → [B, kv_num_heads, T, d_v]
        # Reuse kv_4d_shape for both V and decay (same [B, T, kv_num_heads, -1]).
        kv_4d_shape = op.Concat(
            b_dim,
            t_dim,
            op.Constant(value_ints=[kv_num_heads, -1]),
            axis=0,
        )
        value_4d = op.Transpose(
            op.Reshape(value_v, kv_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, kv_num_heads, T, d_v]

        # --- GQA: expand Q/K heads to match V head count ---
        if kv_num_heads % q_num_heads != 0:
            raise ValueError(
                f"kv_num_heads ({kv_num_heads}) must be divisible by q_num_heads ({q_num_heads})"
            )
        gqa_ratio = kv_num_heads // q_num_heads
        query_expanded, key_expanded = _expand_kv_heads(
            op, query_4d, key_4d, gqa_ratio=gqa_ratio
        )

        # --- Reshape decay/beta 3D → 4D (only when used) ---
        # decay: (B, T, kv_num_heads * d_k) → (B, T, kv_num_heads, d_k)
        #     → transpose to (B, kv_num_heads, T, d_k)
        if uses_decay:
            if scalar_decay:
                decay_4d = op.Transpose(decay_v, perm=[0, 2, 1])  # [B,H,T]
            else:
                decay_4d = op.Transpose(
                    op.Reshape(decay_v, kv_4d_shape), perm=[0, 2, 1, 3]
                )  # [B, kv_num_heads, T, d_k]
        if uses_beta:
            # beta: (B, T, kv_num_heads) → transpose to (B, kv_num_heads, T)
            beta_3d = op.Transpose(beta_v, perm=[0, 2, 1])  # [B, kv_num_heads, T]

        # --- Apply query scale (matches op spec default of 1/sqrt(d_k)) ---
        # CastLike ensures scale constant matches the input dtype.
        if openvino_gdn:
            # The OpenVINO GDN matcher recognizes sqrt(head_dim) explicitly.
            # Scalar Gather preserves the scalar divisor used by that matcher.
            head_dim = op.Gather(op.Shape(query_expanded), 3, axis=0)
            divisor = op.Pow(op.Cast(head_dim, to=ir.DataType.FLOAT), 0.5)
            scaled_query = op.Div(query_expanded, divisor)
        else:
            scaled_query = op.Mul(
                query_expanded,
                op.CastLike(op.Constant(value_float=scale), query_expanded),
            )

        # --- Build Scan for sequential recurrence ---
        scan_body = _build_recurrence_body(uses_decay, uses_beta, stash_type=stash_type)

        # Keep the OpenVINO Loop in native [B, H, T, D] layout. Only Scan
        # requires time-major inputs. Native Loop layout matches the compact
        # recurrence representation used by OpenVINO exports.
        q_t = scaled_query if openvino_gdn else op.Transpose(scaled_query, perm=[2, 0, 1, 3])
        k_t = key_expanded if openvino_gdn else op.Transpose(key_expanded, perm=[2, 0, 1, 3])
        v_t = value_4d if openvino_gdn else op.Transpose(value_4d, perm=[2, 0, 1, 3])

        scan_inputs = [q_t, k_t, v_t]
        if uses_decay:
            decay_t = decay_4d if openvino_gdn else op.Transpose(decay_4d, perm=[2, 0, 1, 3])
            scan_inputs.append(decay_t)
        if uses_beta:
            beta_t = beta_3d if openvino_gdn else op.Transpose(beta_3d, perm=[2, 0, 1])
            scan_inputs.append(beta_t)

        if openvino_gdn:
            present_state_v, output_t = op.RecurrentGDN(
                *scan_inputs,
                past_state_v,
                _domain="org.openvinotoolkit",
                _version=1,
                _outputs=2,
            )
        else:
            present_state_v, output_t = op.Scan(
                past_state_v,  # carry: (B, H, d_k, d_v)
                *scan_inputs,
                body=scan_body,
                num_scan_inputs=len(scan_inputs),
                _outputs=2,
            )
        # present_state_v: (B, H, d_k, d_v)
        # output_t: (B, H, T, d_v) for Loop; (T, B, H, d_v) for Scan.

        # --- Reshape output 4D → 3D ---
        # (T, B, H, d_v) → (B, T, H, d_v) → (B, T, H*d_v)
        output_bthd = op.Transpose(
            output_t, perm=[0, 2, 1, 3] if openvino_gdn else [1, 0, 2, 3]
        )
        out_3d_shape = op.Concat(b_dim, t_dim, op.Constant(value_ints=[-1]), axis=0)
        output = op.Reshape(output_bthd, out_3d_shape)  # [B, T, H*d_v]

        output.name = "output"
        present_state_v.name = "present_state"
        return output, present_state_v

    # --- Build the ir.Function ---
    return build_function(
        body,
        inputs,
        domain=DOMAIN,
        name="LinearAttention",
        attributes=[
            ir.Attr(
                "update_rule",
                ir.AttributeType.STRING,
                update_rule,
                ref_attr_name="update_rule",
            ),
            ir.Attr(
                "scale",
                ir.AttributeType.FLOAT,
                scale,
                ref_attr_name="scale",
            ),
            ir.Attr(
                "q_num_heads",
                ir.AttributeType.INT,
                q_num_heads,
                ref_attr_name="q_num_heads",
            ),
            ir.Attr(
                "kv_num_heads",
                ir.AttributeType.INT,
                kv_num_heads,
                ref_attr_name="kv_num_heads",
            ),
        ],
        opset_imports={"": OPSET_VERSION},
    )


def recurrent_gdn() -> ir.Function:
    """Portable fallback for OpenVINO's native-layout, FLOAT GDN conversion boundary.

    Inputs are q/k/v [B,H,T,D], log-decay/beta [B,H,T], and state [B,H,Dk,Dv].
    Outputs are final state followed by the sequence. Query scaling is external.
    """
    inputs = [
        ir.Value(
            name=name, type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([None] * rank)
        )
        for name, rank in (("q", 4), ("k", 4), ("v", 4), ("g", 3), ("beta", 3), ("state", 4))
    ]

    def body(op, q, k, v, g, beta, state):
        shape = op.Shape(v)
        buffer = op.Expand(op.Constant(value_float=0.0), shape)
        trip = op.Gather(shape, 2, axis=0)
        *_, final_state, sequence = op.Loop(
            trip,
            op.Cast(op.Constant(value_int=1), to=ir.DataType.BOOL),
            q,
            k,
            v,
            g,
            beta,
            state,
            buffer,
            body=_build_openvino_gdn_body(ir.DataType.FLOAT),
            _outputs=7,
        )
        return final_state, sequence

    return build_function(
        body,
        inputs,
        domain="org.openvinotoolkit",
        name="RecurrentGDN",
        opset_imports={"": OPSET_VERSION},
    )


def _build_openvino_gdn_body(stash_type: ir.DataType) -> ir.Graph:
    """Standard ONNX recurrence used when the native conversion is unavailable."""
    from onnxscript import GraphBuilder

    dtype = ir.TensorType(stash_type)
    index = ir.Value(
        name="iteration", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
    )
    condition = ir.Value(
        name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
    )
    sequences = [
        ir.Value(name=name, type=dtype, shape=ir.Shape([None] * rank))
        for name, rank in (
            ("query_sequence", 4),
            ("key_sequence", 4),
            ("value_sequence", 4),
            ("decay_sequence", 3),
            ("beta_sequence", 3),
        )
    ]
    state = ir.Value(name="state", type=dtype, shape=ir.Shape([None] * 4))
    buffer = ir.Value(name="output_buffer", type=dtype, shape=ir.Shape([None] * 4))
    body = ir.Graph(
        [index, condition, *sequences, state, buffer],
        [],
        nodes=[],
        name="openvino_gdn",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(body).op
    q, k, v, decay, beta = [op.Gather(sequence, index, axis=2) for sequence in sequences]
    axes_neg1 = op.Constant(value_ints=[-1])
    axes_neg2 = op.Constant(value_ints=[-2])
    # Sliced scalar decay/beta become [B,H,1] after frontend conversion.
    # Unsqueeze(decay, -1) is equivalent to the slice Squeeze/Unsqueeze pair.
    decay_slice = op.Unsqueeze(decay, axes_neg1)
    decayed = op.Mul(state, op.Unsqueeze(op.Exp(decay_slice), axes_neg1))
    key_column = op.Unsqueeze(k, axes_neg1)
    retrieved = op.ReduceSum(op.Mul(decayed, key_column), axes_neg2, keepdims=0)
    delta = op.Mul(op.Sub(v, retrieved), op.Unsqueeze(beta, axes_neg1))
    updated = op.Add(decayed, op.Mul(key_column, op.Unsqueeze(delta, axes_neg2)))
    step = op.ReduceSum(op.Mul(updated, op.Unsqueeze(q, axes_neg1)), axes_neg2, keepdims=1)
    # A rank-one index survives ScatterElements-to-ScatterUpdate conversion
    # directly, matching ScatterUpdate(buffer, Unsqueeze(iteration, 0), step, 2).
    indices = op.Expand(op.Unsqueeze(index, op.Constant(value_ints=[0])), op.Shape(step))
    written = op.ScatterElements(buffer, indices, step, axis=2)
    body.outputs.extend(
        [
            op.Identity(condition),
            *[op.Identity(sequence) for sequence in sequences],
            updated,
            written,
        ]
    )
    body.sort()
    rename_subgraph_values(body, "gdn_")
    return body


def _build_recurrence_body(
    uses_decay: bool,
    uses_beta: bool,
    *,
    stash_type: ir.DataType = ir.DataType.FLOAT,
) -> ir.Graph:
    """Build the Scan body for single-token delta-rule recurrence.

    The body operates in ``stash_type`` precision.

    Body inputs (in order):
        1. state: (B, H, d_k, d_v) [carry]
        2. q_t: (B, H, d_k) [scan input]
        3. k_t: (B, H, d_k) [scan input]
        4. v_t: (B, H, d_v) [scan input]
        5. decay_t: (B, H, d_k) [scan input]
        6. beta_t: (B, H) [scan input]

    Body outputs:
        1. new_state: (B, H, d_k, d_v) [carry]
        2. output_t: (B, H, d_v) [scan output]
    """
    dtype = ir.TensorType(stash_type)

    # Only specify dtype, not shape. Shapes can be inferred from Scan op's
    # inputs. Hard-coding locally chosen symbolic dim names is incorrect
    # since ONNX has a global scope for symbolic dims. Even the dtype is
    # optional in that it can be inferred, but specifying it is safe and
    # can help as long as we know the exact type.
    state_in = ir.Value(name="state", type=dtype)
    q_t = ir.Value(name="q_t", type=dtype)
    k_t = ir.Value(name="k_t", type=dtype)
    v_t = ir.Value(name="v_t", type=dtype)
    scan_in_vals = [q_t, k_t, v_t]
    decay_t: ir.Value | None = None
    beta_t: ir.Value | None = None
    if uses_decay:
        decay_t = ir.Value(name="decay_t", type=dtype)
        scan_in_vals.append(decay_t)
    if uses_beta:
        beta_t = ir.Value(name="beta_t", type=dtype)
        scan_in_vals.append(beta_t)

    body_graph, body_builder = create_body_graph(
        state_inputs=[state_in],
        scan_inputs=scan_in_vals,
        name="delta_recurrence",
    )
    bop = body_builder.op

    # Shared axes constants (deduplicated)
    axes_neg2 = bop.Constant(value_ints=[-2])
    axes_neg1 = bop.Constant(value_ints=[-1])

    # --- State decay: state = exp(g) * past_state ---
    if uses_decay:
        # decay_t: (B, H, d_k) -> (B, H, d_k, 1) for broadcasting with state (B, H, d_k, d_v)
        g_exp = bop.Exp(bop.Unsqueeze(decay_t, axes_neg1))
        state = bop.Mul(state_in, g_exp)
    else:
        state = state_in

    # --- Retrieval: k @ state -> (B, H, d_v) ---
    # k_row: (B, H, 1, d_k) @ state: (B, H, d_k, d_v) -> (B, H, 1, d_v)
    k_row = bop.Unsqueeze(k_t, axes_neg2)
    retrieval = bop.Squeeze(bop.MatMul(k_row, state), axes_neg2)

    # --- State update ---
    if uses_beta:
        # delta = (v - retrieval) * beta
        delta = bop.Sub(v_t, retrieval)
        beta_expanded = bop.Unsqueeze(beta_t, axes_neg1)  # (B, H, 1)
        delta = bop.Mul(delta, beta_expanded)
    else:
        delta = v_t

    # Outer product: k^T @ delta -> (B, H, d_k, d_v)
    k_col = bop.Unsqueeze(k_t, axes_neg1)  # (B, H, d_k, 1)
    delta_row = bop.Unsqueeze(delta, axes_neg2)  # (B, H, 1, d_v)
    outer = bop.MatMul(k_col, delta_row)
    new_state = bop.Add(state, outer)
    new_state.name = "new_state"

    # --- Output: q @ new_state -> (B, H, d_v) ---
    q_row = bop.Unsqueeze(q_t, axes_neg2)  # (B, H, 1, d_k)
    output_t = bop.Squeeze(bop.MatMul(q_row, new_state), axes_neg2)
    output_t.name = "output_t"

    body_graph.outputs.extend([new_state, output_t])
    rename_subgraph_values(body_graph, "dn_")

    return body_graph


def _expand_kv_heads(op, query, key, *, gqa_ratio: int):
    """Expand Q/K heads to match V head count for GQA.

    When gqa_ratio > 1, each Q/K head is tiled ``gqa_ratio`` times along
    a new dim and then reshaped to merge heads.
    When gqa_ratio == 1, this is a no-op (returns inputs unchanged).

    The expansion ratio is computed at graph-build time so the Tile
    repeats are a static Constant — this avoids Shape→Gather→Div ops
    whose int64 outputs cause CUDA EP memory placement issues.

    Args:
        op: ONNX op builder.
        query: (B, H_kv, T, d_k)
        key: (B, H_kv, T, d_k)
        gqa_ratio: ``num_v_heads // num_k_heads`` (computed at build time).

    Returns:
        query: (B, H, T, d_k)
        key: (B, H, T, d_k)
    """
    if gqa_ratio == 1:
        return query, key

    # (B, H_kv, T, d_k) -> (B, H_kv, 1, T, d_k)
    axes_2 = op.Constant(value_ints=[2])
    q_5d = op.Unsqueeze(query, axes_2)
    k_5d = op.Unsqueeze(key, axes_2)

    # Tile along dim 2 by the static ratio
    repeat_vec = op.Constant(value_ints=[1, 1, gqa_ratio, 1, 1])
    q_tiled = op.Tile(q_5d, repeat_vec)
    k_tiled = op.Tile(k_5d, repeat_vec)

    # Reshape: (B, H_kv, ratio, T, d_k) -> (B, H_kv*ratio, T, d_k)
    # Extract B, T, d_k from the original 4D query to build the target shape.
    # We cannot use '0' sentinels because ONNX Reshape copies from the same
    # positional index, and the 5D→4D dimension mapping doesn't align.
    b_dim = op.Shape(query, start=0, end=1)
    t_dim = op.Shape(query, start=2, end=3)
    dk_dim = op.Shape(query, start=3, end=4)
    expanded_shape = op.Concat(
        b_dim,
        op.Constant(value_ints=[-1]),
        t_dim,
        dk_dim,
        axis=0,
    )
    return op.Reshape(q_tiled, expanded_shape), op.Reshape(k_tiled, expanded_shape)
