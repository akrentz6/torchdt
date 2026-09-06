import math

import torch
import torch.nn.functional as F

from torchdt import DType


def _normalise_attention_mask(mask, scores, *, batch_heads=None):
    if mask is None:
        return None
    if batch_heads is not None and mask.dim() == 3 and mask.shape[0] == batch_heads[0] * batch_heads[1]:
        mask = mask.reshape(batch_heads[0], batch_heads[1], *mask.shape[1:])
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(0)
    try:
        return mask.expand(scores.shape)
    except RuntimeError as exc:
        raise RuntimeError(
            f"attention mask shape {tuple(mask.shape)} is not broadcastable to {tuple(scores.shape)}"
        ) from exc


def _apply_additive_mask(scores, mask):
    if mask is None:
        return scores, None
    mask = _normalise_attention_mask(mask, scores)
    if mask.dtype == torch.bool:
        return scores, mask

    values = mask.to_float() if isinstance(mask, DType) else mask
    if not torch.is_floating_point(values):
        raise TypeError("attention masks must be boolean or floating point")
    if torch.any(torch.isnan(values) | torch.isposinf(values)):
        raise ValueError("attention masks may not contain NaN or positive infinity")
    blocked = torch.isneginf(values)
    finite_values = torch.where(blocked, torch.zeros_like(values), values)
    encoded_values = scores.__class__(finite_values, device=scores.device)
    return torch.add(scores, encoded_values), blocked


def _masked_softmax(scores, blocked, dim=-1):
    if blocked is None:
        return torch.softmax(scores, dim=dim)
    blocked = blocked.expand(scores.shape)
    dim %= scores.dim()
    if scores.shape[dim] == 0:
        return scores.clone()

    valid = ~blocked
    first_value = scores.select(dim, 0)
    has_value = valid.select(dim, 0)
    zero = scores.__class__(0.0, device=scores.device)
    maximum = torch.where(has_value, first_value, zero)
    for position in range(1, scores.shape[dim]):
        candidate = scores.select(dim, position)
        candidate_valid = valid.select(dim, position)
        take = candidate_valid & (~has_value | torch.gt(candidate, maximum))
        maximum = torch.where(take, candidate, maximum)
        has_value = has_value | candidate_valid

    shifted = torch.sub(scores, maximum.unsqueeze(dim))
    exponentials = torch.where(valid, torch.exp(shifted), zero)
    denominator = torch.sum(exponentials, dim=dim, keepdim=True)
    safe_denominator = torch.where(has_value.unsqueeze(dim), denominator, scores.__class__(1.0, device=scores.device))
    probabilities = torch.div(exponentials, safe_denominator)
    return torch.where(has_value.unsqueeze(dim), probabilities, zero)


def _attention(query, key, value, attn_mask, dropout_p, is_causal, scale):
    if query.shape[-1] != key.shape[-1]:
        raise RuntimeError("query and key must have the same head dimension")
    if key.shape[-2] != value.shape[-2]:
        raise RuntimeError("key and value must have the same sequence length")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(torch.mul(query, scale), key.transpose(-2, -1))
    blocked = None
    masks = attn_mask if isinstance(attn_mask, (tuple, list)) else (attn_mask,)
    for mask in masks:
        scores, newly_blocked = _apply_additive_mask(scores, mask)
        if newly_blocked is not None:
            blocked = newly_blocked if blocked is None else blocked | newly_blocked

    if is_causal:
        causal = torch.ones(
            scores.shape[-2:], dtype=torch.bool, device=scores.device
        ).triu(1)
        causal = _normalise_attention_mask(causal, scores)
        blocked = causal if blocked is None else blocked | causal

    probabilities = _masked_softmax(scores, blocked, dim=-1)
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=True)
    return torch.matmul(probabilities, value), probabilities


@DType.register_func(F.scaled_dot_product_attention,
                     cast=("query", "key", "value"))
def dt_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if enable_gqa:
        raise NotImplementedError("grouped-query attention is not yet supported")
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p must be between 0 and 1")
    # SDPA uses True for allowed positions; MHA uses True for blocked positions.
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = ~attn_mask
    output, _ = _attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )
    return output


@DType.register_func(
    F.multi_head_attention_forward,
    cast=(
        "query", "key", "value", "in_proj_weight", "in_proj_bias",
        "out_proj_weight", "out_proj_bias",
    ),
)
def dt_multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
):
    if use_separate_proj_weight or any(
        tensor is not None for tensor in (q_proj_weight, k_proj_weight, v_proj_weight)
    ):
        raise NotImplementedError("separate Q/K/V projection weights are not yet supported")
    if bias_k is not None or bias_v is not None:
        raise NotImplementedError("bias_k and bias_v are not yet supported")
    if add_zero_attn:
        raise NotImplementedError("add_zero_attn is not yet supported")
    if static_k is not None or static_v is not None:
        raise NotImplementedError("static key/value tensors are not yet supported")
    if in_proj_weight is None:
        raise NotImplementedError("a combined in_proj_weight is required")
    if query.dim() not in (2, 3) or key.dim() != query.dim() or value.dim() != query.dim():
        raise RuntimeError("MHA query, key, and value must all be 2D or all be 3D")

    unbatched = query.dim() == 2
    if unbatched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

    target_length, batch_size, embed_dim = query.shape
    source_length = key.shape[0]
    if embed_dim != embed_dim_to_check or embed_dim % num_heads:
        raise RuntimeError("embed_dim must match embed_dim_to_check and be divisible by num_heads")
    if key.shape[1] != batch_size or value.shape[:2] != key.shape[:2]:
        raise RuntimeError("key/value batch and sequence dimensions must match")

    projection_weights = torch.chunk(in_proj_weight, 3, dim=0)
    projection_biases = (None, None, None)
    if in_proj_bias is not None:
        projection_biases = torch.chunk(in_proj_bias, 3, dim=0)
    q = F.linear(query, projection_weights[0], projection_biases[0])
    k = F.linear(key, projection_weights[1], projection_biases[1])
    v = F.linear(value, projection_weights[2], projection_biases[2])

    head_dim = embed_dim // num_heads
    q = q.reshape(target_length, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k.reshape(source_length, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    v = v.reshape(source_length, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)

    if attn_mask is not None:
        if attn_mask.dim() == 3 and attn_mask.shape[0] == batch_size * num_heads:
            attn_mask = attn_mask.reshape(batch_size, num_heads, target_length, source_length)
        elif attn_mask.dim() == 2:
            attn_mask = attn_mask.reshape(1, 1, target_length, source_length)
    if key_padding_mask is not None:
        if unbatched and key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        if tuple(key_padding_mask.shape) != (batch_size, source_length):
            raise RuntimeError(
                f"key_padding_mask must have shape {(batch_size, source_length)}"
            )
        key_padding_mask = key_padding_mask.reshape(batch_size, 1, 1, source_length)
        # Apply masks independently. Adding a bool mask to a finite additive mask
        # would turn blocked positions into +1 instead of preserving mask semantics.
        attn_mask = (attn_mask, key_padding_mask)

    output, weights = _attention(
        q,
        k,
        v,
        attn_mask,
        dropout_p if training else 0.0,
        is_causal,
        None,
    )
    output = output.permute(2, 0, 1, 3).reshape(target_length, batch_size, embed_dim)
    output = F.linear(output, out_proj_weight, out_proj_bias)

    if need_weights:
        if average_attn_weights:
            weights = torch.mean(weights, dim=1)
        if unbatched:
            weights = weights.squeeze(0)
    else:
        weights = None
    if unbatched:
        output = output.squeeze(1)
    return output, weights
