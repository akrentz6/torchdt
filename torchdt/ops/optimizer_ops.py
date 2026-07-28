import torch

from torchdt.ops import register_base_op


def _is_nonzero(ops, value):
    return value.item() != ops.encoded_scalar(0.0)


@register_base_op("sgd_step")
def sgd_step(
    ops,
    params,
    grads,
    momentum_buffers,
    lr,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    maximize,
    use_momentum,
):
    outputs = list(momentum_buffers)
    use_weight_decay = _is_nonzero(ops, weight_decay)
    one_minus_dampening = None
    if use_momentum:
        one_minus_dampening = ops.sub(ops.ones_like(dampening), dampening)

    for index, (param, grad, momentum_buffer) in enumerate(
        zip(params, grads, momentum_buffers)
    ):
        if maximize:
            grad = ops.neg(grad)
        if use_weight_decay:
            grad = ops.add(grad, ops.mul(param, weight_decay))

        if use_momentum:
            if momentum_buffer is None:
                momentum_buffer = torch.empty_like(param)
                momentum_buffer.copy_(grad)
            else:
                momentum_buffer.copy_(
                    ops.add(
                        ops.mul(momentum_buffer, momentum),
                        ops.mul(grad, one_minus_dampening),
                    )
                )
            outputs[index] = momentum_buffer

            if nesterov:
                grad = ops.add(grad, ops.mul(momentum_buffer, momentum))
            else:
                grad = momentum_buffer

        param.copy_(ops.sub(param, ops.mul(grad, lr)))

    return outputs


@register_base_op("madam_step")
def madam_step(
    ops,
    params,
    grads,
    exp_avg_sqs,
    maxima,
    lr,
    beta,
    eps,
    g_bound,
    bias_corr,
    use_pow,
    maximize,
):
    one_minus_beta = ops.sub(ops.ones_like(beta), beta)
    negative_bound = ops.neg(g_bound)

    for param, grad, exp_avg_sq, maximum in zip(
        params, grads, exp_avg_sqs, maxima
    ):
        updated_exp_avg_sq = ops.add(
            ops.mul(beta, exp_avg_sq),
            ops.mul(ops.mul(grad, grad), one_minus_beta),
        )
        exp_avg_sq.copy_(updated_exp_avg_sq)

        corrected = ops.add(ops.div(updated_exp_avg_sq, bias_corr), eps)
        normalized = ops.div(grad, ops.sqrt(corrected))
        clipped = ops.clamp(normalized, negative_bound, g_bound)
        delta = ops.mul(ops.mul(lr, clipped), ops.sign(param))
        if not maximize:
            delta = ops.neg(delta)

        if use_pow:
            updated = ops.mul(param, ops.exp(delta))
        else:
            updated = ops.mul(param, ops.add(ops.ones_like(delta), delta))
        param.copy_(ops.clamp(updated, ops.neg(maximum), maximum))

    return exp_avg_sqs


@register_base_op("adam_step")
def adam_step(
    ops,
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_corr1,
    bias_corr2,
    amsgrad,
    maximize,
):
    use_weight_decay = _is_nonzero(ops, weight_decay)
    one_minus_beta1 = ops.sub(ops.ones_like(beta1), beta1)
    one_minus_beta2 = ops.sub(ops.ones_like(beta2), beta2)
    step_size = ops.div(ops.mul(lr, ops.sqrt(bias_corr2)), bias_corr1)

    for param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq in zip(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs
    ):
        if maximize:
            grad = ops.neg(grad)
        if use_weight_decay:
            grad = ops.add(grad, ops.mul(param, weight_decay))

        updated_exp_avg = ops.add(
            ops.mul(exp_avg, beta1),
            ops.mul(grad, one_minus_beta1),
        )
        updated_exp_avg_sq = ops.add(
            ops.mul(exp_avg_sq, beta2),
            ops.mul(ops.mul(grad, grad), one_minus_beta2),
        )
        exp_avg.copy_(updated_exp_avg)
        exp_avg_sq.copy_(updated_exp_avg_sq)

        if amsgrad:
            updated_max = ops.maximum(max_exp_avg_sq, updated_exp_avg_sq)
            max_exp_avg_sq.copy_(updated_max)
            denominator_state = updated_max
        else:
            denominator_state = updated_exp_avg_sq

        denominator = ops.add(ops.sqrt(denominator_state), eps)
        update = ops.div(ops.mul(step_size, updated_exp_avg), denominator)
        param.copy_(ops.sub(param, update))

    return exp_avgs, exp_avg_sqs, max_exp_avg_sqs
