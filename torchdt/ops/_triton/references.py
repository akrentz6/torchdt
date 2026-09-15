from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class Reference:
    formats: dict
    values: Callable
    initialize: tuple = ()
    valid_config: Callable | None = None


def _channel(value):
    return value.reshape(1, -1, 1, 1)


def _tiles(value, block):
    tail = (-value.shape[-1]) % block
    return F.pad(value, (0, tail)).reshape(*value.shape[:-1], -1, block).sum(-1)


def make_reference(name, a, decode, decode_acc, scalar):
    cache = {}

    def d(key):
        if key not in cache:
            cache[key] = decode(a[key])
        return cache[key]

    def acc(key):
        if key not in cache:
            cache[key] = decode_acc(a[key])
        return cache[key]

    def s(key):
        return scalar(a[key])

    def result(values, *, accum=(), raw=(), initialize=(), block=None):
        formats = {k: 'acc' if k in accum else 'raw' if k in raw else 'dtype' for k in values}
        valid = None if block is None else lambda c: c.kwargs['BLOCK'] == block
        return Reference(formats, lambda config: values, initialize, valid)

    if name == 'sum':
        values = d('x_ptr')
        reduced = values.mean(dim=1) if a['DO_MEAN'] else values.sum(dim=1)
        return result({'y_ptr': reduced})
    if name == 'matmul':
        return result({'c_ptr': torch.matmul(d('a_ptr'), d('b_ptr'))})

    if name.startswith('conv2d'):
        if name == 'conv2d_dbias':
            return result({'dB_ptr': d('dY_ptr').sum((0, 2, 3))})
        stride, padding, dilation = [(a[x], a[y]) for x, y in
                                     (('sh', 'sw'), ('ph', 'pw'), ('dh', 'dw'))]
        groups = a['groups']
        if name == 'conv2d':
            return result({'Y_ptr': F.conv2d(d('X_ptr'), d('W_ptr'),
                          d('B_ptr') if a['HAS_BIAS'] else None,
                          stride, padding, dilation, groups)})
        if name == 'conv2d_dinput':
            dummy = torch.empty_like(a['dX_ptr'], dtype=torch.float64)
            grad = torch.ops.aten.convolution_backward.default(
                d('dY_ptr'), dummy, d('W_ptr'), None, stride, padding,
                dilation, False, [0, 0], groups, [True, False, False])[0]
            return result({'dX_ptr': grad})
        if name == 'conv2d_dweight':
            x, dy = d('X_ptr'), d('dY_ptr')
            shape = (a['Cout'], a['Cin'] // groups, a['Kh'], a['Kw'])
            dummy = torch.empty(shape, dtype=x.dtype, device=x.device)
            splits = a['SPLIT_K']
            def weight_grad(upstream):
                return torch.ops.aten.convolution_backward.default(
                    upstream, x, dummy, None, stride, padding, dilation,
                    False, [0, 0], groups, [False, True, False])[1]
            if splits == 1:
                return result({'partial_ptr': weight_grad(dy)})
            # Split ownership changes with BLOCK_NHW. Mask the corresponding
            # upstream positions before asking ATen for each partial gradient.
            values_by_block = {}
            def values(config):
                block = config.kwargs['BLOCK_NHW']
                if block not in values_by_block:
                    indices = torch.arange(a['N'] * a['Hout'] * a['Wout'], device=x.device)
                    owner = ((indices // block) % splits).reshape(a['N'], 1, a['Hout'], a['Wout'])
                    partials = torch.stack([weight_grad(dy * (owner == i)).flatten()
                                            for i in range(splits)])
                    values_by_block[block] = {'partial_ptr': partials}
                return values_by_block[block]
            return Reference({'partial_ptr': 'acc'}, values)

    if name == 'max_pool2d':
        params = [(a[x], a[y]) for x, y in
                  (('Kh', 'Kw'), ('sh', 'sw'), ('ph', 'pw'), ('dh', 'dw'))]
        # ceil_mode is implicit in the supplied output geometry.
        value, indices = torch.ops.aten.max_pool2d_with_indices.default(d('X_ptr'), *params, False)
        if value.shape != a['Y_ptr'].shape:
            value, indices = torch.ops.aten.max_pool2d_with_indices.default(d('X_ptr'), *params, True)
        return result({'Y_ptr': value, 'idx_ptr': indices}, raw=('idx_ptr',))
    if name == 'adaptive_avg_pool2d':
        return result({'Y_ptr': F.adaptive_avg_pool2d(d('X_ptr'), (a['Hout'], a['Wout']))})
    if name == 'adaptive_avg_pool2d_dinput':
        dummy = torch.empty_like(a['dX_ptr'], dtype=torch.float64)
        return result({'dX_ptr': torch.ops.aten._adaptive_avg_pool2d_backward.default(d('dY_ptr'), dummy)})

    if name.startswith('nll_'):
        target = a['t_ptr'].long()
        valid = target != a['IGNORE_INDEX']
        safe = target.masked_fill(~valid, 0)
        weight = d('w_ptr') if a['HAS_WEIGHT'] else None
        weights = weight[safe] if weight is not None else torch.ones_like(target, dtype=torch.float64)
        denom = weights.masked_fill(~valid, 0)
        if name == 'nll_denominator':
            return result({'denom_ptr': denom})
        if name == 'nll_loss':
            loss = F.nll_loss(d('x_ptr'), target, weight=weight, reduction='none',
                              ignore_index=a['IGNORE_INDEX'])
            values = {'loss_ptr': loss}
            if a['HAS_DENOM']:
                values['denom_ptr'] = denom
            return result(values)
        if name == 'nll_loss_backward':
            # ATen's 2D NLL backward also covers spatial/scalar cases after
            # flattening the non-class dimensions. Use the saved denominator
            # supplied to this stage, including its quantization.
            shape = a['dx_ptr'].shape
            classes = shape[0] if len(shape) == 1 else shape[1]
            dummy = torch.empty((target.numel(), classes), device=target.device, dtype=torch.float64)
            reduction = 0 if a['REDUCTION_NONE'] else 1 if a['REDUCTION_MEAN'] else 2
            upstream = d('dy_ptr').reshape(-1) if reduction == 0 else d('dy_ptr').reshape(())
            total_weight = d('denom_ptr').reshape(()) if reduction == 1 else denom.sum()
            grad = torch.ops.aten.nll_loss_backward.default(
                upstream, dummy, target.reshape(-1), weight, reduction,
                a['IGNORE_INDEX'], total_weight)
            grad = grad.reshape(*target.shape, classes)
            grad = grad.reshape(shape) if len(shape) == 1 else grad.movedim(-1, 1)
            return result({'dx_ptr': grad}, initialize=('dx_ptr',))

    if name == 'batch_norm2d_sum':
        x = d('X_ptr').movedim(1, 0).flatten(1)
        return result({'partial_sum_ptr': _tiles(x, 128)}, accum=('partial_sum_ptr',), block=128)
    if name == 'batch_norm2d_centered_var':
        centered = d('X_ptr') - _channel(d('sm_ptr'))
        square = centered.square().movedim(1, 0).flatten(1)
        return result({'partial_var_ptr': _tiles(square, 128)}, accum=('partial_var_ptr',), block=128)
    if name == 'batch_norm2d_mean_finalize':
        mean = acc('partial_sum_ptr').sum(1) / s('count_dt')
        running = (1 - s('momentum')) * d('rm_ptr') + s('momentum') * mean
        return result({'rm_ptr': running, 'sm_ptr': mean}, initialize=('rm_ptr',))
    if name == 'batch_norm2d_var_finalize':
        count = s('count_dt')
        var = (acc('partial_var_ptr').sum(1) / count).clamp_min(0)
        sample = var * count / (count - 1) if a['count'] > 1 else torch.zeros_like(var)
        running = (1 - s('momentum')) * d('rv_ptr') + s('momentum') * sample
        return result({'rv_ptr': running, 'sis_ptr': torch.rsqrt(var + s('eps'))}, initialize=('rv_ptr',))
    if name == 'batch_norm2d_forward':
        if a['EVAL_FUSED']:
            mean = d('rm_ptr')
            invstd = torch.rsqrt(d('rv_ptr').clamp_min(0) + s('eps'))
        else:
            mean, invstd = d('sm_ptr'), d('sis_ptr')
        value = (d('X_ptr') - _channel(mean)) * _channel(invstd)
        if a['has_weight']:
            value = value * _channel(d('w_ptr'))
        if a['has_bias']:
            value = value + _channel(d('b_ptr'))
        values = {'Y_ptr': value}
        if a['EVAL_FUSED']:
            values.update(sm_ptr=mean, sis_ptr=invstd)
        return result(values)
    if name == 'batch_norm2d_backward_partials':
        dy = d('dY_ptr')
        xhat = (d('X_ptr') - _channel(d('sm_ptr'))) * _channel(d('sis_ptr'))
        # Unlike forward partials, backward tiles restart at every batch item.
        def partial(v):
            return _tiles(v.movedim(1, 0).flatten(2), 256).flatten(1)
        return result({'p_dy_ptr': partial(dy), 'p_dy_xhat_ptr': partial(dy * xhat)},
                      accum=('p_dy_ptr', 'p_dy_xhat_ptr'), block=256)
    if name == 'batch_norm2d_backward_finalize':
        dy, dyx = acc('p_dy_ptr').sum(1), acc('p_dy_xhat_ptr').sum(1)
        values = {'m_dy_ptr': dy / s('count_dt'), 'm_dy_xhat_ptr': dyx / s('count_dt')}
        if a['has_bias']:
            values['dB_ptr'] = dy
        if a['has_weight']:
            values['dW_ptr'] = dyx
        return result(values)
    if name in ('batch_norm2d_dinput_train', 'batch_norm2d_dinput_eval'):
        invstd = _channel(d('sis_ptr'))
        factor = invstd * _channel(d('w_ptr')) if a['has_weight'] else invstd
        inner = d('dY_ptr')
        if name == 'batch_norm2d_dinput_train':
            xhat = (d('X_ptr') - _channel(d('sm_ptr'))) * invstd
            inner = inner - _channel(d('m_dy_ptr')) - xhat * _channel(d('m_dy_xhat_ptr'))
        return result({'dX_ptr': factor * inner})
    raise ValueError(f'No numerical reference for {name!r}')
