import math
import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

@register_base_op("mse_loss")
def dt_mse_loss(ops, x, y, reduction='mean', weight=None):
    errors = ops.sub(x, y)
    squared_errors = ops.square(errors)

    if weight is not None:
        squared_errors = ops.mul(squared_errors, weight)

    if reduction == 'none':
        return squared_errors

    elif reduction == 'sum':
        squared_error_sum = ops.sum(squared_errors)
        return squared_error_sum

    elif reduction == 'mean':
        squared_error_sum = ops.sum(squared_errors)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(squared_error_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = x.numel()
            mean = ops.div(
                squared_error_sum,
                ops.scalar_from_float(num_elements, device=x.device)
            )
            return mean

class DTMSELossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean', weight=None):
        return ops.mse_loss(x, y, reduction, weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction, weight = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors

        if weight is not None:
            grad = ops.sub(x, y)
            grad = ops.mul(grad, ops.scalar_from_float(2.0, device=x.device))
            grad = ops.mul(grad, weight)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad = ops.div(grad, weight_sum)

        else:
            grad = ops.sub(x, y)
            grad = ops.mul(grad, ops.scalar_from_float(2.0, device=x.device))

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad = ops.div(
                    grad, ops.scalar_from_float(num_elements, device=x.device)
                )

        grad_x = ops.mul(grad, grad_output)
        grad_y = ops.neg(grad_x)
        return grad_x, grad_y, None, None

@register_base_op("l1_loss")
def dt_l1_loss(ops, x, y, reduction='mean', weight=None):
    errors = ops.sub(x, y)
    abs_errors = ops.abs(errors)

    if weight is not None:
        abs_errors = ops.mul(abs_errors, weight)

    if reduction == 'none':
        return abs_errors

    elif reduction == 'sum':
        abs_error_sum = ops.sum(abs_errors)
        return abs_error_sum

    elif reduction == 'mean':
        abs_error_sum = ops.sum(abs_errors)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(abs_error_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = x.numel()
            mean = ops.div(
                abs_error_sum, ops.scalar_from_float(num_elements, device=x.device)
            )
            return mean

class DTL1LossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean', weight=None):
        return ops.l1_loss(x, y, reduction, weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction, weight = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors

        sign = ops.sign(ops.sub(x, y))

        if weight is not None:
            grad = ops.mul(sign, weight)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad = ops.div(grad, weight_sum)

        else:
            grad = sign

            if ctx.reduction == 'mean':
                num_elements = x.numel()
                grad = ops.div(
                    grad, ops.scalar_from_float(num_elements, device=x.device)
                )

        grad_x = ops.mul(grad, grad_output)
        grad_y = ops.neg(grad_x)
        return grad_x, grad_y, None, None

@register_base_op("binary_cross_entropy")
def dt_binary_cross_entropy(ops, x, y, weight=None, reduction='mean'):
    log_x = ops.log(x)
    pos_log_prob = ops.mul(y, log_x)
    x2 = ops.sub(ops.scalar_from_float(1.0, device=x.device), x)
    log_x2 = ops.log(x2)
    y2 = ops.sub(ops.scalar_from_float(1.0, device=y.device), y)
    neg_log_prob = ops.mul(y2, log_x2)

    loss = ops.add(pos_log_prob, neg_log_prob)
    if weight is not None:
        loss = ops.mul(loss, weight)
    loss = ops.neg(loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = x.numel()
            mean = ops.div(
                loss_sum, ops.scalar_from_float(num_elements, device=x.device)
            )
            return mean

class DTBCELossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean'):
        return ops.binary_cross_entropy(x, y, weight, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors

        if weight is not None:
            one_minus_x = ops.sub(ops.scalar_from_float(1.0, device=x.device), x)
            one_minus_y = ops.sub(ops.scalar_from_float(1.0, device=y.device), y)
            term1 = ops.div(one_minus_y, one_minus_x)
            term2 = ops.div(y, x)

            grad_x = ops.sub(term1, term2)
            grad_x = ops.mul(grad_x, weight)

            grad_y = ops.div(x, one_minus_x)
            grad_y = ops.log(grad_y)
            grad_y = ops.mul(grad_y, weight)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad_x = ops.div(grad_x, weight_sum)
                grad_y = ops.div(grad_y, weight_sum)

        else:
            one_minus_x = ops.sub(ops.scalar_from_float(1.0, device=x.device), x)
            one_minus_y = ops.sub(ops.scalar_from_float(1.0, device=y.device), y)
            term1 = ops.div(one_minus_y, one_minus_x)
            term2 = ops.div(y, x)

            grad_x = ops.sub(term1, term2)
            grad_y = ops.div(x, one_minus_x)
            grad_y = ops.log(grad_y)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                num_elements = ops.scalar_from_float(x.numel(), device=x.device)
                grad_x = ops.div(grad_x, num_elements)
                grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)
        return grad_x, grad_y, None, None

@register_base_op("binary_cross_entropy_with_logits")
def dt_binary_cross_entropy_with_logits(ops, x, y, weight=None, reduction='mean', pos_weight=None):
    if pos_weight is not None:
        raise NotImplementedError("pos_weight is not currently implemented.")

    sigmoid_x = ops.sigmoid(x)
    log_sigmoid_x = ops.log(sigmoid_x)
    pos_log_prob = ops.mul(y, log_sigmoid_x)

    sigmoid_x2 = ops.sub(
        ops.scalar_from_float(1.0, device=x.device), sigmoid_x
    )
    log_sigmoid_x2 = ops.log(sigmoid_x2)
    y2 = ops.sub(ops.scalar_from_float(1.0, device=y.device), y)
    neg_log_prob = ops.mul(y2, log_sigmoid_x2)

    loss = ops.add(pos_log_prob, neg_log_prob)
    if weight is not None:
        loss = ops.mul(loss, weight)
    loss = ops.neg(loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)

        if weight is not None:
            weight_sum = ops.sum(weight)
            weighted_mean = ops.div(loss_sum, weight_sum)
            return weighted_mean

        else:
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            mean = ops.div(loss_sum, num_elements)
            return mean

class DTBCEWithLogitsLossFunction(DTFunction):
    
    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean', pos_weight=None):
        return ops.binary_cross_entropy_with_logits(x, y, weight, reduction, pos_weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors

        if weight is not None:
            sigmoid_x = ops.sigmoid(x)
            grad_x = ops.sub(sigmoid_x, y)
            grad_x = ops.mul(grad_x, weight)

            grad_y = ops.mul(x, weight)
            grad_y = ops.neg(grad_y)

            if ctx.reduction == 'mean':
                weight_sum = ops.sum(weight)
                grad_x = ops.div(grad_x, weight_sum)
                grad_y = ops.div(grad_y, weight_sum)

        else:
            sigmoid_x = ops.sigmoid(x)
            grad_x = ops.sub(sigmoid_x, y)
            grad_y = ops.neg(x)

            if ctx.reduction == 'mean':
                num_elements = ops.scalar_from_float(x.numel(), device=x.device)
                grad_x = ops.div(grad_x, num_elements)
                grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

def _classification_view(x, target):
    if target.dtype not in (torch.int64, torch.int32):
        raise NotImplementedError("probability targets are not supported")
    if x.dim() == 0:
        raise ValueError("classification input must have at least one dimension")

    class_dim = 0 if x.dim() == 1 else 1
    classes = x.shape[class_dim]
    target_shape = tuple(x.shape[:class_dim]) + tuple(x.shape[class_dim + 1:])
    if tuple(target.shape) != target_shape:
        raise ValueError(
            f"Expected target shape {target_shape}, got {tuple(target.shape)}"
        )
    logits = x.movedim(class_dim, -1).reshape(-1, classes)
    return logits, target.reshape(-1), target_shape, class_dim


def _restore_classification_gradient(gradient, x_shape, class_dim):
    target_shape = tuple(x_shape[:class_dim]) + tuple(x_shape[class_dim + 1:])
    restored = gradient.reshape(*target_shape, x_shape[class_dim])
    return restored.movedim(-1, class_dim)


def _classification_loss(ops, values, target, target_shape, weight, reduction, ignore_index):
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"invalid reduction '{reduction}'")
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if torch.any((safe_target < 0) | (safe_target >= values.shape[1])):
        raise IndexError("Target is out of bounds")

    selected = values.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    zero = ops.scalar_from_float(0.0, device=values.device)
    if weight is None:
        sample_weight = ops.ones(selected.shape, device=values.device)
    else:
        if weight.dim() != 1 or weight.numel() != values.shape[1]:
            raise ValueError("weight must be one-dimensional with one value per class")
        sample_weight = weight[safe_target]
    sample_weight = torch.where(valid, sample_weight, zero)
    loss = torch.where(valid, ops.mul(ops.neg(selected), sample_weight), zero)

    if reduction == "none":
        return loss.reshape(target_shape)
    total = ops.sum(loss)
    if reduction == "sum":
        return total
    denominator = ops.sum(sample_weight)
    return ops.div(total, denominator)


@register_base_op("nll_loss")
def dt_nll_loss(ops, x, y, weight=None, reduction='mean', ignore_index=-100):
    values, target, target_shape, _ = _classification_view(x, y)
    return _classification_loss(
        ops, values, target, target_shape, weight, reduction, ignore_index
    )

class DTNLLLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, reduction='mean', ignore_index=-100):
        return ops.nll_loss(x, y, weight, reduction, ignore_index)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, reduction, ignore_index = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors
        values, target, _, class_dim = _classification_view(x, y)
        valid = target != ctx.ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        rows = torch.arange(target.numel(), device=x.device)
        flat_grad = ops.zeros_like(values)

        if weight is None:
            sample_weight = ops.ones(target.shape, device=x.device)
        else:
            sample_weight = weight[safe_target]
        zero = ops.scalar_from_float(0.0, device=x.device)
        sample_weight = torch.where(valid, sample_weight, zero)
        flat_grad[rows[valid], safe_target[valid]] = ops.neg(sample_weight[valid])

        if ctx.reduction == "mean":
            denominator = ops.sum(sample_weight)
            if ops.scalar_to_float(denominator) != 0.0:
                flat_grad = ops.div(flat_grad, denominator)
        if ctx.reduction == "none":
            flat_grad = ops.mul(flat_grad, grad_output.reshape(-1, 1))
        else:
            flat_grad = ops.mul(flat_grad, grad_output)

        grad_x = _restore_classification_gradient(flat_grad, x.shape, class_dim)
        return grad_x, None, None, None, None

@register_base_op("poisson_nll_loss")
def dt_poisson_nll_loss(ops, x, y, eps, log_input=True, full=False, reduction='mean'):
    if log_input:
        exp_x = ops.exp(x)
        loss = ops.sub(exp_x, ops.mul(y, x))
    else:
        log_x = ops.log(ops.add(x, eps))
        loss = ops.sub(x, ops.mul(y, log_x))

    if full:
        one = ops.scalar_from_float(1.0, device=y.device)
        y_clamped = torch.where(ops.gt(y, one), y, one)

        two_pi = ops.scalar_from_float(2.0 * math.pi, device=x.device)
        stirling_term1 = ops.mul(y_clamped, ops.log(y_clamped))
        stirling_term3 = ops.mul(
            ops.log(ops.mul(two_pi, y_clamped)),
            ops.scalar_from_float(0.5, device=x.device)
        )
        stirling = ops.add(ops.sub(stirling_term1, y_clamped), stirling_term3)

        loss = ops.add(loss, stirling)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.numel(), device=x.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

class DTPoissonNLLLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, eps, log_input=True, full=False, reduction='mean'):
        return ops.poisson_nll_loss(x, y, eps, log_input, full, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, eps, log_input, full, reduction = inputs
        ctx.save_for_backward(x, y, eps)
        ctx.log_input = log_input
        ctx.full = full
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, eps = ctx.saved_tensors

        if ctx.log_input:
            grad_x = ops.sub(ops.exp(x), y)
            grad_y = ops.neg(x)

        else:
            grad_x = ops.div(y, ops.add(x, eps))
            grad_x = ops.sub(ops.scalar_from_float(1.0, device=x.device), grad_x)
            grad_y = ops.neg(ops.log(ops.add(x, eps)))

        if ctx.full:
            stirling_grad = torch.where(
                ops.gt(y, ops.scalar_from_float(1.0, device=y.device)),
                                        ops.add(ops.log(y), ops.div(
                                            ops.scalar_from_float(0.5, device=y.device), y)),
                ops.scalar_from_float(0.0, device=y.device)
            )
            grad_y = ops.add(grad_y, stirling_grad)

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None, None, None

@register_base_op("hinge_embedding_loss")
def dt_hinge_embedding_loss(ops, x, y, margin=None, reduction='mean'):
    positive_mask = ops.eq(y, ops.scalar_from_float(1.0, device=y.device))
    loss = torch.where(
        positive_mask, x,
        ops.maximum(ops.scalar_from_float(0.0, device=x.device), ops.sub(margin, x))
    )

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.numel(), device=x.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

class DTHingeEmbeddingLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, margin, reduction='mean'):
        return ops.hinge_embedding_loss(x, y, margin, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, margin, reduction = inputs
        ctx.save_for_backward(x, y, margin)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, margin = ctx.saved_tensors

        grad_x = torch.where(
            ops.eq(y, ops.scalar_from_float(1.0, device=y.device)),
            ops.scalar_from_float(1.0, device=x.device),
            torch.where(
                ops.gt(
                    ops.sub(margin, x),
                    ops.scalar_from_float(0.0, device=x.device)
                ),
                ops.scalar_from_float(-1.0, device=x.device),
                ops.scalar_from_float(0.0, device=x.device)
            )
        )

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)

        grad_x = ops.mul(grad_x, grad_output)

        return grad_x, None, None, None

@register_base_op("kl_div")
def dt_kl_div(ops, x, y, reduction='mean', log_target=False):
    if log_target:
        loss = ops.mul(ops.exp(y), ops.sub(y, x))
    else:
        loss = ops.mul(y, ops.sub(ops.log(y), x))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.numel(), device=x.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

    elif reduction == 'batchmean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.size(0), device=x.device)
        batch_mean = ops.div(loss_sum, num_elements)
        return batch_mean

class DTKLDivLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, reduction='mean', log_target=False):
        return ops.kl_div(x, y, reduction, log_target)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, reduction, log_target = inputs
        ctx.save_for_backward(x, y)
        ctx.reduction = reduction
        ctx.log_target = log_target

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        if ctx.log_target:
            exp_y = ops.exp(y)
            grad_x = ops.neg(exp_y)
            grad_y = ops.mul(
                exp_y,
                ops.add(ops.sub(y, x), ops.scalar_from_float(1.0, device=x.device))
            )
        else:
            grad_x = ops.neg(y)
            grad_y = ops.add(
                ops.sub(ops.log(y), x), ops.scalar_from_float(1.0, device=x.device)
            )

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        elif ctx.reduction == 'batchmean':
            num_elements = ops.scalar_from_float(x.size(0), device=x.device)
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

@register_base_op("margin_ranking_loss")
def dt_margin_ranking_loss(ops, x1, x2, y, margin, reduction='mean'):
    loss = ops.sub(x1, x2)
    loss = ops.mul(loss, y)
    loss = ops.sub(margin, loss)
    loss = ops.maximum(ops.scalar_from_float(0.0, device=x1.device), loss)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x1.numel(), device=x1.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

class DTMarginRankingLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x1, x2, y, margin, reduction='mean'):
        return ops.margin_ranking_loss(x1, x2, y, margin, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x1, x2, y, margin, reduction = inputs
        ctx.save_for_backward(x1, x2, y, margin)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x1, x2, y, margin = ctx.saved_tensors

        loss = ops.sub(x1, x2)
        loss = ops.mul(loss, y)
        loss = ops.sub(margin, loss)
        gt_zero_mask = ops.gt(
            loss, ops.scalar_from_float(0.0, device=loss.device)
        )

        zero = ops.scalar_from_float(0.0, device=x1.device)
        grad_x1 = torch.where(gt_zero_mask, ops.neg(y), zero)
        grad_x2 = torch.where(gt_zero_mask, y, zero)
        grad_y = torch.where(gt_zero_mask, ops.sub(x2, x1), zero)

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x1.numel(), device=x1.device)
            grad_x1 = ops.div(grad_x1, num_elements)
            grad_x2 = ops.div(grad_x2, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x1 = ops.mul(grad_x1, grad_output)
        grad_x2 = ops.mul(grad_x2, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x1, grad_x2, grad_y, None, None

@register_base_op("gaussian_nll_loss")
def dt_gaussian_nll_loss(ops, x, y, var, eps, full=False, reduction='mean'):
    var_eps = ops.maximum(var, eps)
    loss = ops.square(ops.sub(x, y))
    loss = ops.add(ops.log(var_eps), ops.div(loss, var_eps))

    if full:
        two_pi = ops.scalar_from_float(2.0 * math.pi, device=x.device)
        loss = ops.add(loss, ops.log(two_pi))

    loss = ops.div(loss, ops.scalar_from_float(2.0, device=x.device))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = x.numel()
        mean = ops.div(
            loss_sum, ops.scalar_from_float(num_elements, device=x.device)
        )
        return mean

class DTGaussianNLLLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, var, eps, full=False, reduction='mean'):
        return ops.gaussian_nll_loss(x, y, var, eps, full, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, var, eps, _, reduction = inputs
        ctx.save_for_backward(x, y, var, eps)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, var, eps = ctx.saved_tensors

        var_eps = ops.maximum(var, eps)
        grad_x = ops.div(ops.sub(x, y), var_eps)
        grad_y = ops.neg(grad_x)

        grad_var = ops.square(ops.div(ops.sub(x, y), var))
        grad_var = ops.sub(ops.reciprocal(var), grad_var)
        grad_var = ops.div(
            grad_var, ops.scalar_from_float(2.0, device=x.device)
        )

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)
            grad_var = ops.div(grad_var, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)
        grad_var = ops.mul(grad_var, grad_output)

        return grad_x, grad_y, grad_var, None, None, None

@register_base_op("huber_loss")
def dt_huber_loss(ops, x, y, delta, reduction='mean', weight=None):
    two = ops.scalar_from_float(2.0, device=x.device)

    abs_diff = ops.abs(ops.sub(x, y))
    l1_term = ops.sub(abs_diff, ops.div(delta, two))
    l1_term = ops.mul(l1_term, delta)

    l2_term = ops.square(ops.sub(x, y))
    l2_term = ops.div(l2_term, two)

    loss = torch.where(ops.lt(abs_diff, delta), l2_term, l1_term)
    if weight is not None:
        loss = ops.mul(loss, weight)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.numel(), device=x.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

class DTHuberLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, delta, reduction='mean', weight=None):
        return ops.huber_loss(x, y, delta, reduction, weight)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, delta, reduction, weight = inputs
        ctx.save_for_backward(x, y, delta, weight)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, delta, weight = ctx.saved_tensors
        two = ops.scalar_from_float(2.0, device=x.device)

        l2_loss_grad_x = ops.sub(x, y)
        l2_loss_grad_y = ops.neg(l2_loss_grad_x)
        l1_loss_grad_x = ops.mul(ops.sign(ops.sub(x, y)), delta)
        l1_loss_grad_y = ops.neg(l1_loss_grad_x)

        abs_diff = ops.abs(ops.sub(x, y))
        l2_mask = ops.lt(abs_diff, delta)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        grad_w = None
        if weight is not None:
            abs_diff = ops.abs(ops.sub(x, y))
            l1_term = ops.sub(abs_diff, ops.div(delta, two))
            l1_term = ops.mul(l1_term, delta)

            l2_term = ops.square(ops.sub(x, y))
            l2_term = ops.div(l2_term, two)

            grad_w = torch.where(ops.lt(abs_diff, delta), l2_term, l1_term)
            grad_x = ops.mul(grad_x, weight)
            grad_y = ops.mul(grad_y, weight)

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)
            if weight is not None:
                grad_w = ops.div(grad_w, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)
        if weight is not None:
            grad_w = ops.mul(grad_w, grad_output)

        return grad_x, grad_y, None, None, grad_w

@register_base_op("smooth_l1_loss")
def dt_smooth_l1_loss(ops, x, y, beta, reduction='mean'):
    two = ops.scalar_from_float(2.0, device=x.device)

    abs_diff = ops.abs(ops.sub(x, y))
    l1_term = ops.sub(abs_diff, ops.div(beta, two))
    l2_term = ops.square(ops.sub(x, y))
    l2_term = ops.div(l2_term, ops.mul(two, beta))
    loss = torch.where(ops.lt(abs_diff, beta), l2_term, l1_term)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        loss_sum = ops.sum(loss)
        return loss_sum

    elif reduction == 'mean':
        loss_sum = ops.sum(loss)
        num_elements = ops.scalar_from_float(x.numel(), device=x.device)
        mean = ops.div(loss_sum, num_elements)
        return mean

class DTSmoothL1LossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, beta, reduction='mean'):
        return ops.smooth_l1_loss(x, y, beta, reduction)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, beta, reduction = inputs
        ctx.save_for_backward(x, y, beta)
        ctx.reduction = reduction

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, beta = ctx.saved_tensors

        l2_loss_grad_x = ops.div(ops.sub(x, y), beta)
        l2_loss_grad_y = ops.neg(l2_loss_grad_x)
        l1_loss_grad_x = ops.sign(ops.sub(x, y))
        l1_loss_grad_y = ops.neg(l1_loss_grad_x)

        abs_diff = ops.abs(ops.sub(x, y))
        l2_mask = ops.lt(abs_diff, beta)
        grad_x = torch.where(l2_mask, l2_loss_grad_x, l1_loss_grad_x)
        grad_y = torch.where(l2_mask, l2_loss_grad_y, l1_loss_grad_y)

        if ctx.reduction == 'mean':
            num_elements = ops.scalar_from_float(x.numel(), device=x.device)
            grad_x = ops.div(grad_x, num_elements)
            grad_y = ops.div(grad_y, num_elements)

        grad_x = ops.mul(grad_x, grad_output)
        grad_y = ops.mul(grad_y, grad_output)

        return grad_x, grad_y, None, None

@register_base_op("cross_entropy")
def dt_cross_entropy(ops, x, y, weight = None, ignore_index = -100, reduction = 'mean', label_smoothing = 0.0):
    if label_smoothing is not None and ops.scalar_to_float(label_smoothing) != 0.0:
        raise NotImplementedError("label_smoothing is not currently implemented.")
    values, target, target_shape, _ = _classification_view(x, y)
    log_probs = ops.log_softmax(values, dim=-1)
    return _classification_loss(
        ops, log_probs, target, target_shape, weight, reduction, ignore_index
    )

class DTCrossEntropyLossFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y, weight=None, ignore_index=-100, reduction='mean', label_smoothing=None):
        return ops.cross_entropy(x, y, weight, ignore_index, reduction, label_smoothing)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y, weight, ignore_index, reduction, _ = inputs
        ctx.save_for_backward(x, y, weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, weight = ctx.saved_tensors
        values, target, _, class_dim = _classification_view(x, y)
        valid = target != ctx.ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        rows = torch.arange(target.numel(), device=x.device)

        flat_grad = ops.exp(ops.log_softmax(values, dim=-1))
        if weight is None:
            sample_weight = ops.ones(target.shape, device=x.device)
        else:
            sample_weight = weight[safe_target]
        zero = ops.scalar_from_float(0.0, device=x.device)
        one = ops.scalar_from_float(1.0, device=x.device)
        sample_weight = torch.where(valid, sample_weight, zero)
        flat_grad = ops.mul(flat_grad, sample_weight.unsqueeze(1))
        selected = flat_grad[rows[valid], safe_target[valid]]
        flat_grad[rows[valid], safe_target[valid]] = ops.sub(
            selected, sample_weight[valid]
        )
        flat_grad = torch.where(valid.unsqueeze(1), flat_grad, zero)

        if ctx.reduction == "mean":
            denominator = ops.sum(sample_weight)
            if ops.scalar_to_float(denominator) != 0.0:
                flat_grad = ops.div(flat_grad, denominator)
        if ctx.reduction == "none":
            flat_grad = ops.mul(flat_grad, grad_output.reshape(-1, 1))
        else:
            flat_grad = ops.mul(flat_grad, grad_output)

        grad_x = _restore_classification_gradient(flat_grad, x.shape, class_dim)
        return grad_x, None, None, None, None, None
