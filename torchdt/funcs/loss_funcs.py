import torch
from torch.nn import _reduction as _Reduction
from torchdt import DType
from torchdt.ops.loss_ops import (
    DTMSELossFunction,
    DTL1LossFunction,
    DTBCELossFunction,
    DTBCEWithLogitsLossFunction,
    DTNLLLossFunction,
    DTPoissonNLLLossFunction,
    DTHingeEmbeddingLossFunction,
    DTKLDivLossFunction,
    DTMarginRankingLossFunction,
    DTGaussianNLLLossFunction,
    DTHuberLossFunction,
    DTSmoothL1LossFunction,
    DTCrossEntropyLossFunction,
)

@DType.register_func(torch.nn.functional.mse_loss,
                     cast=("input", "target", "weight"))
def mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTMSELossFunction.apply(input, target, reduction, weight)

@DType.register_func(torch.nn.functional.l1_loss,
                     cast=("input", "target", "weight"))
def l1_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTL1LossFunction(input, target, reduction=reduction, weight=weight)

@DType.register_func(torch.nn.functional.binary_cross_entropy,
                     cast=("input", "target", "weight"))
def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTBCELossFunction.apply(input, target, weight, reduction)

@DType.register_func(torch.nn.functional.binary_cross_entropy_with_logits,
                     cast=("input", "target", "weight", "pos_weight"))
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTBCEWithLogitsLossFunction(input, target, weight, reduction, pos_weight)

@DType.register_func(torch.nn.functional.nll_loss,
                     cast=("input", "weight"))
def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTNLLLossFunction.apply(input, target, weight, reduction, ignore_index)

@DType.register_func(torch.nn.functional.poisson_nll_loss,
                     cast=("input", "target"))
def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTPoissonNLLLossFunction.apply(input, target, eps, log_input, full, reduction)

@DType.register_func(torch.nn.functional.hinge_embedding_loss,
                     cast=("input", "target", "margin"))
def hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTHingeEmbeddingLossFunction.apply(input, target, margin, reduction)

@DType.register_func(torch.nn.functional.kl_div,
                     cast=("input", "target", "weight"))
def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTKLDivLossFunction.apply(input, target, reduction, log_target)

@DType.register_func(torch.nn.functional.margin_ranking_loss,
                     cast=("input1", "input2", "target"))
def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTMarginRankingLossFunction.apply(input1, input2, target, margin, reduction)

@DType.register_func(torch.nn.functional.gaussian_nll_loss,
                     cast=("input", "target", "var", "eps"))
def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction='mean'):
    return DTGaussianNLLLossFunction.apply(input, target, var, eps, full, reduction)

@DType.register_func(torch.nn.functional.huber_loss,
                     cast=("input", "target", "delta", "weight"))
def huber_loss(input, target, reduction='mean', delta=1.0, weight=None):
    return DTHuberLossFunction.apply(input, target, delta, reduction, weight)

@DType.register_func(torch.nn.functional.smooth_l1_loss)
def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTSmoothL1LossFunction.apply(input, target, beta, reduction)

@DType.register_func(torch.nn.functional.cross_entropy,
                     cast=("input", "weight", "label_smoothing"))
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return DTCrossEntropyLossFunction.apply(input, target, weight, ignore_index, reduction, label_smoothing)