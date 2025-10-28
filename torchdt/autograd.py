import torch
from torch.utils._pytree import tree_flatten
from torch.autograd.graph import get_gradient_edge

__all__ = ["DTFunction"]

def _find_first_grad_tensor(args):
    # Flatten the pytree into a list of leaves
    leaves, _ = tree_flatten(args)

    for leaf in leaves:
        if isinstance(leaf, torch.Tensor) and leaf.requires_grad:
            return leaf

    return None

class DTFunction(torch.autograd.Function):
    """
    Parent class for custom autograd Functions that work with DType tensors.
    Subclasses should implement static methods `forward` and `backward` (and
    optionally `setup_context`).
    """

    @classmethod
    def apply(cls, *args, **kwargs):
        from torchdt import DType # avoid circular import

        if kwargs:
            raise ValueError(
                "torch.autograd.Function does not support keyword arguments. "
                "Please use positional arguments only."
            )

        leaves, _ = tree_flatten(args)
        subtypes = [type(obj) for obj in leaves if isinstance(obj, DType)]

        dtype = subtypes[0] if len(subtypes) > 0 else None
        if len(subtypes) != 0 and any(st != dtype for st in subtypes):
            raise ValueError("All DType arguments to DTFunction.apply() must be of the same type.")

        # perform the operation using the Ops class for this DType
        result = super().apply(dtype.ops, *args)

        # get gradient edge to correctly handle grads for DType tensors
        first_tensor = _find_first_grad_tensor(args)
        if first_tensor is not None:
            edge = get_gradient_edge(first_tensor)

            j = 0
            for arg in args:

                # ignore non-tensor arguments
                if not isinstance(arg, torch.Tensor):
                    continue
                j += 1

                # only register hooks for DType inputs with gradients
                if not (isinstance(arg, dtype) and arg.requires_grad):
                    continue

                arg._track_operation(edge, j - 1)

        return result