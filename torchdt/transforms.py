import torch
from typing import Any, Dict, Tuple, Type, Callable, Union, Optional

try:
    import torchvision.transforms.v2
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

__all__ = [
    "register_collate_dtype_fn",
    "ToDType",
    "DTypeNormalize",
]

def register_collate_dtype_fn(dtype_cls):

    def collate_fn(
            batch,
            *,
            collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
    ):
        return torch.stack(batch, 0)
    torch.utils.data._utils.collate.default_collate_fn_map.update({dtype_cls: collate_fn})

class ToDType:
    """
    Convert an image-like object (PIL image or numpy.ndarray) with shape
    (H x W x C) in the range [0, 255] to a DType with shape (C x H x W)
    in the range [0., 1.]. This is analogous to torchvision.transforms.ToTensor
    but wraps the output in the given DType.

    Non-floating point inputs are not converted to the DType unless specified
    by the `wrap_all` parameter.
    """

    def __init__(
            self,
            dtype: type,
            wrap_all: bool = False,
            device: Optional[Union[str, torch.device]] = None
    ):
        if not _TV_AVAILABLE:
            raise ImportError(
                "torchvision is required for ToDType transform. "
                "Install it with:  pip install torchvision."
            )

        self.dtype = dtype
        self.wrap_all = wrap_all
        self.device = torch.device(device) if device is not None else None

        self.pipeline = torchvision.transforms.v2.Compose((
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(dtype.float_dtype, scale=True),
        ))

    def __call__(self, img: Any):
        if torch.is_tensor(img):
            tensor = img
        else:
            tensor = self.pipeline(img)

        if not self.wrap_all and not tensor.is_floating_point():
            # Return unwrapped tensor for integer types unless wrap_all is True
            if self.device is not None:
                tensor = tensor.to(self.device)
            return tensor

        if self.device is not None:
            tensor = tensor.to(self.device)

        return self.dtype(tensor)

class DTypeNormalize:
    """
    Normalize a DType with mean and standard deviation. This is analogous to
    torchvision.transforms.Normalize but performs normalization on the DType.
    """

    def __init__(
        self,
        dtype: type,
        mean: Union[float, Tuple[float, ...]],
        std: Union[float, Tuple[float, ...]],
        device: Optional[Union[str, torch.device]] = None
    ):
        if not isinstance(mean, (float, tuple)):
            raise ValueError(f"mean must be a float or tuple of floats, got {type(mean)}.")
        if not isinstance(std, (float, tuple)):
            raise ValueError(f"std must be a float or tuple of floats, got {type(std)}.")

        self.mean = dtype(mean, device=device)
        self.std = dtype(std, device=device)

        if not (self.mean.ndim == 0 or self.mean.ndim == 1):
            raise ValueError(f"mean must be a scalar or 1D tensor, got shape {self.mean.shape}.")
        if not (self.std.ndim == 0 or self.std.ndim == 1):
            raise ValueError(f"std must be a scalar or 1D tensor, got shape {self.std.shape}.")

        self.mean = self.mean.unsqueeze(-1).unsqueeze(-1)
        self.std = self.std.unsqueeze(-1).unsqueeze(-1)

    def __call__(self, tensor):
        if not tensor.ndim >= 3:
            raise ValueError(f"Input tensor must have at least 3 dimensions (C, H, W), got shape {tensor.shape}.")

        if not (self.mean.numel() == tensor.shape[-3] or self.mean.numel() == 1):
            raise ValueError(
                f"Mean tensor length {self.mean.numel()} does not match "
                f"the number of channels {tensor.shape[-3]}."
            )

        if not (self.std.numel() == tensor.shape[-3] or self.std.numel() == 1):
            raise ValueError(
                f"Std tensor length {self.std.numel()} does not match "
                f"the number of channels {tensor.shape[-3]}."
            )

        return (tensor - self.mean) / self.std