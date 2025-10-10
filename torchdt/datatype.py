import torch
from torch import Tensor
from typing import Any, Optional, Union

_float_dtype = {
    8: torch.float8_e5m2, # we have several variants to pick from
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

_int_dtype = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64
}


class DType(Tensor):
    """
    Parent class for custom dtypes (posit, LNS, etc) that live in a Tensor
    but expose their own semantics.
    """
    bit_width: int = 32 # subclasses override

    def __new__(
            cls,
            data: Any,
            *,
            bit_width: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            requires_grad: bool = False,
            memory_format: torch.memory_format = torch.preserve_format,
    ):
        bw = bit_width if bit_width is not None else cls.bit_width
        if bw not in _float_dtype:
            raise ValueError("bit_width must be 8 / 16 / 32 / 64")

        f_dtype = _float_dtype[bw]

        if isinstance(data, torch.Tensor):
            payload = data.to(dtype=f_dtype, device=device, memory_format=memory_format)
            payload = cls.from_float(payload).view(f_dtype)
        else:
            payload = torch.tensor(data, dtype=f_dtype, device=device)
            payload = cls.from_float(payload).view(f_dtype)
            payload = payload.to(memory_format=memory_format)

        obj = Tensor._make_subclass(cls, payload, requires_grad)
        obj.bit_width = bw
        return obj

    @staticmethod
    def from_float(t: Tensor) -> Tensor:
        "Convert a standard float tensor to this datatype."
        raise NotImplementedError

    @staticmethod
    def to_float(t: Tensor) -> Tensor:
        "Convert this datatype tensor to a standard float tensor."
        raise NotImplementedError

    @property
    def _float(self) -> Tensor:
        "Return the underlying storage as a plain *float* tensor."
        return self.as_subclass(Tensor)

    @property
    def _int(self) -> Tensor:
        "Integer bit-view of the same storage (no copy)."
        return self._float.view(_int_dtype[self.bit_width])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.to_float(self._float)}, bit_width={self.bit_width}, "
            f"shape={tuple(self.shape)}, device={self.device})"
        )
