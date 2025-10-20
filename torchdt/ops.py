from torch import CharTensor, ShortTensor, IntTensor, LongTensor
from typing import Union

InternalTensor = Union[CharTensor, ShortTensor, IntTensor, LongTensor]

__all__ = [
    "OpsBase",
]

class OpsBase:

    @staticmethod
    def add(x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @staticmethod
    def sub(x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @staticmethod
    def mul(x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @staticmethod
    def div(x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError