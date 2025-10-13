import torch
from torch import Tensor
from torchdt import DType

ZERO = torch.tensor(-32768, dtype=torch.int16) # smallest positive value in LNS
POS_INF = torch.tensor(32766, dtype=torch.int16) # largest positive value in LNS
NEG_INF = torch.tensor(32767, dtype=torch.int16) # largest negative value in LNS
base = 2.0 ** (2.0 ** torch.tensor(-10, dtype=torch.float64))

class LNS16(DType):
    bit_width = 16

    @staticmethod
    def from_float(t: Tensor) -> Tensor:
        t = t.to(dtype=torch.float64)
        abs_t = torch.abs(t)

        log_t = torch.log(abs_t) / torch.log(base)
        # clamp to first 15 bits then bitshift to 16 bits
        packed = torch.round(log_t).to(torch.int16).clamp(-16384, 16383) << 1 | (t < 0)

        lns_t = torch.where(
            abs_t == 0, ZERO,
            torch.where(
                torch.isposinf(t), POS_INF,
                torch.where(
                    torch.isneginf(t), NEG_INF,
                    packed.to(torch.int16))))
        return lns_t

    @staticmethod
    def to_float(t: Tensor) -> Tensor:
        packed = t.view(torch.int16)
        log_t = (packed >> 1)
        sign_t = torch.where((packed & 1) == 1, -1.0, 1.0).to(torch.float64)

        abs_t = torch.pow(base, log_t)
        float_t = sign_t * abs_t

        float_t = torch.where(
            packed == ZERO, 0.0,
            torch.where(
                packed == POS_INF, float('inf'),
                torch.where(
                    packed == NEG_INF, float('-inf'),
                    float_t)))
        return float_t.to(torch.float64)