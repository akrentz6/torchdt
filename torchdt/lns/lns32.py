import torch
from torch import Tensor
from torchdt import DType

ZERO = torch.tensor(-2_147_483_648, dtype=torch.int32) # smallest positive value in LNS
POS_INF = torch.tensor(2_147_483_646, dtype=torch.int32) # largest positive value in LNS
NEG_INF = torch.tensor(2_147_483_647, dtype=torch.int32) # largest negative value in LNS
base = 2.0 ** (2.0 ** torch.tensor(-20, dtype=torch.float64))

class LNS32(DType, bitwidth=32):
    pass

