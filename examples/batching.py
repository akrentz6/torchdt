"""Batch encoded inputs while preserving ordinary integer class labels."""
import torch
from torch.utils.data import DataLoader
from torchdt.lns import LNS16


def main():
    samples = [
        (torch.tensor([1.0, 2.0], dtype=LNS16), 0),
        (torch.tensor([3.0, 4.0], dtype=LNS16), 1),
    ]
    inputs, labels = next(iter(DataLoader(samples, batch_size=2)))
    print("Inputs:", inputs.to_float())
    print("Labels:", labels)
    assert isinstance(inputs, LNS16)
    assert labels.dtype == torch.int64
    assert tuple(inputs.shape) == (2, 2)


if __name__ == "__main__":
    main()
