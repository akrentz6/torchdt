"""Create LNS tensors, calculate, and compare with a float baseline."""
import torch
from torchdt.lns import LNS16


def main():
    LNS16.set_prec(7)
    values = torch.tensor([0.3, 0.5, 1.0, 2.0])
    x = torch.tensor(values, dtype=LNS16)
    decoded = x.to_float()
    print("Decoded:", decoded)
    print("Encoding error:", (decoded - values).abs())
    product = (x * x).to_float()
    print("Squares:", product)
    torch.testing.assert_close(product, values.square().to(product.dtype), rtol=0.02, atol=0.002)

    leaf = torch.tensor([1.0, 2.0, 4.0], dtype=LNS16, requires_grad=True)
    (leaf * leaf).sum().backward()
    print("Gradient:", leaf.grad.to_float())
    torch.testing.assert_close(
        leaf.grad.to_float(), torch.tensor([2., 4., 8.], dtype=leaf.grad.to_float().dtype),
        rtol=0.02, atol=0.002,
    )


if __name__ == "__main__":
    main()
