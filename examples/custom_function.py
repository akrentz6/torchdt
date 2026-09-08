"""A differentiable square built from encoded primitives."""
import torch
from torchdt.autograd import DTFunction
from torchdt.lns import LNS16


class Square(DTFunction):
    @staticmethod
    def forward(ops, x):
        return ops.mul(x, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, ops, grad_output):
        (x,) = ctx.saved_tensors
        two_x = ops.add(x, x)
        return ops.mul(grad_output, two_x)


def main():
    x = torch.tensor([1.0, 2.0], dtype=LNS16, requires_grad=True)
    Square.apply(x).sum().backward()
    print(x.grad)
    torch.testing.assert_close(x.grad.to_float(), torch.tensor([2., 4.], dtype=torch.float32))


if __name__ == "__main__":
    main()
