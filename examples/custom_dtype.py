"""A small fixed-point format illustrating TorchDT's extension interface."""
import torch
from torchdt import DType, support_matrix


# format-start
class Fixed16(DType, bitwidth=16):
    """Signed 16-bit codes representing code / 256."""

    conversion_dtype = torch.float64
    scale = 256
    min_code = -32768
    max_code = 32767


@Fixed16.register_op("from_float")
def from_float(ops, values):
    if not torch.isfinite(values).all():
        raise ValueError("Fixed16 requires finite input values")
    dtype = ops.dtype
    values = values.to(torch.float64).clamp(
        dtype.min_code / dtype.scale, dtype.max_code / dtype.scale
    )
    return torch.round(values * dtype.scale).to(dtype.int_dtype)


@Fixed16.register_op("to_float")
def to_float(ops, codes):
    return codes.to(torch.float64) / ops.dtype.scale


@Fixed16.register_op("add")
def add(ops, x, y):
    # Widen before addition so int16 overflow cannot precede saturation.
    codes = x.to(torch.int32) + y.to(torch.int32)
    return codes.clamp(ops.dtype.min_code, ops.dtype.max_code).to(ops.dtype.int_dtype)


@Fixed16.register_op("mul")
def mul(ops, x, y):
    product = x.to(torch.int64) * y.to(torch.int64)
    # Each int16 product is exactly representable in float64. Rescale and
    # round ties to even before narrowing back to the stored code type.
    codes = torch.round(product.to(torch.float64) / ops.dtype.scale)
    return codes.clamp(ops.dtype.min_code, ops.dtype.max_code).to(ops.dtype.int_dtype)
# format-end


# demo-start
def demo():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=Fixed16, requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    print("Loss:", loss.item())       # 5.25
    print("Gradient:", x.grad.to_float())  # [-2, 1, 4]
    print("Addition backend:", support_matrix()["Fixed16"]["cpu"]["operations"]["add"])
    torch.testing.assert_close(loss.to_float(), torch.tensor(5.25, dtype=torch.float64))
    torch.testing.assert_close(
        x.grad.to_float(), torch.tensor([-2.0, 1.0, 4.0], dtype=torch.float64)
    )
# demo-end


def validate():
    """Check the format policy and inherited broadcasting/autograd paths."""
    # Halfway values round to even codes, including negative values.
    x = torch.tensor([0.5 / 256, 1.5 / 256, -0.5 / 256, -1.5 / 256], dtype=Fixed16)
    torch.testing.assert_close(x._int, torch.tensor([0, 2, 0, -2], dtype=torch.int16))

    # Conversion and arithmetic saturate, rather than wrapping int16 codes.
    x = torch.tensor([-1000.0, 1000.0], dtype=Fixed16)
    torch.testing.assert_close(x._int, torch.tensor([-32768, 32767], dtype=torch.int16))
    x = torch.tensor([-100.0, 100.0], dtype=Fixed16)
    torch.testing.assert_close((x + x)._int, torch.tensor([-32768, 32767], dtype=torch.int16))
    torch.testing.assert_close((x * x)._int, torch.tensor([32767, 32767], dtype=torch.int16))
    for value in (float('nan'), float('inf'), -float('inf')):
        try:
            torch.tensor(value, dtype=Fixed16)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite inputs should be rejected")

    # Broadcast gradients must reduce back to each operand's shape.
    x = torch.tensor([[1.0], [2.0]], dtype=Fixed16, requires_grad=True)
    y = torch.tensor([[0.5, 1.0]], dtype=Fixed16, requires_grad=True)
    result = x * y
    torch.testing.assert_close(
        result.to_float(), torch.tensor([[0.5, 1.0], [1.0, 2.0]], dtype=torch.float64)
    )
    result.sum().backward()
    torch.testing.assert_close(x.grad.to_float(), torch.full((2, 1), 1.5, dtype=torch.float64))
    torch.testing.assert_close(y.grad.to_float(), torch.full((1, 2), 3.0, dtype=torch.float64))

    # Independent graph branches accumulate with the format's addition.
    x = torch.tensor([1.0, 2.0], dtype=Fixed16, requires_grad=True)
    ((x * x) + x).sum().backward()
    torch.testing.assert_close(x.grad.to_float(), torch.tensor([3.0, 5.0], dtype=torch.float64))
    print("Fixed16 validation passed")


if __name__ == '__main__':
    demo()
    validate()
