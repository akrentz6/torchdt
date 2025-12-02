import torchdt._C as C

def register_cpp_ops(dtype_cls: type, backend: str) -> None:
    bitwidth = dtype_cls.bitwidth

    dtype_cls.register_op("add")(lambda ops, x, y: add_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("sub")(lambda ops, x, y: sub_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("mul")(lambda ops, x, y: mul_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("div")(lambda ops, x, y: div_op(backend, bitwidth, ops, x, y))

def add_op(backend, bitwidth, _, x, y):
    return C.add(backend, bitwidth, x, y)

def sub_op(backend, bitwidth, _, x, y):
    return C.sub(backend, bitwidth, x, y)

def mul_op(backend, bitwidth, _, x, y):
    return C.mul(backend, bitwidth, x, y)

def div_op(backend, bitwidth, _, x, y):
    return C.div(backend, bitwidth, x, y)