# TorchDT

Experiment with custom number formats in PyTorch. TorchDT provides 16, 32, and 64-bit Logarithmic Number System (LNS) and Posit tensors, with supported PyTorch operations, autograd, and optimizers for training and inference.

```py
import torch
from torchdt.lns import LNS32

x = torch.tensor([1.0, 2.0, 4.0], dtype=LNS32, requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad)  # [2, 4, 8]
```

Custom tensors carry encoded values. Use `.to_float()` to convert to floating point; call `.backward()` on the custom loss itself.

## Install

To install via PyPI:

```sh
python -m pip install torchdt
```

From a source checkout, with PyTorch available for your platform:

```sh
git clone https://github.com/akrentz6/torchdt.git
cd torchdt
python -m pip install .
```

To skip compilation of the optional C++ extension:

```sh
TORCHDT_NO_CPP=1 python -m pip install .
```

The package depends on PyTorch >= 2.0 and NumPy. See the [installation guide](docs/getting-started.rst) and [backend guide](docs/backends.rst) for more detail and optional C++/Triton paths.

## Try it

```sh
python examples/arithmetic.py
python examples/train_regression.py
python examples/batching.py
```

The examples run on the CPU without downloading datasets. The training demo fits a small model with encoded parameters and `torchdt.optim.SGD`.

## Documentation

- [Getting started](docs/getting-started.rst): construction, conversion, and gradients.
- [Examples](docs/examples.rst): arithmetic, training, and DataLoader batching.
- [Formats](docs/formats.rst): precision settings and their lifetime.
- [Training](docs/training.rst): parameters, optimizers, transforms, and checkpoints.
- [Compatibility](docs/compatibility.rst): operation coverage and current limits.
- [API reference](docs/api/index.rst): tensors, optimizers, functions, and backend controls.

Build the Sphinx site locally with Python 3.11+:

```sh
python -m pip install -e '.[docs]'
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`. After editing the docs, rerun the Sphinx
command above to rebuild them.

## Scope

This is an experimental numerical library. Classes such as `LNS16` are tensor subclasses. Create them with `torch.tensor(values, dtype=LNS16)` or pass `dtype=LNS16` to supported PyTorch factories and layer constructors, such as `torch.zeros` and `torch.nn.Linear`. Configure precision or posit exponent settings before creating tensors; settings apply to the whole class. Operation availability and numerical results depend on the format and backend. `torchdt.support_matrix()` reports registrations; test your forward and backward workload before scaling up.

## Create your own datatype

TorchDT also lets you define your own number format. Subclass `DType`, register conversion and arithmetic operations, then use `torch.tensor(..., dtype=YourDType)` with supported PyTorch functions and autograd.

Follow the [custom datatype guide](docs/custom-datatypes.rst) for a complete fixed-point example, the registration process, and validation steps:

```sh
python examples/custom_dtype.py
```

Licensed under the [MIT License](LICENSE).
