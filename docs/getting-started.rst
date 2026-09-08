Getting started
===============

Install from PyPI
-----------------

.. code-block:: console

   python -m pip install torchdt

Install from source
-------------------

Use Python 3.11 or newer for the documentation build. The package declares
PyTorch >= 2.0 and NumPy as dependencies; check your actual PyTorch version
with the examples before starting a larger experiment.

.. code-block:: console

   git clone https://github.com/akrentz6/torchdt.git
   cd torchdt
   python -m venv .venv
   source .venv/bin/activate
   TORCHDT_NO_CPP=1 python -m pip install -e .

This installs the Python implementation without compiling the optional C++
extension. In PowerShell, set ``$env:TORCHDT_NO_CPP = "1"`` before running
``python -m pip install -e .``. See :doc:`backends` for acceleration.

Create, calculate, decode
-------------------------

.. code-block:: python

   import torch
   from torchdt.lns import LNS16

   x = torch.tensor([0.5, 1.0, 2.0], dtype=LNS16)
   y = torch.tensor([2.0, 3.0, 4.0], dtype=LNS16)
   result = torch.sum(x * y)
   print(result.item())  # approximately 12
   print(x.to_float())  # ordinary PyTorch tensor of decoded values

Construct custom tensors with ``torch.tensor(values, dtype=LNS16)``, or pass
``dtype=LNS16`` to supported PyTorch factories and layer constructors:

.. code-block:: python

   zeros = torch.zeros(3, dtype=LNS16)
   samples = torch.randn(2, 3, dtype=LNS16)
   layer = torch.nn.Linear(3, 2, dtype=LNS16)
   output = layer(samples)

Note that these classes are not native ``torch.dtype`` objects. ``torch.tensor(data, dtype=LNS16)`` always copies the data and creates a new leaf with no input autograd history. Set ``requires_grad=True`` to track its gradients. In contrast, ``LNS16(data)`` can reuse an existing LNS16 tensor or preserve the graph when encoding a floating-point tensor.

Keep tensors in one custom format within a calculation. To explicitly change format, pass the destination class as ``dtype`` when copying the source tensor.

Compute gradients
-----------------

.. code-block:: python

   x = torch.tensor([1.0, 2.0, 4.0], dtype=LNS16, requires_grad=True)
   loss = (x * x).sum()
   loss.backward()
   print(x.grad)  # [2, 4, 8]

Call ``backward()`` on the custom loss, before decoding it. Quantization affects
both results and gradients, so compare with a floating-point baseline using
appropriate tolerances. Continue with :doc:`examples` for a complete training loop.
