TorchDT
=======

TorchDT lets you experiment with custom number formats in PyTorch. Wrap values in a custom tensor class, use supported PyTorch operations, and train with gradients and optimizer updates in that format. By default, we support Logarithmic Number Systems (LNS) and Posit arithmetic, though users can implement their own number formats. 

Start with :doc:`getting-started`, then try the small, runnable :doc:`examples`. The :doc:`api/index` holds signatures and detailed options.

.. important::

   **Create your own datatype.** TorchDT is designed to support your own number formats. Follow :doc:`custom-datatypes` to define an encoding, register arithmetic, and use it with PyTorch and autograd.

Why use it?
-----------

Use TorchDT to study how a numerical format changes rounding, arithmetic, and training. It provides 16, 32, and 64-bit LNS and Posit tensors, a Python implementation, and optional accelerated backends. It is an experimental library: operation coverage and numerical behavior depend on the format and backend.

A first calculation
-------------------

.. code-block:: python

   import torch
   from torchdt.lns import LNS16

   x = torch.tensor([1.0, 2.0, 4.0], dtype=LNS16)
   print(x * x)  # [1., 4., 16.]


.. toctree::
   :maxdepth: 2
   :caption: Learn

   getting-started
   examples
   formats
   training

.. toctree::
   :maxdepth: 2
   :caption: Look up

   compatibility
   backends
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Create your own datatype

   custom-datatypes
   extending

* :ref:`genindex`
* :ref:`search`
