Extending TorchDT
=================

To build a new number format, start with :doc:`custom-datatypes`. This page covers adding or overriding differentiable operations once you have a dtype.

There are three layers: PyTorch function registration, autograd adapters, and encoded primitives. Most users only need the first layer's normal PyTorch entry points. Backend and format authors work with the other two.

A differentiable operation
--------------------------

Inside ``DTFunction``, tensors are integer encodings. Use ``ops.mul`` and ``ops.add`` so arithmetic follows the active format. Ordinary ``x * x`` on these integers would multiply bit patterns.

.. literalinclude:: ../examples/custom_function.py
   :language: python

Run ``python examples/custom_function.py`` to check both the forward path and gradient. ``setup_context`` saves encoded inputs and ``backward`` returns one gradient per input. The adapter supplies the active operation table.

Registration
------------

``DType.register_func(torch_function, cast=(...))`` registers a wrapper for a PyTorch entry point. ``cast`` names or indexes arguments that should be converted to the dispatched dtype. Use a concrete class instead of ``DType`` to limit an override to that class, and ``backend=`` for a backend-specific wrapper.

``MyDType.register_op("mul")`` registers an encoded primitive. Its default signature is ``implementation(ops, x, y)``; return an integer tensor containing valid codes. Supported primitive names and signatures appear in :doc:`api/extensions`.

A new format subclasses ``DType`` with a supported ``bitwidth`` and implements conversion and the primitives required by its intended operations. Study the LNS or Posit implementation for encoding boundaries and special values.

Validate conversion, special values, forward arithmetic, broadcasting, and gradients independently before training. Use :func:`torchdt.support_matrix` to inspect registrations and compare results against a suitable numerical reference.
