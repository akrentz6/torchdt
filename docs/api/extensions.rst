Extension interfaces
====================

These interfaces operate on encoded integer tensors. See :doc:`../extending`
for a differentiable example before implementing operations.

.. autoclass:: torchdt.autograd.DTFunction

   Define ``forward(ops, ...)``, optionally ``setup_context(ctx, ops, inputs, output)``, and ``backward(ctx, ops, grad_output)``. The adapter unwraps custom tensors into integer codes and wraps outputs. ``apply`` takes positional arguments only. ``output_indices`` can select which tuple outputs are encoded.

.. autoclass:: torchdt.autograd.DTNonDifferentiableFunction

   Adapter for a ``forward(ops, ...)`` operation without gradient tracking. Set ``output_indices`` for mixed encoded and ordinary outputs.

.. autofunction:: torchdt.ops.register_op

   Register a named primitive on one dtype and backend. Normally the callable receives ``ops`` first. With ``direct=True`` it receives just operation arguments. The name must already exist in ``OpsBase``.

.. autofunction:: torchdt.ops.register_base_op

   Register a shared Python primitive for all formats.

.. autofunction:: torchdt.ops.register_cpp_ops
.. autofunction:: torchdt.ops.register_triton_ops
.. autoclass:: torchdt.ops.TritonScalarOps
.. autoclass:: torchdt.ops.TritonAccumulatorOps
.. autofunction:: torchdt.ops.is_triton_available

   Test whether the Triton module is discoverable; does not check CUDA hardware.

.. autofunction:: torchdt.ops.require_triton

   Import and return ``(triton, triton.language)`` or raise ``ImportError``.

Primitive contracts
-------------------

The signatures below describe encoded inputs and outputs. Arithmetic must use format operations, not ordinary arithmetic on integer bit patterns.

.. autoclass:: torchdt.ops.OpsBase
   :members:
   :undoc-members:
