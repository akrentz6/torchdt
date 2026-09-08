Create your own datatype
========================

**Your own number format can use the same PyTorch interface as LNS and Posit.**
Define how values are encoded and how arithmetic acts on those encodings; TorchDT connects those operations to its registered PyTorch functions and autograd adapters. You can keep the implementation in your own Python module, without editing TorchDT itself.

This guide builds ``Fixed16``, a small fixed-point format. By the end you can create it with ``torch.tensor(..., dtype=Fixed16)``, calculate a loss, and compute gradients. The complete example runs on CPU with no extra dependencies:

.. code-block:: console

   python examples/custom_dtype.py

Download :download:`custom_dtype.py <../examples/custom_dtype.py>` to use as a starting point. This is a reference implementation for learning the interface; expand its operation coverage and validate it for your intended workload.

1. Specify the number format
----------------------------

Decide the storage width, encoding, rounding rule, overflow behavior, and special values before writing arithmetic. For this example:

.. list-table:: Fixed16
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Policy
   * - Encoding
     - Signed int16 code; decoded value is ``code / 256``.
   * - Resolution and range
     - Step size ``1 / 256``; values from ``-128`` to ``127.99609375``.
   * - Rounding
     - Round to nearest, with ties to even during conversion and multiplication.
   * - Overflow
     - Saturate at the smallest or largest code; do not wrap.
   * - Special values
     - Code zero represents zero. NaN and infinities are rejected on input.

Write down the policy for every operation you add. For example, division needs a decision about zero denominators. A format that saturates after each addition can produce different reduction results when the addition order changes.

2. Define the class and conversion boundary
-------------------------------------------

Subclass :class:`torchdt.DType` with ``bitwidth=``. Supported storage widths are 8, 16, 32, and 64; support for particular operations and devices still needs validation. TorchDT supplies the corresponding ``int_dtype``, ``float_dtype``, and a per-class ``ops`` table. For 8 bits, the integer carrier is ``torch.uint8``; this example uses the signed ``torch.int16`` carrier selected by ``bitwidth=16``.

The floating-point carrier holds **encoded bits**, not decoded numbers. ``conversion_dtype`` separately specifies the floating-point precision used when importing ordinary values; choose it explicitly for your format.

Register these conversion primitives:

* ``from_float(ops, values)`` returns integer codes with the input shape and device, stored in ``ops.dtype.int_dtype``.
* ``to_float(ops, codes)`` returns an ordinary tensor of decoded values on the same device.

Do not return a ``Fixed16`` instance from a primitive. The surrounding adapters
handle wrapping; primitive inputs and outputs are ordinary tensors.

3. Register arithmetic on encoded values
----------------------------------------

``@Fixed16.register_op("add")`` installs a Python primitive for this format. The first argument, ``ops``, provides the active operation table and the class as ``ops.dtype``. Operands are integer encodings, and the result must also be an integer encoding. Preserve broadcasting behavior and avoid mutating inputs.

For Fixed16, addition adds codes; multiplication must divide the code product by the scale. Widen intermediates before arithmetic, apply rounding and saturation, then convert back to the storage type.

Here is the complete format implementation:

.. literalinclude:: ../examples/custom_dtype.py
   :language: python
   :start-after: # format-start
   :end-before: # format-end

The multiplication uses float64 to express rescaling and ties-to-even rounding clearly. Every product of two int16 codes fits exactly in float64.

Keep the class and decorators at module scope. Importing that module defines the class and registers its operations. Class creation also registers default DataLoader collation and, when available, PyTorch's serialization safe globals. There is no additional central dtype registry to edit.

4. Use existing PyTorch functions and autograd
----------------------------------------------

The four registrations above are enough for this calculation:

.. literalinclude:: ../examples/custom_dtype.py
   :language: python
   :start-after: # demo-start
   :end-before: # demo-end

Call ``demo()`` after defining the format, or run the full script. It prints a
loss of ``5.25`` and gradients ``[-2, 1, 4]``.

.. list-table:: What TorchDT supplies
   :header-rows: 1
   :widths: 30 70

   * - User call
     - Implementation path
   * - ``torch.tensor(..., dtype=Fixed16)``
     - The registered factory encodes values with ``from_float`` and wraps them.
   * - ``x * x``
     - The existing multiplication wrapper and autograd adapter call ``ops.mul``.
   * - ``.sum()``
     - The shared Python reduction repeatedly calls ``ops.add``.
   * - ``loss.backward()``
     - Existing derivative rules use the format's operations; leaf gradient
       accumulation uses its addition.
   * - ``x.to_float()``
     - The format's ``to_float`` decodes values for inspection.

You do not need a new ``register_func`` wrapper or ``DTFunction`` for an existing supported operation just because you introduce a dtype.

The inherited derivative rules apply familiar mathematical derivatives using custom arithmetic. They do **not** differentiate the discrete rounding or saturation map. If your experiment requires different gradient rules, implement a suitable autograd adapter and a format-specific function override; see :doc:`extending` for the process.

5. Validate before expanding coverage
-------------------------------------

The example's ``validate()`` checks rounding ties, negative values, saturation, non-finite input rejection, broadcasting, gradient shapes, and accumulation from multiple graph branches. Run it whenever you change the format.

For your own implementation, also check empty tensors, reduction dimensions, special values, and values around each encoding boundary. Compare against an independent numerical reference using tolerances derived from your format. Check gradients against your chosen derivative policy. If you implement a new operation, add a test for it.

Use :func:`torchdt.support_matrix` to inspect registrations:

.. code-block:: python

   from torchdt import support_matrix

   status = support_matrix()["Fixed16"]["cpu"]
   print(status["operations"]["add"])  # python
   print(status["operations"]["div"])  # unsupported in this example

The class appears after its module is imported. A shared operation can appear registered while still depending on primitives you have not implemented.

6. Grow the format for your workload
------------------------------------

Add the primitives your model needs, following the signatures in :class:`torchdt.ops.OpsBase`. For example, the shared ``mean`` implementation also needs division, and an optimizer may need comparisons, subtraction, square roots, or other operations. Trace those dependencies before assuming that a layer or optimizer works with the minimal example.

Choose the extension point that matches the change:

* **A missing primitive:** register an implementation with ``MyDType.register_op("name")``. Names must already exist in ``OpsBase``.
* **Different behavior for a PyTorch function:** use ``MyDType.register_func(torch_function, cast=(...))`` and an appropriate ``DTFunction`` if it needs autograd. See :doc:`extending`.
* **An accelerated backend:** first establish a Python reference, then provide backend-specific implementations. :doc:`api/extensions` documents the C++ and Triton registration interfaces; :doc:`backends` covers selection and validation.

Keep encoding settings fixed while tensors are alive. If you add a mutable precision setting, existing codes are not automatically re-encoded, and cached scalar encodings and backend configurations must be refreshed. Prefer separate concrete classes when you need multiple configurations simultaneously.

Keep reusable classes in an importable module and record their encoding settings with checkpoints. Import that module before restoring saved tensors, then test the save/load path with the same format configuration.
