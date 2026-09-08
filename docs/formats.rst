Number formats
=================

A TorchDT tensor stores encoded bits inside a PyTorch tensor subclass. Its
``dtype`` property describes the storage carrier, not the number format.
Use ``type(x)`` to identify the format and ``x.to_float()`` to read its values.

Available formats
-----------------

.. list-table:: Default configuration
   :header-rows: 1

   * - Class
     - Storage bits
     - Default setting
   * - ``torchdt.lns.LNS16``
     - 16
     - ``prec=7``
   * - ``torchdt.lns.LNS32``
     - 32
     - ``prec=23``
   * - ``torchdt.lns.LNS64``
     - 64
     - ``prec=23``
   * - ``torchdt.posit.Posit16``
     - 16
     - ``es=1``
   * - ``torchdt.posit.Posit32``
     - 32
     - ``es=2``
   * - ``torchdt.posit.Posit64``
     - 64
     - ``es=3``

LNS
---

LNS represents a sign and a quantized logarithm. Its base is ``2 ** (2 ** -prec)``. Increasing ``prec`` gives a smaller quantization gap but reduces the available dynamic range at a fixed bitwidth.

.. code-block:: python

   import torch
   from torchdt.lns import LNS16

   LNS16.set_prec(8)
   x = torch.tensor([0.5, 1.0, 2.0], dtype=LNS16)

``set_prec`` accepts values from 1 through 50 in the current implementation. ``table=True`` restricts precision to at most 20 and loads or creates ``{filestem}_{prec}_{bitwidth}.npz``. Tables can be large: begin with analytic mode (the default) and a modest precision. Place tables on the device where they will be used, using ``table_device``.

LNS constructors currently convert ordinary input tensors through float32, including LNS64.

Posits
------

Posits use a variable-length regime, exponent, and fraction. ``es`` controls the maximum number of exponent bits. Valid values are 0 through ``bitwidth - 2``. Posit conversion uses float64.

.. code-block:: python

   import torch
   from torchdt.posit import Posit16

   Posit16.set_es(1)
   x = torch.tensor([0.5, 1.0, 2.0], dtype=Posit16)

The Python implementation performs core posit arithmetic on encoded integers; operations such as ``exp``, ``log``, and ``pow`` use decoded floating-point calculations and re-encode the result. See the implementation through the API reference's source links when an experiment depends on an exact arithmetic path.

Configuration lifetime
----------------------

Settings apply to the whole dtype class in the process. Changing ``prec`` or ``es`` changes how existing encoded tensors are interpreted; it does not re-encode their values. Configure the format **before** creating tensors, parameters, or optimizers, and recreate them when changing settings.

Record format settings alongside checkpoints. Restore those settings before loading or interpreting encoded tensors. If using Triton, configure first, then call ``enable_triton()`` again after a setting or autotune policy changes.
