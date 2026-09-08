Tensors and formats
===================

.. autoclass:: torchdt.DType(data, *, internal=False, device=None, requires_grad=None, memory_format=torch.preserve_format)
   :members: backward, grad, requires_grad_, register_hook, copy_, enable_cpp_backend, register_func, register_op
   :undoc-members:

Construction
------------

Concrete classes share this constructor:

``data`` is a tensor, Python scalar, sequence, or another custom tensor. ``internal=False`` encodes numerical values; ``True`` interprets input as already encoded bits and is reserved for implementations. ``device`` selects the destination for ordinary inputs. Prefer constructing directly on the intended device; use ``x.to(device=...)`` for later moves.

``requires_grad=None`` inherits from tensor input; pass a boolean to set it explicitly. ``memory_format`` defaults to ``torch.preserve_format``. Wrapping an instance of the same class with unchanged options can return that same object. Construct a concrete subclass rather than ``DType`` itself.

.. automethod:: torchdt.DType.to_float

   Return an ordinary tensor of decoded numerical values. This is an inspection and export boundary; decoding integer codes does not preserve autograd.

LNS classes
-----------

.. autoclass:: torchdt.lns.LNS16
   :members: set_prec, enable_triton
   :undoc-members:

.. autoclass:: torchdt.lns.LNS32
   :members: set_prec, enable_triton
   :undoc-members:

.. autoclass:: torchdt.lns.LNS64
   :members: set_prec, enable_triton
   :undoc-members:

For all LNS classes, ``set_prec(prec, table=False, table_device=None, filestem="tab")`` changes the process-wide logarithmic spacing. ``table`` selects cached lookup tables, ``table_device`` places those tables, and ``filestem`` controls their filename prefix. See :doc:`../formats` for ranges and configuration lifetime, and :doc:`../backends` for accumulator modes.

Posit classes
-------------

.. autoclass:: torchdt.posit.Posit16

.. automethod:: torchdt.posit.Posit16.set_es

.. automethod:: torchdt.posit.Posit16.enable_triton

.. autoclass:: torchdt.posit.Posit32

.. automethod:: torchdt.posit.Posit32.set_es

.. automethod:: torchdt.posit.Posit32.enable_triton

.. autoclass:: torchdt.posit.Posit64

.. automethod:: torchdt.posit.Posit64.set_es

.. automethod:: torchdt.posit.Posit64.enable_triton

``set_es(es)`` accepts an integer from zero to ``bitwidth - 2`` and changes interpretation for the whole class. All three ``enable_triton()`` methods register the CUDA backend; they require Triton.

Support report
--------------

.. autofunction:: torchdt.support_matrix

   Returns ``{class_name: {device_type: {backend, operations, torch_functions}}}``. See :doc:`../compatibility` for status meanings and the distinction between registration and executable support.
