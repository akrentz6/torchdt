PyTorch compatibility
=====================

Use the usual PyTorch functions with custom tensor arguments. TorchDT routes registered functions to implementations for the active format and device. An unregistered function raises ``NotImplementedError``. A registered function may still reject a particular argument or require an unavailable primitive.

Common entry points
-------------------

.. list-table:: Families with registered implementations
   :header-rows: 1
   :widths: 25 75

   * - Family
     - Examples
   * - Arithmetic and reductions
     - ``add``, ``sub``, ``mul``, ``div``, ``matmul``, ``sum``, ``mean``, ``var``
   * - Shapes and indexing
     - ``reshape``, ``permute``, ``stack``, ``cat``, ``gather``, ``index_add``
   * - Activations
     - ``relu``, ``sigmoid``, ``softmax``, ``gelu``, ``silu``
   * - Layers
     - ``linear``, ``conv2d``, pooling, normalization, ``embedding``
   * - Losses
     - ``mse_loss``, ``l1_loss``, ``cross_entropy``, binary cross entropy
   * - Attention
     - ``scaled_dot_product_attention``, ``multi_head_attention_forward``

The :doc:`api/functions` reference lists registered entry points and wrapper signatures directly from the source. Those signatures describe TorchDT's accepted arguments, which may be narrower than PyTorch's API.

Inspect the full registration report
------------------------------------

.. code-block:: python

   from torchdt import support_matrix

   report = support_matrix()
   cpu = report["LNS16"]["cpu"]
   for name, status in cpu["operations"].items():
       if status == "unsupported":
           print(name)
   print(sorted(cpu["torch_functions"]))

``native`` means an active non-Python backend registers the entry; ``python`` means the Python backend supplies it; ``python_fallback`` means an accelerated backend falls back to Python; ``unsupported`` means no primitive implementation is registered. ``torch_functions`` lists registered functions only, so missing function names do not appear as ``unsupported`` entries. The report covers CPU and CUDA dispatch and does not probe either device.

Limits to account for
---------------------

* Supported factories accept ``dtype=LNS16`` without an existing custom tensor. Examples include ``tensor``, ``zeros``, ``ones``, ``empty``, ``full``, ``rand``, ``randn``,
  ``arange``, and ``linspace``. Compatible layer constructors such as ``Linear``, ``Conv2d``, and ``BatchNorm2d`` accept it too. ``Module.to(dtype=...)`` is not currently implemented for custom dtype classes.
* ``x.to(device=...)`` supports device movement, not the complete PyTorch ``Tensor.to`` overload set. Use ``OtherDType(x)`` to change custom format.
* Autograd through in-place indexed assignment is rejected. Use functional indexing operations such as ``index_add`` or ``scatter_add`` where suitable.
* In-place dropout and sparse custom gradients are unsupported. Embedding also rejects ``max_norm`` and ``scale_grad_by_freq``.
* Attention rejects grouped-query attention. Multi-head attention rejects separate Q/K/V projection weights, ``bias_k``/``bias_v``, ``add_zero_attn``, and static key/value tensors.

Do not infer support for ``torch.compile``, distributed training, higher-order gradients, or MPS from ordinary eager CPU execution. Validate the specific workflow before using it in an experiment. Start with a small input, exercise both forward and backward, and compare decoded results with a float baseline.
