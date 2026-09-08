Examples
========

These CPU examples use the Python backend and require no downloaded data. Run them from the repository root after installation:

.. code-block:: console

   python examples/arithmetic.py
   python examples/train_regression.py
   python examples/batching.py

Each script includes a small numerical or structural check. Outputs are approximate and may vary with the format, backend, and PyTorch version.

Arithmetic and quantization
---------------------------

Compare decoded values against the original float tensor, then calculate a gradient. Keeping the reference values makes quantization error visible.

.. literalinclude:: ../examples/arithmetic.py
   :language: python

Training a small model
----------------------

The parameter, inputs, targets, loss, and optimizer updates use LNS16. The model fits a single weight to ``y = 2x``; the loss should fall substantially. The ``dtype=LNS16`` layer argument creates encoded parameters directly. Create the optimizer afterward.

.. literalinclude:: ../examples/train_regression.py
   :language: python

Batching inputs and labels
--------------------------

Concrete dtype classes register a collator with PyTorch's default DataLoader collation. Custom inputs are stacked; integer labels remain ordinary tensors.

.. literalinclude:: ../examples/batching.py
   :language: python

Download the scripts: :download:`arithmetic.py <../examples/arithmetic.py>`, :download:`train_regression.py <../examples/train_regression.py>`, and :download:`batching.py <../examples/batching.py>`.
