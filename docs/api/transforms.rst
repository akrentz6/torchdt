Data transforms
===============

See :doc:`../training` for image scaling, channel layout, and integer-label behavior. ``ToDType`` requires torchvision at construction time, even when called only with existing tensors. ``DTypeNormalize`` does not require it.

.. autoclass:: torchdt.transforms.ToDType
   :members: __call__
   :undoc-members:

.. autoclass:: torchdt.transforms.DTypeNormalize
   :members: __call__
   :undoc-members:

.. autofunction:: torchdt.transforms.register_collate_dtype_fn

   Register stacking for ``dtype_cls`` in PyTorch's default collate map. Concrete ``DType`` subclasses call this automatically when defined.
