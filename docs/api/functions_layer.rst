Layers and normalization
========================

These functional entry points underlie compatible ``torch.nn`` modules. Construct compatible modules with ``dtype=LNS16`` to create encoded parameters and floating-point buffers directly. Existing modules need explicit conversion. See :doc:`../compatibility` for embedding and dropout restrictions.

.. torchdt-functions:: layer_funcs
