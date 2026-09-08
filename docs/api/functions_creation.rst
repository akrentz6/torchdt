Tensor creation
===============

The factories below accept a custom class through ``dtype=``; for example, ``torch.zeros(3, dtype=LNS16)`` creates an LNS16 tensor without needing an existing custom tensor argument. ``*_like`` calls can infer the class from the input. Random wrappers sample ordinary float32 values and encode them. ``torch.tensor(data, dtype=LNS16)`` copies numerical data into a fresh leaf without retaining input autograd history. A source of the same custom format is copied bit-for-bit; a different custom format is decoded and re-encoded. Pass the custom ``dtype`` explicitly when copying an encoded tensor. ``LNS16(data)`` remains available when constructor reuse or graph preservation is desired.

.. torchdt-functions:: creation_funcs
