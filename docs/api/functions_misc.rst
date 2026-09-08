Shape, indexing, and movement
=============================

Shape operations preserve encoded values. Keep index tensors as integers and mask tensors as booleans. ``to`` supports device movement only; use a format constructor for conversion and ``to_float()`` for decoding.

.. torchdt-functions:: misc_funcs
