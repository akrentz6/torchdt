Training and data
=================

Parameters and gradients
------------------------

Pass the custom class to compatible layer constructors to create encoded parameters directly, as in :doc:`examples`:

.. code-block:: python

   import torch
   from torchdt.lns import LNS16

   layer = torch.nn.Linear(3, 2, dtype=LNS16)
   output = layer(torch.tensor([[1.0, 2.0, 3.0]], dtype=LNS16))

Layers such as ``Conv2d`` and ``BatchNorm2d`` also accept ``dtype=LNS16``. When building a model, forward the dtype argument to its layer constructors.

Use ``torchdt.optim.SGD``, ``Adam``, or ``Madam`` with the dtype as the first argument. Parameters should have that dtype and must all use one device. Optimizer hyperparameters are encoded too: a very small epsilon may round to zero, and a beta close to one may round to one. Choose representable values when experimenting with a different format configuration.

Call ``loss.backward()`` on the custom loss. Gradients use the custom format, including accumulation at graph leaves. Use ``loss.item()`` or ``loss.to_float()`` for logging after computing the loss, without substituting the decoded value into the training graph.

Schedulers
----------

Use schedulers from ``torchdt.optim.lr_scheduler`` so learning rates are
updated as encoded values. Call ``scheduler.step()`` after optimizer updates.
For ``ReduceLROnPlateau``, pass the validation metric instead:

.. code-block:: python

   from torchdt.optim.lr_scheduler import ReduceLROnPlateau

   # optimizer is an existing TorchDT optimizer; validation_loss is a scalar.
   scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
   # At the end of each validation interval:
   # scheduler.step(validation_loss.item())

Images and labels
-----------------

``torchdt.transforms.ToDType`` optionally uses torchvision to convert PIL images or NumPy images to scaled, channel-first tensors and wrap them. Install ``torchvision`` separately to use it. Existing tensor inputs bypass image scaling and layout conversion: prepare their shape and range yourself. Integer tensor inputs remain unwrapped by default, which is useful for labels. ``wrap_all=True`` wraps them too, without scaling them.

``DTypeNormalize`` applies ``(input - mean) / std`` in the custom format. It accepts a float or tuple for each statistic, with one value or one per channel, and inputs shaped ``(..., C, H, W)``. Put its statistics on the same device as the inputs.

.. code-block:: python

   from torchdt.transforms import ToDType, DTypeNormalize

   convert = ToDType(LNS16)
   normalize = DTypeNormalize(LNS16, mean=(0.5,), std=(0.5,))
   # image = normalize(convert(pil_image))

Keep class indices as ``torch.long`` tensors for classification losses and embedding inputs, and masks as ordinary boolean tensors. See :doc:`compatibility` for operation-specific limits.

Checkpoints
-----------

Save model and optimizer state along with the dtype name and format settings. Recreate the same encoded model and optimizer and restore format settings before loading their state dictionaries. A state dictionary alone does not record the process-wide ``prec`` or ``es`` configuration.
