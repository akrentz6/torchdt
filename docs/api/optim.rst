Optimizers and schedulers
=========================

All optimizers take a concrete dtype class first, followed by parameters or parameter groups. Use encoded parameters of that class on a single device. Learning rates and other numerical hyperparameters are converted to the format.

.. autoclass:: torchdt.optim.SGD
   :members: step
   :undoc-members:

   Gradient descent with optional momentum, dampening, weight decay, Nesterov updates, and maximization. ``lr`` defaults to 0.001; momentum defaults to zero.

.. autoclass:: torchdt.optim.Adam
   :members: step
   :undoc-members:

   Adam with first and second moments, optional AMSGrad, weight decay, and maximization. ``betas`` controls the moments; ``eps`` must remain positive after encoding. Weight decay participates in the gradient update.

.. autoclass:: torchdt.optim.Madam
   :members: step
   :undoc-members:

   Multiplicative updates using a running squared-gradient estimate. ``beta`` controls that estimate; ``p_scale`` scales the initial parameter RMS bound; ``g_bound`` clips the normalized gradient. ``use_pow=True`` selects an exponential multiplier; the default uses a linearized multiplier ``1 + delta``. Zero-valued parameters cannot be moved away from zero by a purely multiplicative update.

.. autoclass:: torchdt.optim.DTOptimizer
   :members: zero_grad, convert_params, validate_param, encoded_step
   :undoc-members:

   Base for encoded optimizers. ``zero_grad(set_to_none=True)`` clears gradients; ``False`` zeroes existing gradients. ``convert_params`` and ``validate_param`` operate on named hyperparameters in each parameter group.

For all ``step(closure=None)`` methods, an optional closure is called once and its return value is returned. The step executes under ``torch.no_grad()``; closures that calculate gradients must explicitly use ``torch.enable_grad()``.

Learning-rate schedules
-----------------------

Step schedules after optimizer updates. ``last_epoch=-1`` starts a new schedule. Use ``get_last_lr()`` to inspect current encoded learning rates.

.. autoclass:: torchdt.optim.lr_scheduler.StepLR

   Multiply the learning rate by ``gamma`` every ``step_size`` steps.

.. autoclass:: torchdt.optim.lr_scheduler.MultiStepLR

   Multiply by ``gamma`` at each milestone; repeated milestones multiply again.

.. autoclass:: torchdt.optim.lr_scheduler.LinearLR

   Interpolate a multiplier from ``start_factor`` to ``end_factor`` over ``total_iters`` steps.

.. autoclass:: torchdt.optim.lr_scheduler.CosineAnnealingLR

   Cosine schedule with period parameter ``T_max`` and minimum ``eta_min``.

.. autoclass:: torchdt.optim.lr_scheduler.SequentialLR

   Switch between supplied schedulers at the specified milestones.

.. autoclass:: torchdt.optim.lr_scheduler.ReduceLROnPlateau
   :members: step
   :undoc-members:

   Reduce by ``factor`` when a metric stops improving for ``patience`` intervals. Pass a scalar metric to ``step(metrics)``. ``mode`` selects minimization or maximization; ``threshold``/``threshold_mode`` define improvement; ``cooldown`` delays subsequent reductions; ``min_lr`` bounds the rate and ``eps`` suppresses negligible updates.
