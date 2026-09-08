Backend controls
================

Autotune policy
---------------

Policies are process-wide and keyed by the names in ``AUTOTUNE_KERNELS``. After changes, call the dtype's ``enable_triton()`` again.

.. autodata:: torchdt.triton.AUTOTUNE_KERNELS

.. autoclass:: torchdt.triton.AutotuneConfig

   Kernel keyword arguments and launch settings. ``kwargs`` is copied into a read-only mapping. Warp, stage, and CTA counts must be positive.

.. autofunction:: torchdt.triton.get_autotune_configs

   Return the effective tuple of candidates, including overrides and exclusions.

.. autofunction:: torchdt.triton.set_autotune_configs

   Replace candidates for a kernel with a nonempty iterable. Accepts ``AutotuneConfig``, keyword mappings, or Triton configuration objects. Existing exclusions still apply.

.. autofunction:: torchdt.triton.select_autotune_config

   Select one candidate and clear that kernel's exclusions.

.. autofunction:: torchdt.triton.exclude_autotune_configs

   Append exclusions. A mapping matches a subset of kernel keywords; a complete configuration matches by equality. Removing every candidate is an error when the policy is resolved.

.. autofunction:: torchdt.triton.reset_autotune_configs

   Clear overrides and exclusions for one kernel, or all kernels if omitted.

Validation
----------

.. autofunction:: torchdt.testing.validate_triton_dtype

   Run CUDA validation and return a report. ``cases`` supplies explicit shapes and operation parameters and cannot be combined with ``batch_sizes`` or ``operations``. ``seed`` controls random inputs. ``enable=True`` enables the dtype's default Triton backend if Triton is not already active; use ``False`` to validate an existing setup. ``raise_on_failure=True`` raises when any case fails.

.. autoclass:: torchdt.testing.TritonTestCase

   Named operation, positive input shapes, and operation-specific parameters.

.. autoclass:: torchdt.testing.TritonCaseResult
   :members: batch_size
   :undoc-members:

   Per-case outcome, errors, and diagnostics.

.. autoclass:: torchdt.testing.TritonValidationReport
   :members: passed, failures, raise_for_failures
   :undoc-members:

   Aggregated results. ``raise_for_failures()`` raises if any case failed.

.. autodata:: torchdt.testing.DEFAULT_OPERATIONS
.. autodata:: torchdt.testing.DEFAULT_BATCH_SIZES

``torchdt.testing.DEFAULT_TEST_CASES`` contains the default shape and operation cases, including backward and optimizer checks.
