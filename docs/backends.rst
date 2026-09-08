Backends and validation
=======================

Python is the default backend. Backend selection is per dtype and device type; enabling a CUDA backend does not select it for CPU tensors. Accelerated backends fall back to registered Python implementations for missing operations. That fallback may still contain floating-point conversion steps.

Inspect availability
--------------------

.. code-block:: python

   from torchdt import support_matrix

   status = support_matrix()["LNS16"]["cpu"]
   print(status["backend"])
   print(status["operations"]["matmul"])

The report describes registrations, not hardware availability or successful execution for every shape and option. See :doc:`compatibility`.

C++ on CPU
----------

The optional extension is built by default during installation and needs a C++17 compiler and PyTorch headers. To attempt a build from a checkout with PyTorch already installed:

.. code-block:: console

   python -m pip install 'setuptools>=69' wheel ninja
   python -m pip install --no-build-isolation -e .

Ensure ``TORCHDT_NO_CPP`` is unset. Enable the supplied LNS16 backend explicitly:

.. code-block:: python

   from torchdt.lns import LNS16

   LNS16.set_prec(10)
   LNS16.enable_cpp_backend()

The current C++ LNS16 implementation hard-codes precision 10. Calling Python's ``set_prec`` does not change that native precision. Set Python precision to 10 before enabling C++ so fallback operations use the same encoding, and create all tensors afterward. The native special-value and overflow handling also differs from Python; validate it separately before relying on it in an experiment. An unavailable extension raises ``ImportError``.

Triton on CUDA
--------------

This path requires a working CUDA PyTorch installation, a CUDA device, and Triton. Installing TorchDT alone does not enable Triton.

.. code-block:: python

   import torch
   from torchdt.lns import LNS16

   LNS16.set_prec(7)
   LNS16.enable_triton()
   x = torch.tensor([1.0, 2.0], dtype=LNS16, device="cuda")
   print(x * x)

LNS32, LNS64, and the posit classes also expose ``enable_triton()``. ``LNS16.enable_triton(accumulator=True)`` (or ``"lns32"``) selects the LNS32 accumulator path for kernels using accumulator operations. ``accumulator="lpvip"`` selects the alternative LPVIP addition path; ``False`` uses native LNS16 accumulation. Configure LNS32 precision first when using an accumulator, and compare results with the native mode.

Validate a backend
------------------

.. code-block:: python

   from torchdt.testing import validate_triton_dtype

   report = validate_triton_dtype(LNS16, operations=("elementwise", "matmul"))
   print(report)
   report.raise_for_failures()

This executes CUDA tests and compares decoded results against references. The default tolerance is ``rtol=0.08, atol=0.04``; select tolerances suitable for your experiment. The helper enables Triton by default if it is not already active. Pass ``enable=False`` to preserve a backend configuration you have already enabled, such as an accumulator mode. See :doc:`api/backends` for custom cases and report fields.

Autotuning
----------

Policies in ``torchdt.triton`` select, replace, or exclude configurations by kernel name. Set a policy before enabling the backend, and enable it again after changes. An empty candidate set is an error.

.. code-block:: python

   from torchdt.triton import get_autotune_configs, select_autotune_config

   candidates = get_autotune_configs("matmul")
   select_autotune_config("matmul", candidates[0])
   LNS16.enable_triton()

Selecting the first candidate demonstrates the API; it is not a performance recommendation. Benchmark your actual shapes and separate initial compilation and autotuning from steady-state execution time.
