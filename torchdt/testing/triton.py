from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_OPERATIONS",
    "DEFAULT_TEST_CASES",
    "TritonCaseResult",
    "TritonTestCase",
    "TritonValidationReport",
    "validate_triton_dtype",
]


@dataclass(frozen=True)
class TritonTestCase:

    name: str
    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes = tuple(tuple(shape) for shape in self.input_shapes)
        if not self.name:
            raise ValueError("A Triton test case requires a name")
        if not shapes or any(
            not shape
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                for size in shape
            )
            for shape in shapes
        ):
            raise ValueError("input_shapes must contain non-empty positive shapes")
        object.__setattr__(self, "input_shapes", shapes)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


DEFAULT_TEST_CASES = (
    TritonTestCase("elementwise_257", "elementwise", ((257,), (257,))),
    TritonTestCase("sum_batch_80", "sum", ((80, 9),), {"dim": 0}),
    TritonTestCase("sum_tail_257", "sum", ((4, 257),), {"dim": -1}),
    TritonTestCase("mean_tail_129", "mean", ((7, 129),), {"dim": -1}),
    TritonTestCase("max_tail_129", "max", ((7, 129),), {"dim": -1}),
    TritonTestCase("matmul_awkward", "matmul", ((3, 17, 13), (13, 11))),
    TritonTestCase("matmul_batch_80", "matmul", ((80, 5, 7), (7, 4))),
    TritonTestCase(
        "conv2d_awkward",
        "conv2d",
        ((3, 4, 9, 11), (6, 4, 3, 3), (6,)),
        {"padding": 1},
    ),
    TritonTestCase(
        "conv2d_backward_small",
        "conv2d_backward",
        ((2, 3, 7, 9), (4, 3, 3, 3), (4,)),
        {"padding": 1},
    ),
    TritonTestCase(
        "max_pool2d_odd",
        "max_pool2d",
        ((5, 3, 11, 13),),
        {"kernel_size": 3, "stride": 2, "padding": 1},
    ),
    TritonTestCase(
        "max_pool2d_backward_overlap",
        "max_pool2d_backward",
        ((2, 3, 7, 9),),
        {"kernel_size": 3, "stride": 2, "padding": 1},
    ),
    TritonTestCase(
        "adaptive_avg_pool2d_nonsquare",
        "adaptive_avg_pool2d",
        ((4, 5, 11, 13),),
        {"output_size": (4, 6)},
    ),
    TritonTestCase(
        "adaptive_avg_pool2d_backward",
        "adaptive_avg_pool2d_backward",
        ((2, 3, 7, 9),),
        {"output_size": (3, 4)},
    ),
    TritonTestCase(
        "batch_norm_spatial",
        "batch_norm",
        ((5, 4, 5, 7),),
        {"training": True},
    ),
    TritonTestCase(
        "batch_norm_batch_80",
        "batch_norm",
        ((80, 4, 2, 2),),
        {"training": True},
    ),
    TritonTestCase(
        "batch_norm_backward",
        "batch_norm_backward",
        ((3, 4, 5, 7),),
        {"training": True},
    ),
    TritonTestCase("softmax_classes_19", "softmax", ((7, 19),), {"dim": -1}),
    TritonTestCase(
        "nll_loss_batch_80",
        "nll_loss",
        ((80, 11),),
        {"reduction": "mean"},
    ),
    TritonTestCase(
        "nll_loss_backward",
        "nll_loss_backward",
        ((17, 11),),
        {"reduction": "mean"},
    ),
)

# Kept for scripts written against the initial batch-sweep API.
DEFAULT_BATCH_SIZES = (1, 80, 128)
DEFAULT_OPERATIONS = ("sum", "matmul", "batch_norm")
_SUPPORTED_OPERATIONS = frozenset(case.operation for case in DEFAULT_TEST_CASES)


@dataclass(frozen=True)
class TritonCaseResult:
    """Result of one operation-specific workload."""

    case_name: str
    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    passed: bool
    max_abs_error: float
    max_rel_error: float
    detail: str = ""

    @property
    def batch_size(self) -> int:
        return self.input_shapes[0][0]


@dataclass(frozen=True)
class TritonValidationReport:

    dtype_name: str
    device: str
    gpu_name: str
    compute_capability: tuple[int, int]
    torch_version: str
    triton_version: str
    cases: tuple[TritonCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def failures(self) -> tuple[TritonCaseResult, ...]:
        return tuple(case for case in self.cases if not case.passed)

    def raise_for_failures(self) -> None:
        if self.passed:
            return
        details = "; ".join(
            f"{case.case_name}{case.input_shapes}: {case.detail or 'mismatch'} "
            f"(max_abs={case.max_abs_error:.6g}, max_rel={case.max_rel_error:.6g})"
            for case in self.failures
        )
        raise AssertionError(
            f"Triton validation failed for {self.dtype_name}: {details}"
        )

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status}: {self.dtype_name} on {self.gpu_name} ({self.device}); "
            f"{len(self.cases) - len(self.failures)}/{len(self.cases)} cases passed"
        )


def _dtype_pair(
    dtype_cls: type,
    values: torch.Tensor,
    device: torch.device,
    *,
    requires_grad: bool = False,
):
    reference = dtype_cls(values.cpu())
    encoded = reference._int.to(device=device)
    candidate = dtype_cls(encoded, internal=True, requires_grad=requires_grad)
    return reference, candidate


def _random(shape, generator, *, positive=False):
    values = torch.rand(shape, generator=generator)
    return values + 0.125 if positive else values - 0.5


def _gradient_pair(dtype_cls, shape, device, generator):
    reference, candidate = _dtype_pair(
        dtype_cls, _random(shape, generator), device
    )
    return reference.to_float().float(), candidate


def _run_case(
    dtype_cls: type,
    case: TritonTestCase,
    device: torch.device,
    generator: torch.Generator,
):
    operation = case.operation
    shapes = case.input_shapes
    params = dict(case.parameters)

    if operation == "elementwise":
        left, gpu_left = _dtype_pair(dtype_cls, _random(shapes[0], generator), device)
        right, gpu_right = _dtype_pair(dtype_cls, _random(shapes[1], generator), device)
        expected = F.relu(left.to_float() * right.to_float() + left.to_float())
        actual = F.relu(gpu_left * gpu_right + gpu_left)
        return expected, actual

    if operation in ("sum", "mean"):
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator, positive=True), device
        )
        function = torch.sum if operation == "sum" else torch.mean
        return function(reference.to_float(), **params), function(candidate, **params)

    if operation == "max":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device
        )
        expected, _ = torch.max(reference.to_float(), **params)
        actual, _ = torch.max(candidate, **params)
        return expected, actual

    if operation == "matmul":
        left, gpu_left = _dtype_pair(dtype_cls, _random(shapes[0], generator), device)
        right, gpu_right = _dtype_pair(dtype_cls, _random(shapes[1], generator), device)
        return torch.matmul(left.to_float(), right.to_float()), torch.matmul(gpu_left, gpu_right)

    if operation == "conv2d":
        value, gpu_value = _dtype_pair(dtype_cls, _random(shapes[0], generator), device)
        weight, gpu_weight = _dtype_pair(dtype_cls, _random(shapes[1], generator), device)
        bias, gpu_bias = _dtype_pair(dtype_cls, _random(shapes[2], generator), device)
        expected = F.conv2d(value.to_float(), weight.to_float(), bias.to_float(), **params)
        actual = F.conv2d(gpu_value, gpu_weight, gpu_bias, **params)
        return expected, actual

    if operation == "conv2d_backward":
        value, gpu_value = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device, requires_grad=True
        )
        weight, gpu_weight = _dtype_pair(
            dtype_cls, _random(shapes[1], generator), device, requires_grad=True
        )
        bias, gpu_bias = _dtype_pair(
            dtype_cls, _random(shapes[2], generator), device, requires_grad=True
        )
        ref_value = value.to_float().float().detach().requires_grad_()
        ref_weight = weight.to_float().float().detach().requires_grad_()
        ref_bias = bias.to_float().float().detach().requires_grad_()
        expected = F.conv2d(ref_value, ref_weight, ref_bias, **params)
        actual = F.conv2d(gpu_value, gpu_weight, gpu_bias, **params)
        ref_grad, gpu_grad = _gradient_pair(
            dtype_cls, tuple(expected.shape), device, generator
        )
        expected_grads = torch.autograd.grad(
            expected, (ref_value, ref_weight, ref_bias), ref_grad
        )
        actual.backward(gpu_grad)
        actual_grads = (gpu_value.grad, gpu_weight.grad, gpu_bias.grad)
        return expected_grads, actual_grads

    if operation == "max_pool2d":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device
        )
        return F.max_pool2d(reference.to_float(), **params), F.max_pool2d(candidate, **params)

    if operation == "max_pool2d_backward":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device, requires_grad=True
        )
        ref_value = reference.to_float().float().detach().requires_grad_()
        expected = F.max_pool2d(ref_value, **params)
        actual = F.max_pool2d(candidate, **params)
        ref_grad, gpu_grad = _gradient_pair(
            dtype_cls, tuple(expected.shape), device, generator
        )
        expected_grad = torch.autograd.grad(expected, ref_value, ref_grad)[0]
        actual.backward(gpu_grad)
        return expected_grad, candidate.grad

    if operation == "adaptive_avg_pool2d":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator, positive=True), device
        )
        return (
            F.adaptive_avg_pool2d(reference.to_float(), **params),
            F.adaptive_avg_pool2d(candidate, **params),
        )

    if operation == "adaptive_avg_pool2d_backward":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device, requires_grad=True
        )
        ref_value = reference.to_float().float().detach().requires_grad_()
        expected = F.adaptive_avg_pool2d(ref_value, **params)
        actual = F.adaptive_avg_pool2d(candidate, **params)
        ref_grad, gpu_grad = _gradient_pair(
            dtype_cls, tuple(expected.shape), device, generator
        )
        expected_grad = torch.autograd.grad(expected, ref_value, ref_grad)[0]
        actual.backward(gpu_grad)
        return expected_grad, candidate.grad

    if operation == "batch_norm":
        reference, candidate = _dtype_pair(
            dtype_cls,
            torch.randn(shapes[0], generator=generator) * 0.25 + 0.5,
            device,
        )
        channels = shapes[0][1]
        _, gpu_mean = _dtype_pair(dtype_cls, torch.zeros(channels), device)
        _, gpu_var = _dtype_pair(dtype_cls, torch.ones(channels), device)
        expected = F.batch_norm(
            reference.to_float().float(),
            torch.zeros(channels),
            torch.ones(channels),
            momentum=0.1,
            eps=1e-5,
            **params,
        )
        actual = F.batch_norm(
            candidate, gpu_mean, gpu_var, momentum=0.1, eps=1e-5, **params
        )
        return expected, actual

    if operation == "batch_norm_backward":
        reference, candidate = _dtype_pair(
            dtype_cls,
            torch.randn(shapes[0], generator=generator) * 0.25 + 0.5,
            device,
            requires_grad=True,
        )
        channels = shapes[0][1]
        ref_value = reference.to_float().float().detach().requires_grad_()
        _, gpu_mean = _dtype_pair(dtype_cls, torch.zeros(channels), device)
        _, gpu_var = _dtype_pair(dtype_cls, torch.ones(channels), device)
        expected = F.batch_norm(
            ref_value,
            torch.zeros(channels),
            torch.ones(channels),
            momentum=0.1,
            eps=1e-5,
            **params,
        )
        actual = F.batch_norm(
            candidate, gpu_mean, gpu_var, momentum=0.1, eps=1e-5, **params
        )
        ref_grad, gpu_grad = _gradient_pair(
            dtype_cls, tuple(expected.shape), device, generator
        )
        expected_grad = torch.autograd.grad(expected, ref_value, ref_grad)[0]
        actual.backward(gpu_grad)
        return expected_grad, candidate.grad

    if operation == "softmax":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device
        )
        return F.softmax(reference.to_float(), **params), F.softmax(candidate, **params)

    if operation == "nll_loss":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device
        )
        classes = shapes[0][1]
        target = torch.arange(shapes[0][0], dtype=torch.long) % classes
        return (
            F.nll_loss(reference.to_float(), target, **params),
            F.nll_loss(candidate, target.to(device), **params),
        )

    if operation == "nll_loss_backward":
        reference, candidate = _dtype_pair(
            dtype_cls, _random(shapes[0], generator), device, requires_grad=True
        )
        ref_value = reference.to_float().float().detach().requires_grad_()
        classes = shapes[0][1]
        target = torch.arange(shapes[0][0], dtype=torch.long) % classes
        expected = F.nll_loss(ref_value, target, **params)
        actual = F.nll_loss(candidate, target.to(device), **params)
        expected_grad = torch.autograd.grad(expected, ref_value)[0]
        actual.backward()
        return expected_grad, candidate.grad

    raise AssertionError(f"Unhandled operation {operation!r}")


def _compare_case(
    case: TritonTestCase,
    expected,
    actual,
    *,
    rtol: float,
    atol: float,
) -> TritonCaseResult:
    prefix = (case.name, case.operation, case.input_shapes)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    if len(expected_values) != len(actual_values):
        return TritonCaseResult(
            *prefix,
            False,
            float("inf"),
            float("inf"),
            "output count mismatch",
        )

    all_abs_error = []
    all_rel_error = []
    passed = True
    for index, (expected_item, actual_item) in enumerate(
        zip(expected_values, actual_values)
    ):
        expected_float = expected_item.detach().float().cpu()
        actual_float = actual_item.to_float().detach().float().cpu()
        if expected_float.shape != actual_float.shape:
            return TritonCaseResult(
                *prefix,
                False,
                float("inf"),
                float("inf"),
                f"output {index} shape mismatch: expected {tuple(expected_float.shape)}, "
                f"got {tuple(actual_float.shape)}",
            )
        finite = torch.isfinite(expected_float) & torch.isfinite(actual_float)
        same_nonfinite = (~finite) & (expected_float == actual_float)
        if not bool(torch.all(finite | same_nonfinite)):
            return TritonCaseResult(
                *prefix,
                False,
                float("inf"),
                float("inf"),
                f"output {index} has a NaN or infinity mismatch",
            )
        if bool(torch.any(finite)):
            abs_error = torch.abs(actual_float[finite] - expected_float[finite])
            denominator = torch.clamp(torch.abs(expected_float[finite]), min=atol)
            all_abs_error.append(abs_error)
            all_rel_error.append(abs_error / denominator)
            passed = passed and bool(
                torch.all(
                    abs_error <= atol + rtol * torch.abs(expected_float[finite])
                )
            )
    if not all_abs_error:
        return TritonCaseResult(*prefix, True, 0.0, 0.0)

    detail = "" if passed else f"outside rtol={rtol:g}, atol={atol:g}"
    return TritonCaseResult(
        *prefix,
        passed,
        max(float(error.max().item()) for error in all_abs_error),
        max(float(error.max().item()) for error in all_rel_error),
        detail,
    )


def _legacy_cases(batch_sizes, operations):
    result = []
    for batch_size in batch_sizes:
        for operation in operations:
            if operation == "sum":
                shapes = ((batch_size, 8),)
                parameters = {"dim": 0}
            elif operation == "matmul":
                shapes = ((batch_size, 5, 7), (7, 4))
                parameters = {}
            elif operation == "batch_norm":
                shapes = ((batch_size, 4, 2, 2),)
                parameters = {"training": True}
            else:
                raise ValueError(
                    "batch_sizes is supported only for sum, matmul, and batch_norm; "
                    "use TritonTestCase for other operations"
                )
            result.append(
                TritonTestCase(
                    f"{operation}_batch_{batch_size}", operation, shapes, parameters
                )
            )
    return tuple(result)


def validate_triton_dtype(
    dtype_cls: type,
    *,
    cases: Optional[Iterable[TritonTestCase]] = None,
    batch_sizes: Optional[Iterable[int]] = None,
    operations: Optional[Iterable[str]] = None,
    device: Optional[torch.device | str] = None,
    rtol: float = 0.08,
    atol: float = 0.04,
    seed: int = 0,
    enable: bool = True,
    raise_on_failure: bool = False,
) -> TritonValidationReport:
    from torchdt import DType

    if (
        not isinstance(dtype_cls, type)
        or not issubclass(dtype_cls, DType)
        or dtype_cls is DType
    ):
        raise TypeError("dtype_cls must be a concrete torchdt.DType subclass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to validate the Triton backend")
    if cases is not None and (batch_sizes is not None or operations is not None):
        raise ValueError("cases cannot be combined with batch_sizes or operations")

    if cases is not None:
        selected_cases = tuple(cases)
    elif batch_sizes is not None:
        sizes = tuple(batch_sizes)
        invalid = any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in sizes
        )
        if not sizes or invalid:
            raise ValueError("batch_sizes must contain positive integers")
        selected_cases = _legacy_cases(
            sizes, DEFAULT_OPERATIONS if operations is None else tuple(operations)
        )
    elif operations is not None:
        names = tuple(operations)
        unknown = set(names) - _SUPPORTED_OPERATIONS
        if not names or unknown:
            choices = ", ".join(sorted(_SUPPORTED_OPERATIONS))
            raise ValueError(f"operations must be selected from: {choices}")
        selected_cases = tuple(
            case for case in DEFAULT_TEST_CASES if case.operation in names
        )
    else:
        selected_cases = DEFAULT_TEST_CASES

    if not selected_cases or any(
        not isinstance(case, TritonTestCase) for case in selected_cases
    ):
        raise TypeError("cases must contain at least one TritonTestCase")
    unknown = {case.operation for case in selected_cases} - _SUPPORTED_OPERATIONS
    if unknown:
        choices = ", ".join(sorted(_SUPPORTED_OPERATIONS))
        raise ValueError(f"case operations must be selected from: {choices}")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")

    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    if dtype_cls.ops.backend_for_device(device) != "triton" and enable:
        method = getattr(dtype_cls, "enable_triton", None)
        if method is None:
            raise RuntimeError(
                f"{dtype_cls.__name__} has no enable_triton(); register its Triton backend first"
            )
        method()
    if dtype_cls.ops.backend_for_device(device) != "triton":
        raise RuntimeError(f"The Triton backend is not enabled for {dtype_cls.__name__}")

    try:
        import triton

        triton_version = getattr(triton, "__version__", "unknown")
    except ImportError:
        triton_version = "unknown"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    results = []
    for case in selected_cases:
        try:
            expected, actual = _run_case(dtype_cls, case, device, generator)
            result = _compare_case(case, expected, actual, rtol=rtol, atol=atol)
        except Exception as error:
            result = TritonCaseResult(
                case.name,
                case.operation,
                case.input_shapes,
                False,
                float("inf"),
                float("inf"),
                f"{type(error).__name__}: {error}",
            )
        results.append(result)

    report = TritonValidationReport(
        dtype_name=dtype_cls.__name__,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        compute_capability=torch.cuda.get_device_capability(device),
        torch_version=torch.__version__,
        triton_version=triton_version,
        cases=tuple(results),
    )
    if raise_on_failure:
        report.raise_for_failures()
    return report
