import sys


def native_config(dtype):
    module = sys.modules[dtype.__module__]
    table = module.tab_sbdb
    return dict(
        precision=module.precision,
        base=float(module.base),
        log_base=float(module.base.log()),
        table=None if table is None else table.detach().cpu().contiguous(),
        table_ez=0 if module.tab_ez is None else int(module.tab_ez),
    )


def refresh_backends(dtype):
    dtype.ops.clear_scalar_cache()
    dtype.ops._direct_ops.clear()
    if getattr(dtype.ops, '_cpp_backend_name', None) is not None:
        from torchdt.ops.cpp_ops import register_cpp_ops
        register_cpp_ops(dtype, dtype.ops._cpp_backend_name)
    if 'cuda' in dtype.ops._enabled_backends and dtype.ops._enabled_backends['cuda'] == 'triton':
        mode = getattr(dtype.ops, '_lns_accumulator_mode', None)
        if mode is None:
            dtype.enable_triton()
        else:
            dtype.enable_triton(accumulator=mode)
    # LNS16's optional accumulator depends on LNS32 precision and tables
    if dtype.__name__ == 'LNS32':
        from .lns16 import LNS16
        mode = getattr(LNS16.ops, '_lns_accumulator_mode', False)
        if mode and LNS16.ops._enabled_backends.get('cuda') == 'triton':
            LNS16.enable_triton(accumulator=mode)
