import os

import numpy as np
import torch


def lns_base(prec: int):
    return 2.0 ** (2.0 ** torch.tensor(-prec, dtype=torch.float64))


def validate_precision(prec: int, table: bool = False) -> None:
    if prec < 1 or prec > 50:
        raise ValueError("Precision must be between 1 and 52")
    if table and prec > 20:
        raise ValueError("Table-based LNS only supports precision up to 20")


def sbdb_ideal(base, int_dtype: torch.dtype, z, s):
    power_term = torch.pow(base, z)
    magnitude = torch.abs(1.0 - 2.0 * s + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    return torch.round(log_term).to(int_dtype) << 1


def load_or_create_table(
    *,
    bitwidth: int,
    prec: int,
    int_dtype: torch.dtype,
    base,
    table_device: str = None,
    filestem: str = "tab",
):
    filename = f"{filestem}_{prec}_{bitwidth}.npz"

    if os.path.isfile(filename):
        data = np.load(filename)
        try:
            tab_sbdb = torch.tensor(data["tab_sbdb"], dtype=int_dtype, device=table_device).contiguous()
            tab_ez = torch.tensor(data["tab_ez"], dtype=int_dtype, device=table_device)
        finally:
            data.close()

        return tab_sbdb, tab_ez

    zero = torch.tensor(0, dtype=int_dtype, device=table_device)
    one = torch.tensor(1, dtype=int_dtype, device=table_device)

    tab_ez = sbdb_ideal(base, int_dtype, one, one)

    zrange = torch.arange(tab_ez, 0, dtype=int_dtype, device=table_device)
    sbt = sbdb_ideal(base, int_dtype, zrange, zero)
    dbt = sbdb_ideal(base, int_dtype, zrange, one)
    tab_sbdb = torch.vstack((sbt, dbt)).contiguous()

    np.savez(filename, tab_ez=tab_ez.cpu().numpy(), tab_sbdb=tab_sbdb.cpu().numpy())
    return tab_sbdb, tab_ez


def register_table_add(dtype_cls: type, *, zero, tab_sbdb, tab_ez) -> None:
    @dtype_cls.register_op("add")
    def lns_add_table(ops, x, y):
        max_operand = torch.maximum(x, y)

        z = -torch.abs((x >> 1) - (y >> 1))
        s = (x ^ y) & 1

        table_col = torch.maximum(tab_ez, torch.where(z == 0, -1, z)).long()
        sbdb = tab_sbdb[s.long(), table_col]
        return torch.where(
            x == zero,
            y, torch.where(
                y == zero,
                x, torch.where(
                    x == ops.neg(y),
                    zero, max_operand + sbdb)))
