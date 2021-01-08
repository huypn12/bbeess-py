from typing import Any

import stormpy


def build_model() -> Any:
    builder = stormpy.SparseMatrixBuilder(
        rows=0,
        columns=0,
        entries=0,
        force_dimensions=False,
        has_custom_row_grouping=False,
    )
    builder.add_next_value(row=0, column=1, value=0.5)
    builder.add_next_value(0, 2, 0.5)
    builder.add_next_value(1, 3, 0.5)
    builder.add_next_value(1, 4, 0.5)
    builder.add_next_value(2, 5, 0.5)
    builder.add_next_value(2, 6, 0.5)
    builder.add_next_value(3, 7, 0.5)
    builder.add_next_value(3, 1, 0.5)
    builder.add_next_value(4, 8, 0.5)
    builder.add_next_value(4, 9, 0.5)
    builder.add_next_value(5, 10, 0.5)
    builder.add_next_value(5, 11, 0.5)
    builder.add_next_value(6, 2, 0.5)
    builder.add_next_value(6, 12, 0.5)
    for s in range(7, 13):
        builder.add_next_value(s, s, 1)

    return builder.build()
