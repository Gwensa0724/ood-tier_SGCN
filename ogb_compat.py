"""Compatibility helpers for OGB metadata loading."""

from __future__ import annotations

from typing import Any

import pandas as pd

_PATCHED = False
_ORIG_READ_CSV = None


def patch_master_csv_read() -> None:
    """Keep optional OGB master.csv fields as empty strings instead of NaN."""
    global _PATCHED, _ORIG_READ_CSV

    if _PATCHED:
        return

    _PATCHED = True
    _ORIG_READ_CSV = pd.read_csv

    def _read_csv(*args: Any, **kwargs: Any):
        path = args[0] if args else kwargs.get("filepath_or_buffer")
        path_str = ""
        if path is not None:
            try:
                path_str = str(path).replace("\\", "/")
            except Exception:
                path_str = ""

        if path_str.endswith("ogb/nodeproppred/master.csv"):
            kwargs.setdefault("keep_default_na", False)

        return _ORIG_READ_CSV(*args, **kwargs)

    pd.read_csv = _read_csv
