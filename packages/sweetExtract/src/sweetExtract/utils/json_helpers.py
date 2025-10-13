
from __future__ import annotations
from dataclasses import is_dataclass, asdict
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from uuid import UUID
import math
from collections.abc import Mapping, Sequence


try:
    import numpy as _np  # optional
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def json_default(o):
    # Dates & times -> ISO 8601 strings
    if isinstance(o, (datetime, date, time)):
        return o.isoformat()

    # Common “stringifiable” types
    if isinstance(o, (Path, UUID)):
        return str(o)

    # Numbers that aren't JSON-native
    if isinstance(o, Decimal):
        return float(o)

    # Dataclasses -> plain dicts
    if is_dataclass(o):
        return asdict(o)

    # Numpy scalars -> native Python scalars
    if _HAS_NUMPY and isinstance(o, _np.generic):
        return o.item()

    # Sets/tuples -> lists
    if isinstance(o, (set, tuple)):
        return list(o)

    # Last-resort fallback
    return str(o)


try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

def to_jsonable(x):
    # Fast path for common primitives (handle non-finite floats explicitly)
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (int, str)):
        return x

    # Dates & times
    if isinstance(x, (datetime, date, time)):
        return x.isoformat()

    # Dataclasses
    if is_dataclass(x):
        x = asdict(x)

    # Mappings
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    # Sequences / sets / tuples
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    # Path/UUID/Decimal
    if isinstance(x, (Path, UUID)):
        return str(x)
    if isinstance(x, Decimal):
        f = float(x)
        return f if math.isfinite(f) else None

    # NumPy scalars/arrays
    if _HAS_NUMPY:
        if isinstance(x, _np.ndarray):
            return [to_jsonable(v) for v in x.tolist()]
        if isinstance(x, _np.generic):
            v = x.item()
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return to_jsonable(v)

    # Fallback
    return str(x)