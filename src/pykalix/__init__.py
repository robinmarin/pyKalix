"""pykalix — Python wrapper for the kalix Kalman filter.

Declarative Kalman filtering from symbolic dynamics expressions, callable
directly from Python.  The package spawns the ``kalix`` Rust binary as a
subprocess and communicates via JSON over stdin/stdout.

Quick start::

    from pykalix import KalixFilter

    with KalixFilter(config="configs/trend_no_accel.toml", mode="live") as f:
        print(f"Ready: {f.ready.filter_name} ({f.ready.variant})")
        result = f.step(t=0.0, dt=1.0, z=[10.0])
        print(f"pos={result.x['pos']:.3f}")
"""

from __future__ import annotations

from ._config import KalixConfig
from ._kalix import KalixFilter
from ._types import BacktestResult, LiveResult, ReadyEvent, SummaryEvent

__version__ = "0.2.1"
__all__ = [
    "BacktestResult",
    "KalixConfig",
    "KalixFilter",
    "LiveResult",
    "ReadyEvent",
    "SummaryEvent",
]
