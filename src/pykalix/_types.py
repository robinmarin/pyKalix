"""Type definitions for pykalix results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "BacktestResult",
    "LiveResult",
    "ReadyEvent",
    "SummaryEvent",
]


@dataclass(frozen=True)
class ReadyEvent:
    """Emitted once on filter startup, before any input is consumed.

    Attributes:
        filter_name: Name from the TOML config ``[filter].name``.
        variant: ``"linear"`` or ``"ekf"``.
        mode: ``"live"`` or ``"backtest"``.
        state_variables: Ordered list of state variable names.
        observation_variables: Ordered list of observation variable names.
        F: Fixed transition matrix for linear filters (``None`` for EKF).
        H: Observation matrix.
    """

    filter_name: str
    variant: str
    mode: str
    state_variables: list[str]
    observation_variables: list[str]
    F: list[list[float]] | None
    H: list[list[float]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReadyEvent:
        """Parse from the raw JSON dict emitted by kalix."""
        return cls(
            filter_name=d["filter"],
            variant=d["variant"],
            mode=d["mode"],
            state_variables=list(d.get("state_variables", [])),
            observation_variables=list(d.get("observation_variables", [])),
            F=d.get("F"),
            H=d.get("H", []),
        )


@dataclass(frozen=True)
class LiveResult:
    """Result of a single step or predict-only in live mode.

    Attributes:
        t: Timestamp echoed from input.
        predict_only: ``True`` if the step was predict-only.
        x: Named state vector as a dict (``{"pos": 10.1, "vel": 0.3, ...}``).
        p_diag: Diagonal of the covariance matrix as a list of floats.
    """

    t: float
    predict_only: bool
    x: dict[str, float]
    p_diag: list[float]


@dataclass(frozen=True)
class _MatrixState:
    """Named state with full covariance matrix (used inside backtest)."""

    x: dict[str, float]
    P: list[list[float]]


@dataclass(frozen=True)
class _BacktestUpdate:
    x: dict[str, float]
    P: list[list[float]]
    residual: dict[str, float]
    kalman_gain: list[list[float]]
    innovation_cov: list[list[float]]


@dataclass(frozen=True)
class BacktestResult:
    """Full per-step result in backtest mode.

    Attributes:
        t: Timestamp echoed from input.
        step: Step counter (1-indexed).
        predict_only: ``True`` if this was a predict-only step.
        predict: Predicted state (before update).
        update: Updated state (``None`` for predict-only steps).
    """

    t: float
    step: int
    predict_only: bool
    predict: _MatrixState
    update: _BacktestUpdate | None = None


@dataclass(frozen=True)
class SummaryEvent:
    """Emitted at end-of-stream in backtest mode.

    Attributes:
        steps: Number of normal steps processed.
        predict_only_steps: Number of predict-only steps.
        skipped_steps: Number of steps skipped due to errors.
        final_x: Final named state vector.
        final_p_diag: Diagonal of the final covariance matrix.
    """

    steps: int
    predict_only_steps: int
    skipped_steps: int
    final_x: dict[str, float]
    final_p_diag: list[float]
