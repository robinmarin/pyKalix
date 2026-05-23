"""Python-native TOML config builder for kalix.

Use :class:`KalixConfig` to build a valid kalix TOML config from Python
objects instead of writing TOML by hand.  The resulting config can be
written to a temp file and passed to :class:`KalixFilter`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

__all__ = ["KalixConfig"]


@dataclass
class KalixConfig:
    """Declarative config for a kalix filter, serialisable to TOML.

    Attributes:
        name: Filter name (written to ``[filter].name``).
        description: Optional description (``[filter].description``).
        state_variables: Ordered list of state variable names.
        dynamics: Mapping of state variable → evolution expression.
        observation_variables: Ordered list of observation variable names.
        observation_expressions: Ordered list of observation expressions
            (one per observation variable, must not reference ``dt``).
        process_noise: Process noise covariance ``Q`` as row-major ``n×n``.
        measurement_noise: Measurement noise covariance ``R`` as ``m×m``.
        initial_state: Initial state vector ``x0`` (length ``n``).
        initial_covariance: Initial covariance ``P0`` as ``n×n``.
    """

    name: str
    state_variables: list[str]
    dynamics: dict[str, str]
    observation_variables: list[str]
    observation_expressions: list[str]
    process_noise: list[list[float]]
    measurement_noise: list[list[float]]
    initial_state: list[float]
    initial_covariance: list[list[float]]
    description: str = ""

    # ── serialisation ──────────────────────────────────────────────────

    def to_toml(self) -> str:
        """Return the config as a TOML string (requires tomli_w)."""
        try:
            import tomli_w  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "tomli_w is required for to_toml(). Install with: pip install tomli-w"
            ) from None

        data: dict[str, Any] = {
            "filter": {
                "name": self.name,
                "description": self.description,
            },
            "state": {"variables": list(self.state_variables)},
            "dynamics": dict(self.dynamics),
            "observation": {
                "variables": list(self.observation_variables),
                "expressions": list(self.observation_expressions),
            },
            "noise": {
                "process": self.process_noise,
                "measurement": self.measurement_noise,
            },
            "initial": {
                "state": self.initial_state,
                "covariance": self.initial_covariance,
            },
        }
        return tomli_w.dumps(data)  # type: ignore[no-any-return]

    def to_file(self, path: str | Path) -> None:
        """Write the config to a ``.toml`` file."""
        with open(path, "w") as f:
            f.write(self._to_toml_builtin())

    def to_tempfile(self, directory: str | None = None) -> str:
        """Write to a temporary file and return its path.

        The caller is responsible for cleaning up the file.
        """
        fd, path = tempfile.mkstemp(
            suffix=".toml", prefix="pykalix_", dir=directory, text=True
        )
        with os.fdopen(fd, "w") as f:
            f.write(self._to_toml_builtin())
        return path

    # ── helpers ────────────────────────────────────────────────────────

    def _to_toml_builtin(self) -> str:
        """Render TOML using only stdlib (no tomli_w dependency).

        This is a minimal formatter sufficient for kalix configs.
        tomli_w produces prettier output but requires an extra install.
        """
        lines: list[str] = []
        a = lines.append

        a("[filter]")
        a(f'name = "{self.name}"')
        if self.description:
            a(f'description = "{self.description}"')
        a("")

        a("[state]")
        a(_toml_str_array("variables", self.state_variables))
        a("")

        a("[dynamics]")
        for var in self.state_variables:
            expr = self.dynamics.get(var, "")
            a(f'{var} = "{expr}"')
        a("")

        a("[observation]")
        a(_toml_str_array("variables", self.observation_variables))
        a(_toml_str_array("expressions", self.observation_expressions))
        a("")

        a("[noise]")
        a(_toml_f64_matrix("process", self.process_noise))
        a(_toml_f64_matrix("measurement", self.measurement_noise))
        a("")

        a("[initial]")
        a(_toml_f64_array("state", self.initial_state))
        a(_toml_f64_matrix("covariance", self.initial_covariance))
        a("")

        return "\n".join(lines)

    @classmethod
    def from_toml(cls, s: str) -> KalixConfig:
        """Parse a kalix TOML string into a :class:`KalixConfig`."""
        data = tomllib.loads(s)
        filt = data.get("filter", {})
        state = data.get("state", {})
        dynamics = data.get("dynamics", {})
        obs = data.get("observation", {})
        noise = data.get("noise", {})
        init = data.get("initial", {})

        return cls(
            name=filt.get("name", ""),
            description=filt.get("description", ""),
            state_variables=list(state.get("variables", [])),
            dynamics=dict(dynamics),
            observation_variables=list(obs.get("variables", [])),
            observation_expressions=list(obs.get("expressions", [])),
            process_noise=noise.get("process", []),
            measurement_noise=noise.get("measurement", []),
            initial_state=init.get("state", []),
            initial_covariance=init.get("covariance", []),
        )


# ── TOML formatting helpers (stdlib only) ──────────────────────────────


def _toml_str_array(key: str, values: list[str]) -> str:
    inner = ", ".join(f'"{v}"' for v in values)
    return f"{key} = [{inner}]"


def _toml_f64_array(key: str, values: list[float]) -> str:
    inner = ", ".join(_fmt_f64(v) for v in values)
    return f"{key} = [{inner}]"


def _toml_f64_matrix(key: str, rows: list[list[float]]) -> str:
    row_strs = ", ".join(f"[{', '.join(_fmt_f64(v) for v in row)}]" for row in rows)
    return f"{key} = [{row_strs}]"


def _fmt_f64(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)
