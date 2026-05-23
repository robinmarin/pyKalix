"""Main KalixFilter class — the primary API for pykalix."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

from ._config import KalixConfig
from ._process import KalixProcess
from ._types import LiveResult, ReadyEvent

logger = logging.getLogger(__name__)

__all__ = ["KalixFilter"]


class KalixFilter:
    """Python wrapper around the kalix Kalman filter binary.

    Spawns ``kalix`` as a subprocess and communicates via JSON over
    stdin/stdout.  The process is started lazily on the first interaction
    and cleaned up when the context manager exits or :meth:`close` is called.

    Parameters:
        config: Path to a kalix TOML config file, or a :class:`KalixConfig`
            instance (which is automatically serialised to a temp file).
        mode: ``"live"`` (low-latency streaming) or ``"backtest"`` (full
            audit trail with complete covariance matrices).
        on_error: ``"skip"`` (default) to silently skip malformed input
            lines, or ``"halt"`` to exit the process on error.
        binary: Name or path of the ``kalix`` binary.  Defaults to
            ``"kalix"`` (looked up on ``PATH``).

    Example:

        >>> with KalixFilter(config="configs/trend_no_accel.toml") as f:
        ...     print(f.ready.variant)
        ...     result = f.step(t=0.0, dt=1.0, z=[10.0])
        ...     print(f"pos={result.x.get('pos', 0):.3f}")
    """

    def __init__(
        self,
        config: str | KalixConfig,
        mode: str = "live",
        on_error: str = "skip",
        binary: str = "kalix",
    ) -> None:
        if mode not in ("live", "backtest"):
            raise ValueError(f"mode must be 'live' or 'backtest', got '{mode}'")
        if on_error not in ("skip", "halt"):
            raise ValueError(f"on_error must be 'skip' or 'halt', got '{on_error}'")

        self._mode = mode
        self._on_error = on_error
        self._binary = binary
        self._temp_config: str | None = None
        self._config_path: str

        if isinstance(config, KalixConfig):
            self._temp_config = config.to_tempfile()
            self._config_path = self._temp_config
        else:
            self._config_path = config

        self._proc = KalixProcess(
            binary=self._binary,
            config_path=self._config_path,
            mode=self._mode,
            on_error=self._on_error,
        )
        self._ready: ReadyEvent | None = None
        self._started = False

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self) -> KalixFilter:
        self._ensure_started()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the kalix process and clean up temp files."""
        self._proc.close()
        if self._temp_config is not None:
            with suppress(OSError):
                os.unlink(self._temp_config)
            self._temp_config = None
        self._started = False

    @property
    def ready(self) -> ReadyEvent:
        """The ready event emitted by kalix on startup.

        Contains the filter name, variant (linear/ekf), mode, state/obs
        variable names, and the derived F and H matrices.

        Raises:
            RuntimeError: The process is not yet started.
        """
        self._ensure_started()
        assert self._ready is not None  # set by _ensure_started
        return self._ready

    # ── live mode methods ──────────────────────────────────────────────

    def step(self, t: float, dt: float, z: list[float] | None) -> LiveResult:
        """Run a predict+update step in live mode.

        Parameters:
            t: Timestamp (echoed back in output, not used by the filter).
            dt: Timestep.  Must be > 0.
            z: Observation vector.  If ``None``, this is a predict-only step.

        Returns:
            :class:`LiveResult` with named state ``x`` and ``p_diag``.
        """
        self._ensure_started()
        payload: dict[str, Any] = {"t": t, "dt": dt, "z": z}
        self._proc.send_line(json.dumps(payload))
        raw = json.loads(self._proc.read_line())
        return LiveResult(
            t=raw["t"],
            predict_only=raw.get("predict_only", False),
            x=raw["x"],
            p_diag=raw["p_diag"],
        )

    def predict_only(self, t: float, dt: float) -> LiveResult:
        """Predict-only step (no measurement) in live mode."""
        return self.step(t, dt, z=None)

    # ── streaming ─────────────────────────────────────────────────────

    def stream_file(self, input_path: str) -> Iterator[dict[str, Any]]:
        """Run kalix with ``--input`` and iterate over result messages.

        Launches a fresh kalix process with ``--input <input_path>`` and
        yields each output line as a parsed dict.  The ready event is
        consumed internally and exposed via :attr:`ready`.

        Example:

            >>> with KalixFilter(config=..., mode="backtest") as f:
            ...     for msg in f.stream_file("prices.jsonl"):
            ...         if msg.get("event") == "summary":
            ...             print(f"Done: {msg['steps']} steps")
            ...             break
            ...         print(msg["step"], msg["update"]["x"]["pos"])
        """
        # Close the default process and launch a new one with --input.
        self._proc.close()

        cmd = [
            self._binary,
            "--config",
            self._config_path,
            "--mode",
            "backtest",
            "--input",
            input_path,
            "--on-error",
            self._on_error,
        ]
        logger.debug("spawning (input file): %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Drain stderr
        stderr_thread = threading.Thread(
            target=self._proc._drain_stderr,
            args=(proc.stderr,),
            daemon=True,
        )
        stderr_thread.start()

        # Read ready event
        assert proc.stdout is not None
        first_line = proc.stdout.readline()
        if not first_line:
            # Process exited without output — collect stderr for diagnostics
            stderr_thread.join(timeout=2)
            proc.wait(timeout=2)
            raise RuntimeError(
                f"kalix exited with code {proc.returncode} without emitting ready event"
            )
        ready_raw = json.loads(first_line.rstrip("\n"))
        self._ready = ReadyEvent.from_dict(ready_raw)
        self._started = True

        # Yield remaining lines
        try:
            for line in proc.stdout:
                yield json.loads(line.rstrip("\n"))
        finally:
            if proc.stdin is not None:
                with suppress(Exception):
                    proc.stdin.close()
            proc.wait(timeout=5)
            stderr_thread.join(timeout=2)

    def stream_stdin(self) -> Iterator[dict[str, Any]]:
        """Iterate over stdout lines from the current process.

        Use this when you want to manually write lines to stdin between
        iterations (e.g. for backtest mode via pipes).

        Yields:
            Parsed JSON dicts from stdout.  The ready event has already
            been consumed.
        """
        self._ensure_started()
        for line in self._proc.iter_lines():
            yield json.loads(line)

    def write_line(self, line: str) -> None:
        """Write a raw JSON line to the process stdin.

        Useful in combination with :meth:`stream_stdin` for interactive
        backtesting.
        """
        self._ensure_started()
        self._proc.send_line(line)

    # ── internals ──────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        """Lazily start the process and consume the ready event."""
        if self._started:
            return
        self._proc.start()
        ready_raw = json.loads(self._proc.read_line())
        self._ready = ReadyEvent.from_dict(ready_raw)
        self._started = True

    @classmethod
    def from_config(
        cls,
        config: KalixConfig,
        mode: str = "live",
        on_error: str = "skip",
        binary: str = "kalix",
    ) -> KalixFilter:
        """Create a :class:`KalixFilter` from a :class:`KalixConfig` instance.

        The config is written to a temp file automatically.
        """
        return cls(config=config, mode=mode, on_error=on_error, binary=binary)
