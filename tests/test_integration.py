"""Integration tests requiring the kalix binary.

These tests spawn the actual kalix process.  Skip them if kalix is not
installed on PATH.
"""

import json
import os
import tempfile

import pytest

from pykalix import KalixConfig, KalixFilter, LiveResult, find_kalix

# Check if kalix binary is available (bundled, PATH, or KALIX_BINARY env)
_kalix_binary = find_kalix(os.environ.get("KALIX_BINARY"))

_HAS_KALIX = _kalix_binary is not None

pytestmark = pytest.mark.skipif(not _HAS_KALIX, reason="kalix binary not found")


# ── Test configs ────────────────────────────────────────────────────────


def _trend_no_accel_config() -> KalixConfig:
    return KalixConfig(
        name="trend_no_accel",
        state_variables=["pos", "vel"],
        dynamics={"pos": "pos + vel*dt", "vel": "vel"},
        observation_variables=["z"],
        observation_expressions=["pos"],
        process_noise=[[0.01, 0.0], [0.0, 0.01]],
        measurement_noise=[[1.0]],
        initial_state=[0.0, 0.0],
        initial_covariance=[[1.0, 0.0], [0.0, 1.0]],
    )


class TestLiveMode:
    """Tests for live mode (default, low-latency streaming)."""

    def test_ready_event_live(self) -> None:
        """Ready event is read and parsed correctly."""
        cfg = _trend_no_accel_config()
        with KalixFilter(config=cfg, mode="live", binary=_kalix_binary) as f:
            ready = f.ready
            assert ready.filter_name == "trend_no_accel"
            assert ready.variant == "linear"
            assert ready.mode == "live"
            assert ready.state_variables == ["pos", "vel"]
            assert ready.observation_variables == ["z"]
            assert ready.F is not None
            assert len(ready.F) == 2
            assert ready.H is not None

    def test_step_returns_live_result(self) -> None:
        """A normal step returns a LiveResult with the right shape."""
        cfg = _trend_no_accel_config()
        with KalixFilter(config=cfg, mode="live", binary=_kalix_binary) as f:
            result = f.step(t=0.0, dt=1.0, z=[10.0])
            assert result.t == 0.0
            assert not result.predict_only
            assert "pos" in result.x
            assert "vel" in result.x
            assert len(result.p_diag) == 2
            # State and covariance should be finite
            assert all(isinstance(v, float) for v in result.x.values())
            assert all(v == v for v in result.x.values())  # not NaN
            assert all(v == v for v in result.p_diag)  # not NaN

    def test_predict_only(self) -> None:
        """Predict-only step returns predict_only=True."""
        cfg = _trend_no_accel_config()
        with KalixFilter(config=cfg, mode="live", binary=_kalix_binary) as f:
            # Do a normal step first to get the filter going
            f.step(t=0.0, dt=1.0, z=[10.0])
            # Then predict-only
            result = f.predict_only(t=1.0, dt=1.0)
            assert result.t == 1.0
            assert result.predict_only
            assert "pos" in result.x
            assert "vel" in result.x

    def test_multiple_steps_converge(self) -> None:
        """After many steps with constant z=10.0, pos converges near 10."""
        cfg = _trend_no_accel_config()
        result: object = None
        with KalixFilter(config=cfg, mode="live", binary=_kalix_binary) as f:
            for i in range(50):
                result = f.step(t=float(i), dt=1.0, z=[10.0])
            assert isinstance(result, LiveResult)  # type narrowing
            assert 9.0 < result.x["pos"] < 11.0
            assert abs(result.x["vel"]) < 5.0

    def test_context_manager_cleanup(self) -> None:
        """Context manager exits cleanly without errors."""
        cfg = _trend_no_accel_config()
        with KalixFilter(config=cfg, mode="live", binary=_kalix_binary) as f:
            assert f.ready is not None
        # After exit, process should be closed
        # No exception = success


class TestBacktestMode:
    """Tests for backtest mode (full audit trail)."""

    def _write_input_file(self, lines: list[dict]) -> str:
        """Write a JSON-lines input file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="pykalix_test_")
        with os.fdopen(fd, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    def test_stream_file_ready_and_summary(self) -> None:
        """stream_file yields a summary event at the end."""
        cfg = _trend_no_accel_config()
        input_path = self._write_input_file(
            [
                {"t": 0.0, "dt": 1.0, "z": [10.0]},
                {"t": 1.0, "dt": 1.0, "z": [11.0]},
                {"t": 2.0, "dt": 1.0, "z": [12.0]},
            ]
        )
        try:
            with KalixFilter(config=cfg, mode="backtest", binary=_kalix_binary) as f:
                messages = list(f.stream_file(input_path))
                assert f.ready.filter_name == "trend_no_accel"

                # Last message should be summary
                assert messages[-1]["event"] == "summary"
                summary = messages[-1]
                assert summary["steps"] == 3
                assert "final_x" in summary
                assert "final_p_diag" in summary
        finally:
            os.unlink(input_path)

    def test_stream_file_step_structure(self) -> None:
        """Each backtest step has predict and update sections."""
        cfg = _trend_no_accel_config()
        input_path = self._write_input_file(
            [
                {"t": 0.0, "dt": 1.0, "z": [10.0]},
            ]
        )
        try:
            with KalixFilter(config=cfg, mode="backtest", binary=_kalix_binary) as f:
                messages = list(f.stream_file(input_path))
                # First non-ready message is step 1
                step_msg = messages[0]
                assert step_msg["step"] == 1
                assert "predict" in step_msg
                assert "update" in step_msg
                # Named state fields
                assert "pos" in step_msg["update"]["x"]
                assert "vel" in step_msg["update"]["x"]
                # Named residual
                assert "z" in step_msg["update"]["residual"]
                # Full P matrix
                pred_p = step_msg["predict"]["P"]
                assert len(pred_p) == 2
                assert len(pred_p[0]) == 2
        finally:
            os.unlink(input_path)

    def test_predict_only_in_backtest(self) -> None:
        """Predict-only steps omit the update section in backtest."""
        cfg = _trend_no_accel_config()
        input_path = self._write_input_file(
            [
                {"t": 0.0, "dt": 1.0, "z": [10.0]},
                {"t": 1.0, "dt": 1.0, "z": None},  # predict-only
            ]
        )
        try:
            with KalixFilter(config=cfg, mode="backtest", binary=_kalix_binary) as f:
                messages = list(f.stream_file(input_path))
                # First message has update
                assert "update" in messages[0]
                # Second message is predict-only
                assert "update" not in messages[1]
                assert messages[1].get("predict_only") is True
                # Summary should count correctly
                summary = messages[-1]
                assert summary["steps"] == 1
                assert summary["predict_only_steps"] == 1
                assert summary["skipped_steps"] == 0
        finally:
            os.unlink(input_path)

    def test_from_config_shortcut(self) -> None:
        """KalixFilter.from_config() works as a convenience constructor."""
        cfg = _trend_no_accel_config()
        with KalixFilter.from_config(cfg, mode="live", binary=_kalix_binary) as f:
            assert f.ready.filter_name == "trend_no_accel"
            result = f.step(t=0.0, dt=1.0, z=[10.0])
            assert not result.predict_only


class TestConfigFile:
    """Tests using a TOML file path instead of KalixConfig."""

    def test_file_path_works(self) -> None:
        """Passing a TOML file path works directly."""
        cfg = _trend_no_accel_config()
        path = cfg.to_tempfile()
        try:
            with KalixFilter(config=path, mode="live", binary=_kalix_binary) as f:
                assert f.ready.filter_name == "trend_no_accel"
                result = f.step(t=0.0, dt=1.0, z=[10.0])
                assert "pos" in result.x
        finally:
            os.unlink(path)
