"""Unit tests that do not require the kalix binary.

These test the Python-side logic: config serialisation, types, etc.
"""

import dataclasses
import os
import tempfile
from pathlib import Path

import pytest

from pykalix import KalixConfig, LiveResult, ReadyEvent


class TestKalixConfig:
    """Tests for the KalixConfig builder."""

    def _make_basic_config(self) -> KalixConfig:
        return KalixConfig(
            name="test_filter",
            state_variables=["pos", "vel"],
            dynamics={"pos": "pos + vel*dt", "vel": "vel"},
            observation_variables=["z"],
            observation_expressions=["pos"],
            process_noise=[[0.01, 0.0], [0.0, 0.01]],
            measurement_noise=[[1.0]],
            initial_state=[0.0, 0.0],
            initial_covariance=[[1.0, 0.0], [0.0, 1.0]],
        )

    def test_to_file_creates_valid_toml(self) -> None:
        """Built-in TOML formatter produces valid kalix config."""
        cfg = self._make_basic_config()
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            cfg.to_file(f.name)
            path = f.name

        try:
            content = Path(path).read_text()
            # Check key sections are present
            assert "[filter]" in content
            assert 'name = "test_filter"' in content
            assert "[state]" in content
            assert 'variables = ["pos", "vel"]' in content
            assert "[dynamics]" in content
            assert 'pos = "pos + vel*dt"' in content
            assert 'vel = "vel"' in content
            assert "[observation]" in content
            assert "[noise]" in content
            assert "process = [" in content
            assert "[initial]" in content
        finally:
            os.unlink(path)

    def test_to_tempfile_writes_and_returns_path(self) -> None:
        """to_tempfile() returns a valid path and writes the config."""
        cfg = self._make_basic_config()
        path = cfg.to_tempfile()
        try:
            assert os.path.isfile(path)
            content = Path(path).read_text()
            assert "[filter]" in content
        finally:
            os.unlink(path)

    def test_from_toml_round_trip(self) -> None:
        """Config parsed back from TOML matches the original."""
        cfg = self._make_basic_config()
        path = cfg.to_tempfile()
        try:
            toml_str = Path(path).read_text()
            parsed = KalixConfig.from_toml(toml_str)
            assert parsed.name == "test_filter"
            assert parsed.state_variables == ["pos", "vel"]
            assert parsed.dynamics == {"pos": "pos + vel*dt", "vel": "vel"}
            assert parsed.observation_variables == ["z"]
            assert parsed.observation_expressions == ["pos"]
            assert parsed.process_noise == [[0.01, 0.0], [0.0, 0.01]]
            assert parsed.measurement_noise == [[1.0]]
            assert parsed.initial_state == [0.0, 0.0]
            assert parsed.initial_covariance == [[1.0, 0.0], [0.0, 1.0]]
        finally:
            os.unlink(path)

    def test_config_with_description(self) -> None:
        """Description field is serialised when non-empty."""
        cfg = KalixConfig(
            name="desc_test",
            state_variables=["x"],
            dynamics={"x": "x"},
            observation_variables=["z"],
            observation_expressions=["x"],
            process_noise=[[0.1]],
            measurement_noise=[[1.0]],
            initial_state=[0.0],
            initial_covariance=[[1.0]],
            description="A test filter",
        )
        path = cfg.to_tempfile()
        try:
            content = Path(path).read_text()
            assert 'description = "A test filter"' in content
        finally:
            os.unlink(path)

    def test_matrix_formatting_is_valid_toml(self) -> None:
        """Matrix/array formatting matches what kalix expects."""
        cfg = KalixConfig(
            name="mat_test",
            state_variables=["a", "b", "c"],
            dynamics={"a": "a", "b": "b", "c": "c"},
            observation_variables=["z1", "z2"],
            observation_expressions=["a", "b"],
            process_noise=[
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            measurement_noise=[[1.0, 0.0], [0.0, 1.0]],
            initial_state=[0.0, 0.0, 0.0],
            initial_covariance=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        path = cfg.to_tempfile()
        try:
            content = Path(path).read_text()
            # kalix uses tomllib for parsing, verify round-trip
            parsed = KalixConfig.from_toml(content)
            assert parsed.state_variables == ["a", "b", "c"]
            assert len(parsed.process_noise) == 3
            assert len(parsed.process_noise[0]) == 3
        finally:
            os.unlink(path)


class TestTypes:
    """Tests for the type/dataclass definitions."""

    def test_ready_event_from_dict(self) -> None:
        """ReadyEvent.from_dict parses a real kalix ready event."""
        raw = {
            "event": "ready",
            "filter": "trend_no_accel",
            "variant": "linear",
            "mode": "live",
            "state_variables": ["pos", "vel"],
            "observation_variables": ["z"],
            "F": [[1.0, 1.0], [0.0, 1.0]],
            "H": [[1.0, 0.0]],
        }
        ready = ReadyEvent.from_dict(raw)
        assert ready.filter_name == "trend_no_accel"
        assert ready.variant == "linear"
        assert ready.mode == "live"
        assert ready.state_variables == ["pos", "vel"]
        assert ready.observation_variables == ["z"]
        assert ready.F == [[1.0, 1.0], [0.0, 1.0]]
        assert ready.H == [[1.0, 0.0]]

    def test_ready_event_ekf_no_f_matrix(self) -> None:
        """EKF ready event has F=None."""
        raw = {
            "event": "ready",
            "filter": "pendulum",
            "variant": "ekf",
            "mode": "live",
            "state_variables": ["theta", "omega"],
            "observation_variables": ["z"],
            "H": [[1.0, 0.0]],
        }
        ready = ReadyEvent.from_dict(raw)
        assert ready.F is None

    def test_live_result_immutable(self) -> None:
        """LiveResult is frozen (immutable)."""
        result = LiveResult(
            t=1000.0, predict_only=False, x={"pos": 10.0}, p_diag=[0.5, 0.2]
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.t = 2000.0  # type: ignore[misc]
