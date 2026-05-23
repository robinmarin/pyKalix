# pyKalix — Python wrapper for Kalix

[Kalix](https://crates.io/crates/kalix) is a declarative Kalman filter library written
in Rust. You write the physics as symbolic expressions, and it automatically derives the
F (transition) and H (observation) matrices, detects linearity, and selects the
appropriate filter (standard KF or EKF).

`pykalix` is a thin Python wrapper that spawns the `kalix` binary as a subprocess and
communicates via JSON over stdin/stdout — the same bridge documented in the kalix
README.

Platform-specific wheels include a pre-compiled `kalix` binary.  No Rust toolchain
required.

## Installation

```bash
pip install pykalix
```

The package auto-discovers the `kalix` binary: checks the bundled version in the wheel,
then the `KALIX_BINARY` environment variable, then `PATH`.

If a pre-built wheel isn't available for your platform, install kalix manually:

```bash
cargo install kalix
```

## Quick start

```python
from pykalix import KalixFilter

# Live mode — low-latency streaming, minimal output
with KalixFilter(config="configs/trend_no_accel.toml", mode="live") as f:
    # Read ready event
    print(f"Ready: {f.ready.filter_name} ({f.ready.variant})")

    # Normal observation
    result = f.step(t=1000.0, dt=1.0, z=[10.3])
    print(f"pos={result.x['pos']:.3f}, vel={result.x['vel']:.3f}")

    # Predict-only (sensor dropout)
    result = f.predict_only(t=1001.0, dt=1.0)
    print(f"predict-only: pos={result.x['pos']:.3f}")

# Backtest mode — full audit trail
with KalixFilter(config="configs/trend_no_accel.toml", mode="backtest") as f:
    results = []
    for msg in f.stream_file("prices.jsonl"):
        if msg.get("event") == "summary":
            print(f"Done: {msg['steps']} steps")
            break
        results.append(msg)
```

## API

### `KalixFilter`

```python
KalixFilter(
    config: str,                  # Path to TOML config file, or KalixConfig
    mode: str = "live",           # "live" or "backtest"
    on_error: str = "skip",       # "skip" or "halt"
    binary: str | None = None,    # auto-discover (bundled > env > PATH)
)
```

Methods:
- `step(t, dt, z) -> LiveResult` — predict + update
- `predict_only(t, dt) -> LiveResult` — predict only (no measurement)
- `stream_file(input_path) -> Iterator[dict]` — stream results lazily
- `close()` — shut down the subprocess
- Context manager support (`with KalixFilter(...) as f:`)

### `KalixConfig` (Python-native config builder)

```python
from pykalix import KalixConfig

config = KalixConfig(
    name="my_filter",
    state_variables=["pos", "vel"],
    dynamics={"pos": "pos + vel*dt", "vel": "vel"},
    observation_variables=["z"],
    observation_expressions=["pos"],
    process_noise=[[0.01, 0], [0, 0.01]],
    measurement_noise=[[1.0]],
    initial_state=[0.0, 0.0],
    initial_covariance=[[1, 0], [0, 1]],
)

# Write to file
config.to_file("my_config.toml")

# Pass directly to KalixFilter
with KalixFilter.from_config(config, mode="live") as f:
    result = f.step(t=0.0, dt=1.0, z=[10.0])
```

## How it works

Platform-specific wheels include a compiled `kalix` binary (Rust) inside the package.
`pykalix` spawns it as a child process and talks to it over pipes:

```
Python --[JSON lines]--> kalix stdin
Python <--[JSON lines]--  kalix stdout
                     kalix stderr --> logged via Python's logging
```

The subprocess is fully managed: started on first interaction, cleaned up on `close()`
or context-manager exit. A ready event is read automatically before any data is exchanged.

Binary discovery order:
1. User-supplied `binary=` parameter
2. `KALIX_BINARY` environment variable
3. Bundled binary in the wheel
4. `kalix` on `PATH`

## Requirements

- Python >= 3.10
- Pre-built wheels available for macOS, Linux, and Windows.
  If your platform isn't covered: `cargo install kalix`

## License

MIT — see [LICENSE](LICENSE).
