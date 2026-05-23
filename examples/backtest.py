#!/usr/bin/env python3
"""Backtest example: run kalix over a JSON-lines file for a full audit trail.

Usage:
    python3 examples/backtest.py <input.jsonl>

The input file should contain one JSON object per line, e.g.:
    {"t": 0.0, "dt": 1.0, "z": [10.0]}
    {"t": 1.0, "dt": 1.0, "z": [11.0]}
    {"t": 2.0, "dt": 1.0, "z": null}

Requires the ``kalix`` binary to be installed on PATH.
"""

import json
import sys
import tempfile

from pykalix import KalixConfig, KalixFilter

# Build a simple trend-following config in Python (no TOML file needed)
CONFIG = KalixConfig(
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


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else None

    if input_path is None:
        # Create a small input file for the demo
        fd, input_path = tempfile.mkstemp(suffix=".jsonl", prefix="demo_")
        with open(fd, "w") as f:
            for i, z in enumerate([10.0, 10.3, 10.1, 10.4, 10.2, 10.5]):
                f.write(json.dumps({"t": i, "dt": 1.0, "z": [z]}) + "\n")
        own_temp = True
    else:
        own_temp = False

    try:
        with KalixFilter(config=CONFIG, mode="backtest") as f:
            print(f"Filter: {f.ready.filter_name} ({f.ready.variant})")
            print(f"H matrix: {f.ready.H}")
            print()

            for msg in f.stream_file(input_path):
                if msg.get("event") == "summary":
                    print("--- Summary ---")
                    print(f"Steps:           {msg['steps']}")
                    print(f"Predict-only:    {msg['predict_only_steps']}")
                    print(f"Skipped:         {msg['skipped_steps']}")
                    print(f"Final pos:       {msg['final_x'].get('pos', '?'):.3f}")
                    print(f"Final vel:       {msg['final_x'].get('vel', '?'):.3f}")
                    print(f"Final p_diag:    {msg['final_p_diag']}")
                    break

                step = msg["step"]
                update = msg.get("update", {})
                residual = update.get("residual", {}).get("z", "—")
                pos = update.get("x", {}).get("pos", "—")
                vel = update.get("x", {}).get("vel", "—")

                z_str = residual if isinstance(residual, str) else f"{residual:+.4f}"
                p_str = pos if isinstance(pos, str) else f"{pos:.4f}"
                v_str = vel if isinstance(vel, str) else f"{vel:.4f}"
                print(f"Step {step:>3}  z_residual={z_str}  pos={p_str}  vel={v_str}")
    finally:
        if own_temp:
            import os

            os.unlink(input_path)


if __name__ == "__main__":
    main()
