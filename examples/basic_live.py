#!/usr/bin/env python3
"""Minimal example: run a linear Kalman filter in live mode.

Requires the ``kalix`` binary to be installed on PATH.
Install it with: ``cargo install kalix``
"""

from pykalix import KalixFilter

# Path to a kalix TOML config file.
# You can find examples in the kalix repository:
#   https://github.com/robinmarin/kalix/tree/main/configs
CONFIG_PATH = "configs/trend_no_accel.toml"


def main() -> None:
    with KalixFilter(config=CONFIG_PATH, mode="live") as f:
        # The ready event tells us the filter variant and derived matrices
        print(f"Filter ready: {f.ready.filter_name} ({f.ready.variant})")
        print(f"F matrix: {f.ready.F}")
        print(f"H matrix: {f.ready.H}")
        print()

        # Feed observations one at a time
        measurements = [10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6]
        for i, z in enumerate(measurements):
            result = f.step(t=float(i), dt=1.0, z=[z])
            print(
                f"t={i}  z={z:.1f}  "
                f"pos={result.x.get('pos', 0):.3f}  "
                f"vel={result.x.get('vel', 0):.3f}  "
                f"p_diag={[f'{p:.2f}' for p in result.p_diag]}"
            )

        # Predict-only step (simulate sensor dropout)
        result = f.predict_only(t=float(len(measurements)), dt=1.0)
        print()
        print(
            f"Predict-only: pos={result.x.get('pos', 0):.3f}  "
            f"vel={result.x.get('vel', 0):.3f}"
        )


if __name__ == "__main__":
    main()
