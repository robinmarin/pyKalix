#!/usr/bin/env python3
"""Config builder example: build a kalix TOML config from Python dataclasses.

The config can be saved to a file or passed directly to KalixFilter.
"""

from pykalix import KalixConfig

# ── Constant-velocity model ─────────────────────────────────────────────
# This is a standard 2-state linear KF: position and velocity.
# Dynamics:  pos_next = pos + vel*dt
#            vel_next = vel
# Observation: we measure position directly.

constant_velocity = KalixConfig(
    name="constant_velocity",
    description="Simple 2-state constant-velocity model",
    state_variables=["pos", "vel"],
    dynamics={
        "pos": "pos + vel*dt",
        "vel": "vel",
    },
    observation_variables=["z"],
    observation_expressions=["pos"],
    process_noise=[
        [0.1, 0.0],
        [0.0, 0.1],
    ],
    measurement_noise=[
        [5.0],
    ],
    initial_state=[0.0, 0.0],
    initial_covariance=[
        [10.0, 0.0],
        [0.0, 10.0],
    ],
)

# ── Save to TOML file ───────────────────────────────────────────────────
constant_velocity.to_file("constant_velocity.toml")
print("Saved config to constant_velocity.toml")

# ── Read it back ────────────────────────────────────────────────────────
with open("constant_velocity.toml") as f:
    parsed = KalixConfig.from_toml(f.read())
print(f"Parsed filter: {parsed.name}")
print(f"State: {parsed.state_variables}")
print(f"Dynamics: {parsed.dynamics}")
print(f"Q matrix: {parsed.process_noise}")
print()

# ── The file can be used directly with KalixFilter ─────────────────────
# Uncomment to run (requires kalix binary installed):
# with KalixFilter(config="constant_velocity.toml", mode="live") as f:
#     print(f"Ready: {f.ready.filter_name} ({f.ready.variant})")
#     result = f.step(t=0.0, dt=1.0, z=[10.0])
#     print(f"pos={result.x['pos']:.3f}, vel={result.x['vel']:.3f}")

# ── Or pass the config object directly (no file needed!) ────────────────
print("Config object can also be passed directly to KalixFilter.from_config()")
print("  KalixFilter.from_config(constant_velocity, mode='live')")
print()

# ── Pendulum (EKF with trig) ────────────────────────────────────────────
pendulum = KalixConfig(
    name="pendulum",
    state_variables=["theta", "omega"],
    dynamics={
        "theta": "theta + omega*dt",
        "omega": "omega - 9.81*sin(theta)*dt",
    },
    observation_variables=["z"],
    observation_expressions=["theta"],
    process_noise=[
        [0.001, 0.0],
        [0.0, 0.001],
    ],
    measurement_noise=[
        [0.1],
    ],
    initial_state=[0.1, 0.0],
    initial_covariance=[
        [0.1, 0.0],
        [0.0, 0.1],
    ],
)

pendulum.to_file("pendulum.toml")
print("Saved pendulum.toml (EKF with sin() dynamics)")
