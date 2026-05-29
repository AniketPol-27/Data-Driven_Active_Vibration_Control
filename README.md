# Data-Driven Active Vibration Control

**Hybrid MPC + Neural Network Controller for a 6-DOF Spatial Platform**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OSQP](https://img.shields.io/badge/Solver-OSQP-4c8cbf)](https://osqp.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue)](https://www.gnu.org/licenses/gpl-3.0)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Repository Structure](#repository-structure)
- [Verification Scenarios](#verification-scenarios)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Notes](#technical-notes)
- [License](#license)

---

## Overview

This project implements a **hybrid Model Predictive Control and Neural Network** controller for active vibration suppression on a **6-DOF spatial platform**, controlling heave, roll, and pitch under adversarial operating conditions including unknown mass variations, wind-like step disturbances, and multivariate sensor noise.

The system achieves **real-time control at 100 Hz** through a dual-speed architecture that separates the computationally expensive MPC solve from fast neural inference. Three interlocking subsystems form the core of the design:

- A **constrained QP-based MPC** (solved via OSQP) for optimal actuator trajectory planning over a finite horizon
- A **HyperResidual Neural Network** that corrects model-plant mismatch without destabilizing the nominal physics model
- A **Joint Extended Kalman Filter (JEKF)** for simultaneous state estimation and online identification of the unknown platform mass

---

## System Architecture

```
+-------------------------------------------------------+
|               SpatialMPCOrchestrator                  |
|                                                       |
|  z_measurement --> JointEKF                           |
|                    Augmented state:                   |
|                    [z, phi, theta, dz, dphi,          |
|                     dtheta, m]                        |
|                    Predict : Newton-Euler Jacobians   |
|                    Update  : full-state sensor fusion |
|                         |                             |
|                         v   x_est, m_est              |
|                    SpatialHybridDynamics              |
|                    +---------------------------+      |
|                    | Newton-Euler  (physics)   |      |
|                    | + HyperResidualNet * a(t) |      |
|                    | RK4 + jacrev / vmap       |      |
|                    +---------------------------+      |
|                         |   A_k, B_k Jacobians        |
|                         v                             |
|                    MIMO_HyperRTIMPC                   |
|                    OSQP  |  N=40  |  +/-300 N limits  |
|                    Sparse KKT  |  warm-start          |
|                         |                             |
+-------------------------|-----------------------------+
                          |  u_opt  [4 actuator forces]
                          v
               +---------------------+
               |  Adversarial Plant  |   drifting mass,
               |   (Ground Truth)    |   stiffness, damping,
               +---------------------+   wind disturbance
```

---

## Core Components

### Newton-Euler 6-DOF Dynamics (State_space.py)

The platform is modeled with 6 states — heave (z), roll (phi), pitch (theta), and their rates — driven by 4 corner actuators. The transformation matrix **T** maps corner-level spring, damper, and actuator forces into a wrench at the center of gravity, yielding a compact MIMO state-space form `x_dot = Ax + Bu`, where A and B are recomputed at each timestep based on the current mass estimate. Numerical integration uses a 4th-order Runge-Kutta scheme.

### Constrained MPC via OSQP (MPC.py)

At each control cycle, the nominal trajectory is linearized around the current state prediction using automatic differentiation (`torch.func.jacrev` + `vmap`), producing batched Jacobians A_k, B_k across the full horizon. These feed into a convex Quadratic Program:

    minimize    sum [ x'Qx + u'Ru ]
    subject to  x_{k+1} = A_k x_k + B_k u_k   (dynamics equality)
                u_min <= u_k <= u_max            (actuator inequality)

The KKT system is assembled in sparse CSC format and solved by OSQP with warm-starting and horizon shifting. If the solver returns an infeasible status, the previous control input is held.

### HyperResidual Neural Network (State_space.py)

A 3-layer MLP (64 neurons, Tanh activations, Xavier init with gain 0.1) takes the concatenated state, control input, and mass estimate as input and outputs 3 acceleration corrections blended into the physics model via a scalar alpha(t):

    f_hybrid(x, u) = f_physics(x, u) + alpha(t) * [ 0_3 ; f_net(x, u, m_hat) ]

The **AlphaScheduler** gates alpha through a sigmoid of the normalized validation loss, preventing the network from contributing until it has demonstrably reduced model error. This preserves physics-based stability guarantees during warm-up.

### Joint Extended Kalman Filter (model.py)

The JEKF augments the 6-DOF state with mass as a 7th hidden variable: `x_aug = [z, phi, theta, dz, dphi, dtheta, m]`. The predict step uses linearized Newton-Euler Jacobians recomputed at the current mass estimate. The update step fuses full 6-channel sensor measurements. Mass is modeled as a slowly varying parameter driven by zero-mean process noise, enabling continuous online identification without a dedicated mass sensor.

### Adversarial Plant (model.py)

The ground-truth plant introduces stochastic parameter drift at every simulation step across mass, stiffness, and damping. Multivariate Gaussian sensor noise is injected on all 6 output channels. A wind-like impulse disturbance is applied to the velocity states at 40% of the scenario duration to stress-test disturbance rejection.

---

## Repository Structure

    Data-Driven_Active_Vibration_Control/
    |
    +-- State_space.py          # HyperResidualNet, AlphaScheduler, SpatialHybridDynamics
    +-- MPC.py                  # MIMO_HyperRTIMPC: QP formulation, KKT assembly, OSQP interface
    +-- model.py                # AdversarialPlant, JointEKF, SpatialMPCOrchestrator,
    |                           #   simulation loop, performance metric computation
    +-- simulation.py           # Entry point: 6-scenario robustness verification suite
    +-- analyze.py              # Post-simulation statistical analysis
    +-- plot.py                 # Trajectory and dashboard visualization
    |
    +-- spatial_mpc_control/    # ROS2 package for real-time hardware deployment
    +-- ros2_ws/                # ROS2 workspace
    +-- spatial_mpc_ndoe.py     # ROS2 node implementation
    |
    +-- requirements.txt
    +-- PAPERS.md               # Reference literature
    +-- ME444 Project PPT Final.pdf

---

## Verification Scenarios

Six independent maneuvers validate full-envelope robustness. All scenarios start from the origin with adversarial plant conditions active throughout.

| Scenario | Target [z, phi, theta] |
|---|---|
| Pure Heave (+Z) | 0.20 m, 0.00 rad, 0.00 rad |
| Pure Roll (+Phi) | 0.00 m, 0.15 rad, 0.00 rad |
| Pure Pitch (-Theta) | 0.00 m, 0.00 rad, -0.15 rad |
| Coupled Heave and Roll | 0.15 m, -0.10 rad, 0.00 rad |
| Coupled Heave and Pitch | 0.15 m, 0.00 rad, 0.10 rad |
| Full Spatial Translation | 0.25 m, 0.10 rad, -0.10 rad |

Results are written to `data/spatial_telemetry.pkl` and printed as a formatted robustness matrix.

---

## Performance Metrics

| Metric | Description |
|---|---|
| ITAE | Integral Time-weighted Absolute Error: penalizes errors that persist late in the maneuver |
| Control TV | Total variation of actuator commands: proxy for actuator wear and high-frequency chatter |
| Max Z Error | Peak absolute heave deviation during the maneuver |
| Settling Time | Time for heave error to remain within 2% of the initial displacement |
| Mass Est. RMSE | JEKF tracking accuracy against the true drifting platform mass |

---

## Installation

    git clone https://github.com/AniketPol-27/Data-Driven_Active_Vibration_Control.git
    cd Data-Driven_Active_Vibration_Control
    pip install -r requirements.txt

Python 3.10 or later is required. The codebase uses `X | Y` union type syntax and `list[float]` generics introduced in Python 3.10.

Dependencies: `torch` `numpy` `scipy` `pandas` `scikit-learn` `matplotlib` `seaborn` `osqp` `tqdm` `tabulate` `torchvision` `requests`

---

## Usage

**Run the full verification suite**

    python simulation.py

Executes all six scenarios, prints the robustness matrix to stdout, and saves trajectory data and metrics to `data/spatial_telemetry.pkl`.

**Analyze and plot results**

    python analyze.py
    python plot.py

**ROS2 deployment**

The `ros2_ws/` workspace and `spatial_mpc_control/` package contain the ROS2 implementation for hardware deployment. Build with `colcon build` inside `ros2_ws/` and launch via `spatial_mpc_ndoe.py`.

---

## Technical Notes

**Why Successive Linearization MPC instead of a global linear model?**
The corner-force-to-CoG mapping makes the platform dynamics configuration-dependent. Linearizing around the current predicted trajectory at each timestep using differentiable RK4 and batch Jacobians keeps the QP convex while preserving accuracy within the local operating region.

**Why augment mass into the Kalman state rather than treat it as a fixed parameter?**
Mass can shift substantially during operation due to payload changes or structural variation. Augmenting it into the JEKF state vector allows the filter to continuously correct the mass estimate from motion data alone, updating the MPC internal model without any scheduled re-identification.

**Why blend neural corrections additively rather than replace the physics model?**
A purely data-driven controller extrapolates poorly outside its training distribution. The additive blending architecture ensures that at alpha = 0 the controller reduces exactly to the Newton-Euler MPC. The network contributes only where it has demonstrated accuracy, gated by the AlphaScheduler.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](./LICENSE) for the full terms.
