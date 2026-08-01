# GitHub Copilot Instructions for g1-handover-project

This repository configures a reinforcement learning environment for the Unitree G1 humanoid robot using Isaac Sim and Isaac Lab. Always adhere to the following contextual constraints and environment definitions when generating code, debugging, or explaining architecture.

## 1. Project Overview & Constraints
*   **Goal:** Train a reinforcement learning policy for the Unitree G1 humanoid robot (29DoF + Dex3 hands) to perform a novelty handover task (e.g., passing a bottle to a human) in simulation.
*   **Portability First:** The environment is built to be 100% portable. It must run seamlessly on any AWS GPU instance type (`g5`, `g6`, or `g7e`) via Docker Compose with dynamic S3 infrastructure mapping.

## 2. Active Software & Network Stack (As of August 2026)
*   **Server Environment:** Isaac Sim `6.0.1` (Docker Container) + Isaac Lab `Core` running on `torch 2.10.0+cu128`.
*   **Client Environment:** Isaac Sim WebRTC Streaming Client `2.0.0` (Local PC).
*   **Network Ports:** Fixed exclusively to:
    *   `TCP 49100` (Signaling)
    *   `UDP 47998` (Livestream)
*   **Data Lifecycle:** Source code is managed in GitHub. Heavy RL logs and checkpoints (`logs/`) are completely excluded from Git via `.gitignore`. They are synced bidirectionally between the host and **AWS S3 (`s3://g1-gr00t-models-380421147972-us-east-1-an/`)** before and after container execution via `run_env.sh`.

## 3. Critical Architectural Rules (Do Not Violate)
*   **No Isaac Lab Extras Installation:** Installing full Isaac Lab extensions induces severe dependency loops. **Never install or rely on `mimic`, `visualizers`, or `teleop` packages.** They create fatal version conflicts between `psutil` (v5 vs v7) and `websockets` (v13 vs v14). Maintain only `core`, `assets`, and `rl` minimal installations in the container.
*   **Isaac Lab 6.x Manager-Based API Compliance:** Older launch arguments (like `--headless`) and environment configurations are deprecated. When writing or modifying environment files (e.g., `src/tasks/g1_handover_env.py`), strictly follow the 6.x `ManagerBasedRLEnvCfg` pattern utilizing `ObservationManagerCfg`, `ActionTermCfg`, and `TerminationTermCfg`.

## 4. Current Reinforcement Learning Bottlenecks
*   **Symptom:** The training workflow successfully completes 1,000 iterations (exit code 0), but the robot suffers from rapid terminal fall rates (`robot_fall=1.0000` constantly) with mean rewards flattening at the minimum floor due to initial random exploration shocks.
*   **Mitigation Strategies:**
    1.  Keep `action_scale` heavily tuned down (around `0.05` to `0.1`) to prevent violent joint initialization jitter.
    2.  Keep the `root_height` termination threshold temporarily lowered (around `0.3m`) during initial boot training to construct viable baseline reward gradients.
    3.  Prioritize Alive rewards (`func="isaaclab.envs.mdp:alive"`) and Root Orientation error penalties (`func="isaaclab.envs.mdp:root_orientation_error"`) to ensure survival up to the maximum timeout before heavily shifting weights to the physical object handover mechanics.