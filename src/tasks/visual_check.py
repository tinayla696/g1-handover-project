import argparse
import os

import torch
from isaaclab.app import AppLauncher

TASK_ID = os.getenv("TASK_ID", "g1_handover_base")

parser = argparse.ArgumentParser(description="Run a zero-action visual check for Unitree G1 Handover.")
parser.add_argument("--task", type=str, default="G1-Handover-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the check run.")
parser.add_argument("--steps", type=int, default=7200, help="Number of zero-action steps to execute.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import tasks


def main():
    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "experiment_name"):
        env_cfg.experiment_name = TASK_ID

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset(seed=args_cli.seed)

    action_dim = int(env.action_space.shape[-1])
    device = getattr(env, "device", "cpu")
    zero_actions = torch.zeros((1, action_dim), device=device)

    print(f"🧩 TASK_ID: {TASK_ID}")
    print("🔍 Running visual check with zero actions...")
    try:
        for step in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero_actions)
            if (step + 1) % 60 == 0:
                print(f"  step {step + 1}/{args_cli.steps}")
            if terminated.any().item() or truncated.any().item():
                env.reset(seed=args_cli.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()