import argparse
import importlib.metadata as metadata
import os
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher
from filelock import FileLock
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

TASK_ID = os.getenv("TASK_ID", "g1_handover_base")
MOTION_DIR = Path(os.getenv("MOTION_DIR", "data/motions/HandOver7"))

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="Train an RL policy for Unitree G1 Handover.")
parser.add_argument("--task", type=str, default=os.getenv("TRAIN_TASK", "G1-Handover-v0"), help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=int(os.getenv("NUM_ENVS", "64")), help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
parser.add_argument("--motion_dir", type=str, default=str(MOTION_DIR), help="HandOver7 motion directory.")
parser.add_argument("--distributed", action="store_true", help="Run one training rank per GPU via torchrun.")
parser.add_argument("--resume_checkpoint", type=str, default=os.getenv("RESUME_CHECKPOINT"), help="Checkpoint to load before training.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

os.environ["MOTION_DIR"] = args_cli.motion_dir


class RslRlRewardCompatWrapper(RslRlVecEnvWrapper):
    """RSL-RLが期待する報酬テンソルへ互換変換する薄いラッパー。"""

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        while isinstance(rew, dict) and len(rew) > 0:
            rew = rew["policy"] if "policy" in rew else next(iter(rew.values()))
        return obs, rew, dones, extras


def main():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Kit's USD stage initialization is not safe when every local rank starts it simultaneously.
    startup_lock = FileLock(os.path.join(tempfile.gettempdir(), "g1_handover_kit_startup.lock"))
    with startup_lock:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from isaaclab.utils.string import string_to_callable
        import tasks

        if args_cli.distributed:
            import torch

            torch.cuda.set_device(app_launcher.device_id)

        env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed + rank
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.device_id}"

        print(
            f"🎬 MOTION_DIR: {args_cli.motion_dir} | rank={rank}/{world_size} | "
            f"device={env_cfg.sim.device} | envs_per_rank={env_cfg.scene.num_envs}",
            flush=True,
        )
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    runner_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    runner_cfg = string_to_callable(runner_cfg_entry_point)()
    if hasattr(runner_cfg, "experiment_name"):
        runner_cfg.experiment_name = TASK_ID
    if hasattr(runner_cfg, "seed"):
        runner_cfg.seed = args_cli.seed + rank
    if args_cli.distributed:
        runner_cfg.device = env_cfg.sim.device
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner_cfg_dict = runner_cfg.to_dict()
    env = RslRlRewardCompatWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        env,
        runner_cfg_dict,
        log_dir=os.path.join(os.path.dirname(__file__), "../logs") if rank == 0 else None,
        device=runner_cfg.device,
    )
    if args_cli.resume_checkpoint:
        checkpoint_path = Path(args_cli.resume_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        print(f"🔁 Loading checkpoint: {checkpoint_path}", flush=True)
        runner.load(str(checkpoint_path))
    
    print(f"🧩 TASK_ID: {TASK_ID}")
    print("🚀 強化学習ポリシーのトレーニングを開始します...")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()