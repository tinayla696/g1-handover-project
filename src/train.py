import argparse
import csv
import importlib.metadata as metadata
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

TASK_ID = os.getenv("TASK_ID", "g1_handover_base")
MOTION_DIR = Path(os.getenv("MOTION_DIR", "data/motions/HandOver7"))

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="Train an RL policy for Unitree G1 Handover.")
parser.add_argument("--task", type=str, default="G1-Handover-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# アプリケーションの起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.string import string_to_callable

# カスタムタスクをインポートしてレジストリに登録させる
import tasks 


class RslRlRewardCompatWrapper(RslRlVecEnvWrapper):
    """RSL-RLが期待する報酬テンソルへ互換変換する薄いラッパー。"""

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        while isinstance(rew, dict) and len(rew) > 0:
            rew = rew["policy"] if "policy" in rew else next(iter(rew.values()))
        return obs, rew, dones, extras


def _normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(":", "")
        .replace("/", "")
        .replace(".", "")
    )


def _load_reference_first_joint_pos(motion_dir: Path, joint_names: list[str]) -> np.ndarray | None:
    csv_path = motion_dir / "HandOver7_unitree_g1.csv"
    npz_path = motion_dir / "HandOver7.npz"

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_row = next(reader, None)

        if first_row is None:
            return None

        row = np.asarray(first_row, dtype=np.float32)
        if header:
            header_map = {_normalize_name(col): idx for idx, col in enumerate(header)}
            indices: list[int] = []
            for joint_name in joint_names:
                key = _normalize_name(joint_name)
                if key in header_map:
                    indices.append(header_map[key])
            if len(indices) == len(joint_names):
                return row[indices]
        if row.shape[0] >= len(joint_names):
            return row[: len(joint_names)]
        return None

    if npz_path.exists():
        payload = np.load(npz_path)
        candidate_keys = ["joint_pos", "joint_positions", "qpos", "positions", "motion", "data"]
        arr = None
        for key in candidate_keys:
            if key in payload:
                arr = payload[key]
                break
        if arr is None:
            if len(payload.files) == 1:
                arr = payload[payload.files[0]]
            else:
                return None

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] < len(joint_names):
            return None
        return arr[0, : len(joint_names)]

    return None

def main():
    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    reference_joint_pos = None
    try:
        robot_joint_names = list(env_cfg.scene.robot.joint_names)
        reference_joint_pos = _load_reference_first_joint_pos(MOTION_DIR, robot_joint_names)
        if reference_joint_pos is not None and hasattr(env_cfg.scene.robot, "init_state"):
            if hasattr(env_cfg.scene.robot.init_state, "joint_pos"):
                env_cfg.scene.robot.init_state.joint_pos = tuple(float(v) for v in reference_joint_pos)
            if hasattr(env_cfg.scene.robot.init_state, "joint_vel"):
                env_cfg.scene.robot.init_state.joint_vel = tuple(0.0 for _ in reference_joint_pos)
    except Exception as exc:
        print(f"[WARN] Motion initialization skipped: {exc}")
    
    # 環境の生成
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # RSL-RL ランナーの設定と実行
    runner_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    runner_cfg = string_to_callable(runner_cfg_entry_point)()
    if hasattr(runner_cfg, "experiment_name"):
        runner_cfg.experiment_name = TASK_ID
    if hasattr(runner_cfg, "seed"):
        runner_cfg.seed = args_cli.seed
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner_cfg_dict = runner_cfg.to_dict()
    env = RslRlRewardCompatWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        env,
        runner_cfg_dict,
        log_dir=os.path.join(os.path.dirname(__file__), "../logs"),
        device=runner_cfg.device,
    )
    
    print(f"🧩 TASK_ID: {TASK_ID}")
    if reference_joint_pos is not None:
        print(f"🎬 Motion init loaded from: {MOTION_DIR}")
    print("🚀 強化学習ポリシーのトレーニングを開始します...")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()