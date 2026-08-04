import argparse
import importlib.metadata as metadata
import os
from isaaclab.app import AppLauncher

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

def main():
    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # 環境の生成
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # RSL-RL ランナーの設定と実行
    runner_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    runner_cfg = string_to_callable(runner_cfg_entry_point)()
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
    
    print("🚀 強化学習ポリシーのトレーニングを開始します...")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()