import argparse
import os
from isaaclab.app import AppLauncher

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="Train an RL policy for Unitree G1 Handover.")
parser.add_argument("--task", type=str, default="G1-Handover-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
AppLauncher.add_app_launch_args(parser)
args_cli = parser.parse_args()

# アプリケーションの起動 (Headlessモード等を有効化)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv

# カスタムタスクをインポートしてレジストリに登録させる
import tasks 

def main():
    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 環境の生成
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else "human")
    
    # RSL-RL ランナーの設定と実行
    runner_cfg = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    runner = RslRlOnPolicyRunner(env, runner_cfg, log_dir=os.path.join(os.path.dirname(__file__), "../logs"))
    
    print("🚀 強化学習ポリシーのトレーニングを開始します...")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_with_graceful_stop=True)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()