# インフラ構成の概要

本プロジェクトのインフラは、AWSのGPUインスタンスの供給状況に応じて、いつでも別のインスタンスへ引っ越しができる「ステートレスな開発環境」を Docker を用いて実現しています。

## 🐳 コンテナ設計方針

環境のポータビリティを担保するため、ホストOS側には最小限の依存（NVIDIA Driver、Docker、AWS CLI）のみを要求し、それ以外のすべてのセットアップをコンテナ起動時に動的に処理します。

### 🚨 依存衝突の完全回避（開発ルール）
Isaac Lab のデフォルトの全部入りインストール（`--install`）を実行すると、以下の拡張機能（extras）間で深刻なパッケージの奪い合い（依存衝突）が発生し、ビルドが無限ループまたはクラッシュします。

| 拡張機能名 | 衝突を起こす原因パッケージ | 競合する要求バージョン | 本プロジェクトでの扱い |
| :--- | :--- | :--- | :--- |
| **isaaclab_rl** | `psutil` | `psutil < 6` | **必須** (強化学習用) |
| **isaaclab_mimic** | `psutil` | `psutil >= 7` | **禁止** (不使用) |
| **isaaclab_visualizers**| `websockets` / `psutil` | `websockets < 17` / `psutil >= 7` | **禁止** (ヘッドレスのため不要) |
| **isaaclab_teleop** | `websockets` | `websockets >= 14` | **禁止** (自動学習のため不要) |

このため、`scripts/container_entrypoint.sh` ではこれらを排除し、`omni.isaac.lab`、`omni.isaac.lab_assets`、`omni.isaac.lab_rl` の**最小セットのみをピンポイントでインストール**する構成をとっています。これにより、環境は `torch 2.10.0+cu128` レールに固定され、安定動作します。