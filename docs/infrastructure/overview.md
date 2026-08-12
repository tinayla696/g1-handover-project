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

## 🧾 実行ログ運用（非致命Warningの扱い）

### NICE DCV + NVIDIA GPU GUI再生

WebRTC Clientを使わず、NICE DCVのGUI上でIsaac Simを操作する場合は、SSHトンネルとGDMのXauthorityを使用します。

```bash
ssh -L 8443:localhost:8443 avita-g5.24
```

DCV Clientは `localhost:8443` に接続します。DCV接続後、EC2上で以下を実行します。

```bash
sudo XAUTHORITY=/run/user/127/gdm/Xauthority DISPLAY=:0 xhost +
sudo cp /run/user/127/gdm/Xauthority "$HOME/.Xauthority"
sudo chown "$(id -u):$(id -g)" "$HOME/.Xauthority"
export XAUTHORITY="$HOME/.Xauthority"
export DISPLAY=:0
glxinfo | grep -E "OpenGL vendor|OpenGL renderer"
```

`OpenGL renderer string: NVIDIA A10G/PCIe/SSE2` を確認できたら、policy再生を実行します。

```bash
./scripts/check_dcv_environment.sh
./scripts/run_playback_dcv.sh g1_handover_teacher policy
```

DCV ComposeではXauthorityをコンテナ内の `/root/.Xauthority` としてマウントし、WebRTCポートや `--livestream` は使用しません。

Isaac Sim 6.0.1 の visual check / train 実行時には、終了コードが `0` でも以下の Warning が出る場合があります。これらは本プロジェクトでは**既知のノイズ**として扱います。

| ログ断片 | 主な意味 | 判定 | 対応方針 |
| :--- | :--- | :--- | :--- |
| `OmniHub ... failed to launch ... retry` | OmniHub の補助プロセス再接続 | 非致命 | 終了コードとステップ完走を優先して判定 |
| `failed to open the default display` | コンテナ側でX表示を検証できない | 非致命 | headless/ストリーミング運用では許容 |
| `Enable omni.materialx.libs extension...` | MaterialX拡張の案内 | 非致命 | MaterialXを使わない限り無視可 |
| `TGS solver ... noisy velocities` | 物理ソルバ設定に関する注意 | 注意 | 学習安定性を見て必要時のみ調整 |
| `Seed not set for the environment` | 環境seed未設定 | 改善対象 | `src/train.py` と `src/tasks/visual_check.py` で seed を明示設定 |
| `GLXBadFBConfig` | DCV/X11 がIsaac Sim用のGPU OpenGL FBConfigを提供していない | 致命的 | DCV GL/GPUアクセラレーションをサーバー側で有効化 |

### ✅ 成功判定の基準

以下を満たす場合は、Warning が残っていても実行成功とみなします。

* コンテナの終了コードが `0`
* visual check が指定ステップ（例: `600/600`）まで完走
* `SimulationContext cleared` まで到達

### 🚨 失敗として扱う条件

次のいずれかに該当する場合は、即時にエラー調査を優先してください。

* `There was an error running python` が出る
* コンテナ終了コードが `0` 以外
* 引数エラー（`unrecognized arguments`）や import エラーで途中停止する