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

WebRTC Clientを使わず、NICE DCVのGUI上でIsaac Simを操作する場合は、SSHトンネルとDCVセッションのXauthorityを使用します。

```bash
ssh -L 8443:localhost:8443 avita-g5.24
```

DCV Clientは `localhost:8443` に接続します。DCV接続後、EC2上で以下を実行します。

```bash
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"
xdpyinfo -display "$DISPLAY" >/dev/null
glxinfo | grep -E "OpenGL vendor|OpenGL renderer"
```

`OpenGL renderer string: NVIDIA A10G/PCIe/SSE2` を確認できたら、policy再生を実行します。

```bash
./scripts/check_dcv_environment.sh
./scripts/run_dcv_gui_check.sh
./scripts/run_playback_dcv.sh g1_handover_teacher motion
./scripts/run_playback_dcv.sh g1_handover_teacher policy
```

DCV ComposeではXauthorityをコンテナ内の `/root/.Xauthority` としてマウントし、WebRTCポートや `--livestream` は使用しません。

この構成は検証済みの標準手順です。`playback-dcv` は `ipc: host`、NVIDIAの
`graphics/display/video` capability、非headless Kit window、`simulation_app.update()`
によるViewport更新を使用します。G1、机、ボトルの表示と `model_4999.pt` のpolicy推論を
DCV上で確認済みです。現在のpolicyは低速な動作を示しますが、HandOver7の手渡し動作には
未到達であり、追加学習が必要です。

### HandOver7模倣学習の次フェーズ

`G1HandoverObservationCfg.PolicyCfg` には `motion_phase` を追加しています。これは
`episode_length_buf / max_episode_length` の値（0.0から1.0）で、policyが軌道上の現在位置を
識別するための入力です。学習前に、以下で教師軌道を目視確認します。

机と丸棒には明示的なcollision、contact offset、rest offsetを設定し、丸棒の初期位置を
カウンター面からわずかに離しています。これにより初期接触の不安定さやカウンターの
すり抜けを抑えます。この物理設定変更後のcheckpointは再学習が必要です。

HandOver7のNPZは`posed_joints`と`root_positions`を格納しており、再生ローダーは
`posed_joints`を関節軌道として使用します。NPZには丸棒の物体軌道が含まれていないため、
丸棒の受け渡し軌道を完全に再現するには、物体姿勢を記録したデータを追加する必要があります。

```bash
./scripts/run_playback_dcv.sh g1_handover_teacher motion
```

学習済みpolicyを使わず、HandOver7の43 DoF・300フレーム軌道を直接PD目標へ流し込む
基準再生は次のコマンドです。これはpolicy性能の評価ではなく、右腕を含む教師軌道と
joint mapping、物理アセット配置を切り分けるための確認経路です。

```bash
chmod +x ./scripts/run_direct_kinematic_track.sh
PLAYBACK_SPEED=1.0 ./scripts/run_direct_kinematic_track.sh g1_handover_teacher
```

効果測定では無制限ループを使わず、上限を指定します。例えば3エピソードを4倍速で実行し、
各周回の最小距離、把持、解放、終了理由を`[METRIC]`で確認します。

```bash
PLAYBACK_SPEED=4.0 \
PLAYBACK_MAX_EPISODES=3 \
./scripts/run_direct_kinematic_track.sh g1_handover_teacher
```

2026-08-16の検証では3エピソードすべてで次の結果を確認しました。

```text
episode=1 min_distance=0.2189m grasp=1 release=1
episode=2 min_distance=0.2189m grasp=1 release=1
episode=3 min_distance=0.2189m grasp=1 release=1
```

把持はstep 421、解放はstep 451で発火しました。これにより、Loop 1の把持、Loop 3の
解放、Loop 4の3周リセットを定量的に確認済みです。Loop 2の指角度とアタッチオフセットは
`FINGER_TARGET_ANGLE`と`ATTACH_OFFSET`で調整できます。

HandOver7のNPZには丸棒の物体軌道は含まれないため、ロボットの関節軌道は直接再現できますが、
丸棒の手渡し動作は物理接触と別途の物体軌道データが必要です。まず直接PD再生で右腕の
動作を確認し、その後に残差RLまたは物体軌道付きの模倣学習へ進みます。

学習サービスはDCV描画を使わず、`NUM_ENVS`、`ipc: host`、GPU compute capabilityを利用します。
例えば環境数を増やす場合:

```bash
NUM_ENVS=2048 ./run_env.sh train g1_handover_teacher
```

現時点の`train.py`は単一プロセスのRSL-RL実行であり、`torchrun`/DDPの初期化は実装されていません。
そのため、`--distributed`を付けるだけでは4 GPU分散学習にはならず、まず単一GPUで環境数の
スケーリングを検証します。DDP化はrankごとの環境・checkpoint・optimizer同期を実装してから
有効化します。

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