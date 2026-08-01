# g1-handover-project

Unitree G1 ヒューマノイドロボットを用いた、ノベルティハンドオーバー（手渡し）タスクの強化学習プロジェクト。AWSマルチインスタンス環境において、完全にポータブルかつ一発で再現可能なシミュレーション環境を提供します。

## 🛠️ 技術スタック

![Isaac Sim 6.0.1](https://img.shields.io/badge/Isaac_Sim-6.0.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Isaac Lab Core](https://img.shields.io/badge/Isaac_Lab-Core-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS EC2/S3](https://img.shields.io/badge/AWS-EC2%20%2F%20S3-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)

---

## 📊 システムアーキテクチャ (Mermaid)

AWSのキャパシティ不足を回避するため、コードと学習成果物、シミュレーション映像の経路を完全に分離・ポータブル化しています。

```mermaid
graph TD
    subgraph Local PC
        Client[Isaac Sim WebRTC Client 2.0.0]
    end

    subgraph AWS EC2 Instance (g7e / g5 / g6)
        Host[Host OS / run_env.sh]
        Docker[Docker Compose / Container]
        Logs[./logs / Checkpoints]
    end

    subgraph Remote Infrastructure
        GitHub[GitHub Repo]
        S3[AWS S3 Bucket]
    end

    %% ワークフローの流れ
    GitHub -->|1. git clone / pull| Host
    S3 -->|2. aws s3 sync latest| Logs
    Host -->|3. compose up| Docker
    Docker -->|4. Train Loop 1000 iter| Logs
    Docker -.->|5. WebRTC Stream / TCP:49100, UDP:47998| Client
    Logs -->|6. aws s3 sync archive| S3
```

## 🚀 Quick Start

どのAWS GPUインスタンス（Ubuntu 22.04 LTS, NVIDIA Driver/Dockerインストール済み）からでも、以下のステップのみで環境構築と学習が自動実行されます。

### 1. 前提条件の確認 (AWS セキュリティグループ)

接続元のローカルPC（マイIP）に対し、以下のインバウンドルールが開放されていることを確認してください。

* **TCP:** `49100` (シグナリング用固定ポート)
* **UDP:** `47998` (映像ストリーム用固定ポート)

### 2. クローンと実行

```bash
# 1. リポジトリの取得
git clone [https://github.com/](https://github.com/)<your-username>/g1-handover-project.git
cd g1-handover-project

# 2. S3バケット名の環境変数を設定 (または run_env.sh 内で固定)
# export S3_BUCKET="your-isaac-sim-s3-bucket-name"

# 3. 起動スクリプトを実行 (環境構築・S3同期・学習キックまで自動)
chmod +x run_env.sh
./run_env.sh

```

### 3. ローカルPCからの接続

1. **Isaac Sim WebRTC Streaming Client 2.0.0** を起動。
2. Server 欄に EC2 の `パブリックIPアドレスのみ` を入力（※ポート番号は入力しない）。
3. **Connect** をクリックして画面が展開することを確認。


## 🛠 開発ルール (Docs as Code)

プロジェクトの再現性を維持するため、以下のルールを厳守してください。

1. **GitHubとS3の役割分担 (必須)**
* **GitHub:** ソースコードのみを管理。`logs/` ディレクトリは大容量の `.pt` ファイルを含むため、`.gitignore` で完全除外されています。
* **S3:** 学習成果物（チェックポイント、TensorBoardログ）を管理。`run_env.sh` が起動時と終了時に自動で双方向同期します。


2. **依存関係の追加ルール**
* Isaac Lab の `extras`（`mimic`, `visualizers`, `teleop`）は `psutil` や `websockets` の致命的なバージョン衝突を引き起こすため、**インストール禁止**とします。
* 必要なモジュールは `scripts/container_entrypoint.sh` に `omni.isaac.lab` コア系のみを明示して最小インストールしてください。


3. **Isaac Lab 6.x API 準拠**
* 旧引数（`--headless` など）は非推奨です。
* タスククラス（`g1_handover_env.py`）の実装時は、最新の `ManagerBasedRLEnvCfg` に従い、`ObservationManagerCfg`, `ActionTermCfg`, `TerminationTermCfg` を用いてマネージャー駆動で記述してください。

### ブランチ命名規則

| Prefix | 用途 | SemVer影響 | 例 |
| :--- | :--- | :--- | :--- |
| `main` | メインブランチ | なし | `main` |
| `develop` | ステージングブランチ | なし | `develop` |
| `feature/` | 新機能追加 | Minor | `feature/add-login-function` |
| `bugfix/` | バグ修正 | Patch | `bugfix/fix-crash-on-startup` |
| `hotfix/` | 緊急修正 | Patch | `hotfix/fix-security-vulnerability` |
| `release/` | リリース準備 | Patch/Minor | `release/v1.2.0-prep` |
| `docs/` | ドキュメント更新のみ | Patch | `docs/update-api-docs` |
| `chore/` | その他メンテナンス | Patch | `chore/update-dependencies` |

### コミットメッセージ規約

- `type(scope): subject` 例: `feat(api): add login`
- type例: feat, fix, docs, chore, refactor, test, ci
- scopeは任意、subjectは簡潔に

### 運用のポイント

- **Docs as Code**: コード修正時はdocs/も必ず更新
- **main直Push禁止**: PR経由でマージ
- **CI/CD必須**: GitHub Actions等で自動テスト・デプロイ
- **README.md整備**: QuickStart・開発手順・依存関係を明記
- **テンプレート活用**: PRテンプレート・Issueテンプレートを用意
