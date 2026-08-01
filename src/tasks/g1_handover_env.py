import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

# Isaac Lab標準のアセットライブラリからG1とオブジェクトの定義をインポート
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import RigidBodyPropertiesCfg

import isaaclab.sim as sim_utils

@configclass
class G1HandoverSceneCfg(InteractiveSceneCfg):
    """シミュレーションステージ上の配置定義"""
    # 地面
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # ロボット (Unitree G1)
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # ノベルティ（500mlペットボトルを模した円柱）
    novelty = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Novelty",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.035, height=0.2,
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        ),
    )

@configclass
class G1HandoverObservationCfg:
    """ポリシーに入力する観測データ (State)"""

    @configclass
    class PolicyCfg(ObsGroup):
        # ロボットの関節角度・速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class G1HandoverRewardsCfg:
    """ハンドオーバーを成功させるための報酬設計"""
    # 転倒ペナルティ（ロボットの体幹が直立しているか）
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class G1HandoverActionsCfg:
    """ロボットへのアクション（制御命令）の定義"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class G1HandoverTerminationsCfg:
    """エピソードの終了条件（リセットタイミング）"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class G1HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """環境全体の統合設定"""
    scene: G1HandoverSceneCfg = G1HandoverSceneCfg(num_envs=64, env_spacing=2.5)
    observations: G1HandoverObservationCfg = G1HandoverObservationCfg()
    actions: G1HandoverActionsCfg = G1HandoverActionsCfg()
    rewards: G1HandoverRewardsCfg = G1HandoverRewardsCfg()
    terminations: G1HandoverTerminationsCfg = G1HandoverTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4   # ポリシーは50Hzで制御
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0

# --- PPO (強化学習アルゴリズム) のハイパーパラメータ設定 ---
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class G1HandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000  # テスト用にイテレーションを設定
    save_interval = 50
    experiment_name = "g1_handover"
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 300, 200],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 300, 200],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )