from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationManagerCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Isaac Lab標準のアセットライブラリからG1とオブジェクトの定義をインポート
from isaaclab_assets.robots.unitree import UNITREE_G1_CFG 
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import RigidBodyPropertiesCfg, UsdGeomMeshPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class G1HandoverSceneCfg:
    """シミュレーションステージ上の配置定義"""
    # 地面
    ground = sim_utils.GroundPlaneCfg()
    # ロボット (Unitree G1)
    robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # ノベルティ（500mlペットボトルを模した円柱）
    novelty = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Novelty",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.035, height=0.2,
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_props=UsdGeomMeshPropertiesCfg(color=(0.0, 0.8, 0.0)),
        ),
    )

@configclass
class G1HandoverObservationCfg(ObservationManagerCfg):
    """ポリシーに入力する観測データ (State)"""
    class policy(ObservationTermCfg):
        # ロボットの関節角度・速度
        joint_pos = ObservationTermCfg(func="isaaclab.envs.mdp:joint_pos_rel")
        角速度 = ObservationTermCfg(func="isaaclab.envs.mdp:joint_vel_rel")
        # 対象物（ボトル）の位置
        object_pos = ObservationTermCfg(func="isaaclab.envs.mdp:object_position", params={"asset_cfg": SceneEntityCfg("novelty")})

@configclass
class G1HandoverRewardsCfg:
    """ハンドオーバーを成功させるための報酬設計"""
    # 転倒ペナルティ（ロボットの体幹が直立しているか）
    termination_penalty = RewTerm(func="isaaclab.envs.mdp:is_terminated", weight=-200.0)
    
    # 手（目標座標）にボトルを近づけるプラス報酬
    reach_object = RewTerm(
        func="isaaclab.envs.mdp:object_position_tracking",
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("novelty"), "target_positions": (0.5, 0.0, 0.7)}
    )

@configclass
class G1HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """環境全体の統合設定"""
    scene: G1HandoverSceneCfg = G1HandoverSceneCfg()
    observations: G1HandoverObservationCfg = G1HandoverObservationCfg()
    rewards: G1HandoverRewardsCfg = G1HandoverRewardsCfg()

    def __init__(self):
        super().__init__()
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4   # ポリシーは50Hzで制御

# --- PPO (強化学習アルゴリズム) のハイパーパラメータ設定 ---
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class G1HandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000  # テスト用にイテレーションを設定
    save_interval = 50
    experiment_name = "g1_handover"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[400, 300, 200], critic_hidden_dims=[400, 300, 200])
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, learning_rate=1e-3, num_learning_epochs=5, num_mini_batches=4)