import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
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


MOTION_DIR = Path(os.getenv("MOTION_DIR", "data/motions/HandOver7"))
CONTROL_DT = 0.02

ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
]

MOTION_INIT_JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.35, 3.05),
    "right_hip_pitch_joint": (-2.35, 3.05),
    "torso_joint": (-2.618, 2.618),
    "left_hip_roll_joint": (-0.260, 2.530),
    "right_hip_roll_joint": (-2.530, 0.260),
    "left_shoulder_pitch_joint": (-2.967, 2.792),
    "right_shoulder_pitch_joint": (-2.967, 2.792),
    "left_hip_yaw_joint": (-2.750, 2.750),
    "right_hip_yaw_joint": (-2.750, 2.750),
    "left_shoulder_roll_joint": (-1.588, 2.251),
    "right_shoulder_roll_joint": (-2.251, 1.588),
    "left_knee_joint": (-0.335, 2.545),
    "right_knee_joint": (-0.335, 2.545),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "left_ankle_pitch_joint": (-0.680, 0.730),
    "right_ankle_pitch_joint": (-0.680, 0.730),
    "left_elbow_pitch_joint": (-0.227, 3.421),
    "right_elbow_pitch_joint": (-0.227, 3.421),
    "left_ankle_roll_joint": (-0.552, 0.552),
    "right_ankle_roll_joint": (-0.552, 0.552),
    "left_elbow_roll_joint": (-1.745, 1.745),
    "right_elbow_roll_joint": (-1.745, 1.745),
    "left_five_joint": (0.0, 1.840),
    "left_three_joint": (0.0, 1.840),
    "left_zero_joint": (0.0, 1.840),
    "right_five_joint": (0.0, 1.840),
    "right_three_joint": (0.0, 1.840),
    "right_zero_joint": (0.0, 1.840),
    "left_six_joint": (0.0, 1.840),
    "left_four_joint": (0.0, 1.840),
    "left_one_joint": (0.0, 1.840),
    "right_six_joint": (0.0, 1.840),
    "right_four_joint": (0.0, 1.840),
    "right_one_joint": (0.0, 1.840),
    "left_two_joint": (0.0, 1.840),
    "right_two_joint": (0.0, 1.840),
}

MOTION_INIT_EXCLUDED_JOINTS = {
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
}

class MotionBufferManager:
    """Load HandOver7 and resample it to the policy control frequency."""

    _buffer: torch.Tensor | None = None
    _joint_names: list[str] | None = None  # HandOver7 joint names from NPZ

    @classmethod
    def get_buffer(cls, motion_dir: Path = MOTION_DIR, control_dt: float = CONTROL_DT) -> torch.Tensor:
        if cls._buffer is not None:
            return cls._buffer

        npz_path = motion_dir / "HandOver7_unitree_g1.npz"
        if not npz_path.exists():
            npz_path = motion_dir / "HandOver7.npz"

        if not npz_path.exists():
            print(f"[WARN] Motion file not found at {npz_path}. Using zero buffer fallback.")
            cls._buffer = torch.zeros((400, 29), dtype=torch.float32)
            return cls._buffer

        payload = np.load(npz_path, allow_pickle=True)
        if "fps" in payload:
            fps = float(np.asarray(payload["fps"]).item())
        else:
            fps = 30.0

        if "joint_names" in payload:
            cls._joint_names = [str(n) for n in np.asarray(payload["joint_names"]).tolist()]

        candidate_keys = ("joint_positions", "joint_pos", "joint_position", "qpos", "positions")
        joints = None
        for key in candidate_keys:
            if key in payload:
                joints = np.asarray(payload[key], dtype=np.float32)
                break

        if joints is None:
            if len(payload.files) == 1:
                joints = np.asarray(payload[payload.files[0]], dtype=np.float32)
            else:
                raise ValueError(f"Unknown motion format in {npz_path}; available keys: {payload.files}")

        if joints.ndim == 1:
            joints = joints[None, :]
        if joints.ndim > 2:
            joints = joints.reshape(joints.shape[0], -1)

        num_frames = joints.shape[0]
        duration = max((num_frames - 1) / max(fps, 1e-6), control_dt)
        t_orig = np.linspace(0.0, duration, num_frames)
        t_target = np.arange(0.0, duration + 1e-9, control_dt)

        interp_fn = interp1d(t_orig, joints, kind="linear", axis=0, fill_value="edge", bounds_error=False)
        resampled = interp_fn(t_target)

        cls._buffer = torch.tensor(resampled, dtype=torch.float32)
        print(f"✓ [MotionBuffer] Resampled HandOver7 to {1.0 / control_dt:.1f}Hz: {tuple(cls._buffer.shape)}, joint_names: {len(cls._joint_names) if cls._joint_names else 'N/A'}")
        return cls._buffer


def _project_motion_initial_pose(motion_frame: torch.Tensor) -> dict[str, float]:
    eps = 1e-3
    joint_pos: dict[str, float] = {}
    for joint_name, value in zip(ACTUATED_JOINT_NAMES, motion_frame[: len(ACTUATED_JOINT_NAMES)].tolist()):
        if joint_name in MOTION_INIT_EXCLUDED_JOINTS:
            continue
        lower, upper = MOTION_INIT_JOINT_LIMITS[joint_name]
        safe_lower = lower + eps
        safe_upper = upper - eps
        if safe_lower >= safe_upper:
            safe_lower, safe_upper = lower, upper
        joint_pos[joint_name] = float(np.clip(value, safe_lower, safe_upper))
    return joint_pos


# HandOver7→G1 joint name alias (matches checkpoint_playback.py / motion_replay.py)
_HANDOVER7_ALIAS: dict[str, str] = {
    "torso_joint": "waist_pitch_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
    "left_zero_joint": "left_hand_thumb_0_joint",
    "left_one_joint": "left_hand_thumb_1_joint",
    "left_two_joint": "left_hand_thumb_2_joint",
    "left_three_joint": "left_hand_index_0_joint",
    "left_four_joint": "left_hand_index_1_joint",
    "left_five_joint": "left_hand_middle_0_joint",
    "left_six_joint": "left_hand_middle_1_joint",
    "right_zero_joint": "right_hand_thumb_0_joint",
    "right_one_joint": "right_hand_thumb_1_joint",
    "right_two_joint": "right_hand_thumb_2_joint",
    "right_three_joint": "right_hand_index_0_joint",
    "right_four_joint": "right_hand_index_1_joint",
    "right_five_joint": "right_hand_middle_0_joint",
    "right_six_joint": "right_hand_middle_1_joint",
}

# Cache: G1 joint index → HandOver7 source column index (-1 = unmapped)
_motion_col_indices: list[int] | None = None


def _build_motion_col_indices(robot_joint_names: list[str], motion_joint_names: list[str]) -> list[int]:
    """Build G1→HandOver7 column index map once and cache it."""
    src = {name: i for i, name in enumerate(motion_joint_names)}
    indices: list[int] = []
    mapped = 0
    for rname in robot_joint_names:
        src_name = rname if rname in src else _HANDOVER7_ALIAS.get(rname)
        if src_name and src_name in src:
            indices.append(src[src_name])
            mapped += 1
        else:
            indices.append(-1)
    print(f"[motion_tracking_reward] Mapped {mapped}/{len(robot_joint_names)} G1 joints to HandOver7", flush=True)
    return indices


def motion_tracking_reward(
    env: ManagerBasedRLEnv, std: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Exponential joint-tracking reward against the resampled HandOver7 trajectory."""
    global _motion_col_indices

    motion_buffer = MotionBufferManager.get_buffer()
    robot = env.scene[asset_cfg.name]
    current_joint_pos = robot.data.joint_pos
    # env.device is a string ("cuda:0") — safe for torch .to()
    device: str = env.device
    current_joint_pos = robot.data.joint_pos
    n_robot = current_joint_pos.shape[-1]

    # Build column-index mapping on first call
    if _motion_col_indices is None:
        robot_joint_names = list(robot.joint_names) if hasattr(robot, "joint_names") else ACTUATED_JOINT_NAMES
        # HandOver7 NPZ joint_names are stored in MotionBufferManager payload; fall back to ACTUATED_JOINT_NAMES
        motion_joint_names_raw = getattr(MotionBufferManager, "_joint_names", None)
        if motion_joint_names_raw is None:
            # No name info → skip tracking (return 0)
            return torch.zeros(current_joint_pos.shape[0], device=device)
        _motion_col_indices = _build_motion_col_indices(robot_joint_names[:n_robot], motion_joint_names_raw)

    step_ids = env.episode_length_buf.long().cpu().clamp(max=motion_buffer.shape[0] - 1)
    motion_frame = motion_buffer[step_ids].to(device)  # (envs, 43)

    # Remap motion columns to G1 joint order
    target_pos = current_joint_pos.clone()
    for g1_idx, src_idx in enumerate(_motion_col_indices):
        if src_idx >= 0 and g1_idx < n_robot:
            target_pos[:, g1_idx] = motion_frame[:, src_idx]

    joint_err = torch.sum(torch.square(current_joint_pos - target_pos), dim=-1)
    return torch.exp(-joint_err / (std**2))


def object_approach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("novelty"),
) -> torch.Tensor:
    """Reward the robot for approaching the novelty object and keeping it near the handover zone."""

    robot = env.scene[robot_cfg.name]
    novelty = env.scene[object_cfg.name]

    robot_pos = robot.data.root_pos_w.torch
    object_pos = novelty.data.root_pos_w.torch
    target_pos = torch.tensor((0.88, 0.0, 1.02), dtype=object_pos.dtype, device=object_pos.device)

    approach_error = torch.linalg.norm(robot_pos - object_pos, dim=-1)
    handover_error = torch.linalg.norm(object_pos - target_pos, dim=-1)
    return torch.exp(-approach_error / std) + 0.5 * torch.exp(-handover_error / (std * 1.5))

@configclass
class G1HandoverSceneCfg(InteractiveSceneCfg):
    """シミュレーションステージ上の配置定義"""
    # 全体を明るくする環境光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )
    # 地面
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # ロボット (Unitree G1)
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 手渡し用の机（静的）
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 0.35), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.6, 0.7),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.2)),
        ),
    )
    
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
    # Phase 1: 生存優先。motion_tracking は joint mapping 修正済みで有効化
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
    motion_tracking = RewTerm(func=motion_tracking_reward, weight=3.0, params={"std": 0.5})
    object_approach = RewTerm(
        func=object_approach_reward,
        weight=1.0,
        params={"std": 0.3, "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("novelty")},
    )


@configclass
class G1HandoverActionsCfg:
    """ロボットへのアクション（制御命令）の定義"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05,
        use_default_offset=True,
    )


@configclass
class G1HandoverTerminationsCfg:
    """エピソードの終了条件（リセットタイミング）"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class G1HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """環境全体の統合設定"""
    scene: G1HandoverSceneCfg = G1HandoverSceneCfg(num_envs=512, env_spacing=2.5)
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
        self.episode_length_s = 20.0  # HandOver7は9.98秒、余裕をもたせる

        # G1デフォルト立位姿勢で初期化（ランダムポリシーによる即時転倒を防ぐ）
        try:
            MotionBufferManager.get_buffer()
        except Exception as exc:
            print(f"[WARN] Motion buffer preload skipped: {exc}")

# --- PPO (強化学習アルゴリズム) のハイパーパラメータ設定 ---
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class G1HandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "g1_handover_teacher"
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )