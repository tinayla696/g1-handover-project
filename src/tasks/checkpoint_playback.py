import argparse
import copy
import importlib.metadata as metadata
import os
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

TASK_ID = os.getenv("TASK_ID", "g1_handover_base")
LOGS_MOUNT_ROOT = Path(os.getenv("LOGS_MOUNT_ROOT", "/workspace/g1-handover-project/logs"))
MOTION_DIR = Path(os.getenv("MOTION_DIR", "data/motions/HandOver7"))
CONTROL_DT = 0.02
GRASP_DISTANCE = float(os.getenv("GRASP_DISTANCE", "0.22"))
FINGER_TARGET_ANGLE = float(os.getenv("FINGER_TARGET_ANGLE", "1.2"))
ATTACH_OFFSET = (0.0, 0.0, 0.02)
RELEASE_FRACTION = float(os.getenv("RELEASE_FRACTION", "0.90"))

parser = argparse.ArgumentParser(description="Play back a trained G1 Handover checkpoint.")
parser.add_argument("--task", type=str, default="G1-Handover-v0", help="Registered gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the playback run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments; playback should be 1.")
parser.add_argument("--loop", action="store_true", help="Loop playback until Ctrl+C.")
parser.add_argument("--max_episodes", type=int, default=1, help="Maximum motion episodes; 0 means unlimited.")
parser.add_argument("--max_steps", type=int, default=-1, help="Maximum motion steps per episode; -1 means timeout limit.")
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed relative to real time: 1.0 is real-time, 0.5 is half-speed, 2.0 is double-speed.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="policy",
    choices=("policy", "motion"),
    help="Playback mode: policy checkpoint inference or direct HandOver7 motion replay.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.playback_speed <= 0:
    parser.error("--playback_speed must be greater than zero")
if args_cli.max_episodes < 0 or args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_episodes must be >= 0 and --max_steps must be -1 or greater")
PLAYBACK_DT = CONTROL_DT / args_cli.playback_speed
args_cli.headless = False
args_cli.visualizer_intent = {"has_any_visualizers": True, "has_kit_visualizer": True}

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import tasks  # noqa: F401


def _safe_render(env) -> None:
    """Render with wrapper-awareness (RSL wrappers do not expose render())."""
    render_target = env.unwrapped if hasattr(env, "unwrapped") else env
    render_fn = getattr(render_target, "render", None)
    if callable(render_fn):
        render_fn()
    # Keep the Isaac Sim Kit viewport responsive when optional visualizer Extras are absent.
    simulation_app.update()


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    candidates = [path]
    if path.name:
        candidates.append(LOGS_MOUNT_ROOT / path.name)
    if "logs" in path.parts:
        logs_index = path.parts.index("logs")
        relative_parts = path.parts[logs_index + 1 :]
        if relative_parts:
            candidates.append(LOGS_MOUNT_ROOT.joinpath(*relative_parts))
        if len(relative_parts) > 1:
            candidates.append(LOGS_MOUNT_ROOT.joinpath(*relative_parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def _load_motion_buffer(
    motion_dir: Path = MOTION_DIR, control_dt: float = CONTROL_DT
) -> tuple[torch.Tensor, list[str] | None, torch.Tensor | None, torch.Tensor | None]:
    npz_path = motion_dir / "HandOver7_unitree_g1.npz"
    if not npz_path.exists():
        npz_path = motion_dir / "HandOver7.npz"

    if not npz_path.exists():
        print(f"[WARN] Motion file not found at {npz_path}. Using zero fallback.", flush=True)
        return torch.zeros((400, 37), dtype=torch.float32), None, None, None

    payload = np.load(npz_path, allow_pickle=True)
    fps = float(np.asarray(payload["fps"]).item()) if "fps" in payload else 30.0

    joint_names = None
    if "joint_names" in payload:
        joint_names = [str(name) for name in np.asarray(payload["joint_names"]).tolist()]

    joints = None
    for key in ("joint_positions", "joint_pos", "joint_position", "posed_joints", "qpos", "positions"):
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

    num_frames, dof = joints.shape
    duration = max((num_frames - 1) / max(fps, 1e-6), control_dt)
    t_orig = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    t_target = np.arange(0.0, duration + 1e-9, control_dt, dtype=np.float32)

    resampled = np.empty((t_target.shape[0], dof), dtype=np.float32)
    for i in range(dof):
        resampled[:, i] = np.interp(t_target, t_orig, joints[:, i])

    root_position = None
    if "root_position" in payload:
        raw_root_position = np.asarray(payload["root_position"], dtype=np.float32)
        if raw_root_position.shape == (num_frames, 3):
            root_position = np.empty((t_target.shape[0], 3), dtype=np.float32)
            for i in range(3):
                root_position[:, i] = np.interp(t_target, t_orig, raw_root_position[:, i])

    root_quaternion = None
    if "root_quaternion_wxyz" in payload:
        raw_root_quaternion = np.asarray(payload["root_quaternion_wxyz"], dtype=np.float32)
        if raw_root_quaternion.shape == (num_frames, 4):
            root_quaternion = np.empty((t_target.shape[0], 4), dtype=np.float32)
            for i in range(4):
                root_quaternion[:, i] = np.interp(t_target, t_orig, raw_root_quaternion[:, i])
            norms = np.linalg.norm(root_quaternion, axis=1, keepdims=True)
            root_quaternion = root_quaternion / np.maximum(norms, 1e-8)
            # HandOver7 stores quaternions as wxyz; Isaac Lab expects xyzw root poses.
            root_quaternion = root_quaternion[:, [1, 2, 3, 0]]

    motion = torch.tensor(resampled, dtype=torch.float32)
    root_position_tensor = torch.tensor(root_position, dtype=torch.float32) if root_position is not None else None
    root_quaternion_tensor = (
        torch.tensor(root_quaternion, dtype=torch.float32) if root_quaternion is not None else None
    )
    print(
        f"✓ [MotionBuffer] Resampled HandOver7 to {1.0 / control_dt:.1f}Hz: {tuple(motion.shape)}; "
        f"root_pose={'loaded' if root_position_tensor is not None and root_quaternion_tensor is not None else 'absent'}",
        flush=True,
    )
    return motion, joint_names, root_position_tensor, root_quaternion_tensor


def _write_root_pose_to_sim(robot, root_pose: torch.Tensor) -> None:
    """Write a root pose using the current Isaac Lab articulation API."""
    writer = getattr(robot, "write_root_link_pose_to_sim", None)
    if callable(writer):
        writer(root_pose)
        return
    writer = getattr(robot, "write_root_pose_to_sim", None)
    if callable(writer):
        writer(root_pose)
        return
    raise RuntimeError("Robot articulation does not expose a root pose write API.")


def _write_root_velocity_to_sim(asset, root_velocity: torch.Tensor) -> None:
    """Clear root linear and angular velocity when a pose is kinematically applied."""
    writer = getattr(asset, "write_root_velocity_to_sim", None)
    if callable(writer):
        writer(root_velocity)
        return
    writer = getattr(asset, "write_root_com_velocity_to_sim", None)
    if callable(writer):
        writer(root_velocity)


def _find_body_index(robot, preferred_name: str) -> int | None:
    body_names = list(getattr(robot, "body_names", []))
    if preferred_name in body_names:
        return body_names.index(preferred_name)
    candidates = [
        (index, name)
        for index, name in enumerate(body_names)
        if "right" in name.lower() and ("palm" in name.lower() or "hand" in name.lower())
    ]
    return candidates[0][0] if candidates else None


def main():
    checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
    print(f"Resolved checkpoint path: {checkpoint_path}", flush=True)
    task_spec = gym.spec(args_cli.task)
    env_cfg = copy.deepcopy(task_spec.kwargs["cfg"])
    runner_cfg_entry_point = task_spec.kwargs["rsl_rl_cfg_entry_point"]

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "log_dir"):
        env_cfg.log_dir = str(checkpoint_path.parent)

    print("Creating playback environment...", flush=True)
    # Keep rendering attached to the DCV desktop window.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array", disable_env_checker=True)

    if args_cli.mode == "motion":
        print(f"TASK_ID: {TASK_ID}", flush=True)
        print(f"Playback checkpoint: {checkpoint_path} (ignored in motion mode)", flush=True)
        print("Running HandOver7 motion replay...", flush=True)

        obs, _ = env.reset(seed=args_cli.seed)
        action_dim = int(env.action_space.shape[-1])
        motion_buffer, motion_joint_names, root_positions, root_quaternions = _load_motion_buffer()
        motion_buffer = motion_buffer.to(dtype=torch.float32)

        env_unwrapped = env.unwrapped
        robot = env_unwrapped.scene["robot"]
        default_joint_pos = robot.data.default_joint_pos[:, :action_dim].to(robot.device)
        robot_joint_names = list(getattr(robot, "joint_names", []))[:action_dim]
        try:
            novelty = env_unwrapped.scene["novelty"]
        except KeyError:
            novelty = None
        hand_body_index = _find_body_index(robot, "right_hand_palm_link")
        right_finger_indices = [
            index
            for index, name in enumerate(robot_joint_names)
            if name.startswith("right_") and "joint" in name and any(
                token in name for token in ("zero", "one", "two", "three", "four", "five", "six", "thumb", "index", "middle")
            )
        ]
        grasped = False
        released = False
        grasp_step = None
        episode_index = 1
        episode_step = 0
        minimum_distance = float("inf")
        grasp_count = 0
        release_count = 0
        grasp_distance = GRASP_DISTANCE
        release_fraction = RELEASE_FRACTION
        if hand_body_index is not None:
            print(f"Right hand body index: {hand_body_index}", flush=True)
        else:
            print("[WARN] right_hand_palm_link not found; grasp attachment disabled.", flush=True)
        if root_positions is not None and root_quaternions is not None:
            root_positions = root_positions.to(robot.device)
            root_quaternions = root_quaternions.to(robot.device)
        root_pose_buffer = torch.empty((1, 7), dtype=torch.float32, device=robot.device)
        zero_root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=robot.device)
        novelty_pose_buffer = torch.empty((1, 7), dtype=torch.float32, device=robot.device)
        action_scale = float(getattr(env_unwrapped.cfg.actions.joint_pos, "scale", 1.0))
        action_scale = max(action_scale, 1e-6)

        source_index_by_name = {}
        if motion_joint_names:
            source_index_by_name = {name: idx for idx, name in enumerate(motion_joint_names)}

        alias_name_map = {
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

        mapped_source_indices = torch.full((action_dim,), -1, dtype=torch.long, device=robot.device)
        if source_index_by_name and robot_joint_names:
            mapped = 0
            for action_idx, joint_name in enumerate(robot_joint_names):
                source_name = joint_name if joint_name in source_index_by_name else alias_name_map.get(joint_name)
                if source_name in source_index_by_name:
                    mapped_source_indices[action_idx] = int(source_index_by_name[source_name])
                    mapped += 1
            print(f"Mapped {mapped}/{action_dim} action joints from HandOver7 joint names.", flush=True)
        else:
            fallback_dim = min(action_dim, int(motion_buffer.shape[-1]))
            mapped_source_indices[:fallback_dim] = torch.arange(fallback_dim, device=robot.device)
            print(
                "[WARN] joint_names not found in motion or robot; falling back to index-based mapping.",
                flush=True,
            )

        step_index = 0
        try:
            while simulation_app.is_running():
                t_step_start = time.monotonic()

                frame = motion_buffer[step_index % motion_buffer.shape[0]].to(robot.device)
                target_joint_pos = default_joint_pos.clone()

                valid = mapped_source_indices >= 0
                if bool(valid.any().item()):
                    target_joint_pos[:, valid] = frame[mapped_source_indices[valid]].unsqueeze(0)

                if not grasped and not released and novelty is not None and hand_body_index is not None:
                    hand_position = robot.data.body_state_w.torch[:, hand_body_index, :3]
                    novelty_position = novelty.data.root_pos_w.torch
                    distance = torch.linalg.norm(hand_position - novelty_position, dim=-1)
                    minimum_distance = min(minimum_distance, float(distance.min().item()))
                    if step_index % 30 == 0:
                        print(
                            f"  [DISTANCE] step={step_index} | hand-novelty dist={float(distance.min().item()):.4f}m "
                            f"threshold={grasp_distance:.3f}m",
                            flush=True,
                        )
                    if bool((distance < grasp_distance).any().item()):
                        grasped = True
                        grasp_step = step_index
                        grasp_count += 1
                        print(
                            f"[GRASP] step={step_index} dist={float(distance.min().item()):.4f}m "
                            "right hand reached novelty; attaching object.",
                            flush=True,
                        )

                if right_finger_indices and grasped:
                    target_joint_pos[:, right_finger_indices] = FINGER_TARGET_ANGLE

                # Motion replay uses absolute trajectory targets. Do not clamp to [-1, 1],
                # otherwise the 0.05 action scale compresses large arm motions to near-static output.
                raw_action = (target_joint_pos - default_joint_pos) / action_scale

                with torch.inference_mode():
                    obs, _, terminated, truncated, _ = env.step(raw_action)

                if root_positions is not None and root_quaternions is not None:
                    frame_id = step_index % root_positions.shape[0]
                    root_pose_buffer[:, :3].copy_(root_positions[frame_id])
                    root_pose_buffer[:, 3:].copy_(root_quaternions[frame_id])
                    _write_root_pose_to_sim(robot, root_pose_buffer)
                    _write_root_velocity_to_sim(robot, zero_root_velocity)

                if grasped and novelty is not None and hand_body_index is not None:
                    hand_state = robot.data.body_state_w.torch[:, hand_body_index]
                    novelty_pose_buffer.copy_(hand_state[:, :7])
                    novelty_pose_buffer[:, 0].add_(ATTACH_OFFSET[0])
                    novelty_pose_buffer[:, 1].add_(ATTACH_OFFSET[1])
                    novelty_pose_buffer[:, 2].add_(ATTACH_OFFSET[2])
                    novelty.write_root_pose_to_sim(novelty_pose_buffer)
                    _write_root_velocity_to_sim(novelty, zero_root_velocity)

                release_step = int(motion_buffer.shape[0] * release_fraction)
                if grasped and grasp_step is not None and step_index >= max(release_step, grasp_step + 30):
                    grasped = False
                    released = True
                    release_count += 1
                    print(f"[RELEASE] step={step_index} handover phase reached; releasing novelty.", flush=True)

                # Explicit render keeps the DCV desktop window responsive.
                _safe_render(env)

                # Rate-limit to the selected playback speed while preventing encoder overload.
                elapsed = time.monotonic() - t_step_start
                if elapsed < PLAYBACK_DT:
                    time.sleep(PLAYBACK_DT - elapsed)

                step_index += 1
                episode_step += 1
                if step_index % 60 == 0:
                    max_delta = float((target_joint_pos - default_joint_pos).abs().max().item())
                    max_action = float(raw_action.abs().max().item())
                    print(
                        f"  motion replay step {step_index} | max_joint_delta={max_delta:.3f}rad | max_action={max_action:.2f}",
                        flush=True,
                    )
                # Log per-joint deltas once at startup to identify which joints move
                if step_index == 60:
                    deltas = (target_joint_pos - default_joint_pos).abs().squeeze(0)
                    print("  [JOINT DELTA SNAPSHOT] top-10 moving joints:", flush=True)
                    top_indices = deltas.argsort(descending=True)[:10]
                    for idx in top_indices:
                        jname = robot_joint_names[int(idx)] if int(idx) < len(robot_joint_names) else f"joint_{idx}"
                        print(f"    {jname}: {float(deltas[idx]):.4f} rad", flush=True)

                if bool((terminated | truncated).any().item()):
                    reason = "timeout" if bool(truncated.any().item()) else "fall"
                    print(
                        f"[METRIC] episode={episode_index} reason={reason} steps={episode_step} "
                        f"min_distance={minimum_distance:.4f}m grasp={grasp_count} release={release_count}",
                        flush=True,
                    )
                    if args_cli.loop and (args_cli.max_episodes == 0 or episode_index < args_cli.max_episodes):
                        obs, _ = env.reset(seed=args_cli.seed)
                        step_index = 0  # restart motion from frame 0
                        episode_index += 1
                        episode_step = 0
                        grasped = False
                        released = False
                        grasp_step = None
                        minimum_distance = float("inf")
                        grasp_count = 0
                        release_count = 0
                        continue
                    break
                if args_cli.max_steps > 0 and episode_step >= args_cli.max_steps:
                    print(
                        f"[METRIC] episode={episode_index} reason=max_steps steps={episode_step} "
                        f"min_distance={minimum_distance:.4f}m grasp={grasp_count} release={release_count}",
                        flush=True,
                    )
                    break
        finally:
            env.close()
        return

    runner_cfg_cls = string_to_callable(runner_cfg_entry_point)
    runner_cfg = runner_cfg_cls()
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    print("Wrapping environment for RSL-RL...", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    print("Resetting playback environment...", flush=True)
    env.reset()

    print("Constructing OnPolicyRunner...", flush=True)
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=None,
        device=runner_cfg.device,
    )
    print("Loading checkpoint weights...", flush=True)
    runner.load(str(checkpoint_path))
    print("Fetching inference policy...", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"TASK_ID: {TASK_ID}", flush=True)
    print(f"Playback checkpoint: {checkpoint_path}", flush=True)
    print(f"Playback speed: {args_cli.playback_speed:.2f}x", flush=True)
    print("Running checkpoint playback...", flush=True)

    obs = env.get_observations()
    step_index = 0
    try:
        while simulation_app.is_running():
            t_step_start = time.monotonic()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            # Explicit render keeps the DCV desktop window responsive.
            _safe_render(env)
            elapsed = time.monotonic() - t_step_start
            if elapsed < PLAYBACK_DT:
                time.sleep(PLAYBACK_DT - elapsed)
            step_index += 1
            if step_index % 60 == 0:
                max_action = float(actions.abs().max().item())
                mean_action = float(actions.abs().mean().item())
                print(
                    f"  playback step {step_index} | max_action={max_action:.3f} | mean_action={mean_action:.3f}",
                    flush=True,
                )
            if bool(dones.any().item()):
                if args_cli.loop:
                    env.reset()
                    obs = env.get_observations()
                    continue
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()