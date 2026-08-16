import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay teleoperated G1 joint motion in Isaac Lab.")
parser.add_argument("--task", type=str, default="G1-Handover-v0", help="Registered gym task name.")
parser.add_argument("--motion_dir", type=str, default="data/motions/HandOver7", help="Directory containing HandOver7 files.")
parser.add_argument("--source", choices=["auto", "csv", "npz"], default="auto", help="Motion file source type.")
parser.add_argument("--csv_file", type=str, default="HandOver7_unitree_g1.csv", help="CSV filename inside motion_dir.")
parser.add_argument("--npz_file", type=str, default="HandOver7.npz", help="NPZ filename inside motion_dir.")
parser.add_argument("--fps", type=float, default=30.0, help="Replay frames per second.")
parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic setup.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments; replay probe should be 1.")
parser.add_argument("--loop", action="store_true", help="Loop playback until Ctrl+C.")
parser.add_argument("--max_frames", type=int, default=-1, help="Optional frame cap; -1 means full clip.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import tasks


def _normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(":", "")
        .replace("/", "")
        .replace(".", "")
    )


def _get_robot_joint_names(robot) -> list[str]:
    if hasattr(robot, "joint_names"):
        return list(robot.joint_names)
    if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
        return list(robot.data.joint_names)
    raise RuntimeError("Could not discover robot joint names from articulation object.")


def _load_csv_motion(csv_path: Path, joint_names: list[str]) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < len(joint_names):
            raise ValueError(
                f"CSV has {data.shape[1]} columns, but robot needs {len(joint_names)} joints."
            )
        return data[:, : len(joint_names)]

    header_map = {_normalize_name(col): idx for idx, col in enumerate(header)}
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]

    joint_indices: list[int] = []
    missing: list[str] = []
    for jn in joint_names:
        key = _normalize_name(jn)
        if key in header_map:
            joint_indices.append(header_map[key])
        else:
            missing.append(jn)

    if not missing:
        return rows[:, joint_indices]

    if rows.shape[1] >= len(joint_names):
        print(
            "[WARN] Could not map all joints by CSV header names. "
            "Falling back to first N columns by robot joint count."
        )
        return rows[:, : len(joint_names)]

    missing_preview = ", ".join(missing[:8])
    raise ValueError(
        "CSV header does not match robot joints and fallback is not possible. "
        f"Missing examples: {missing_preview}"
    )


def _load_npz_motion(npz_path: Path, joint_count: int) -> np.ndarray:
    payload = np.load(npz_path)
    candidate_keys = [
        "joint_pos",
        "joint_positions",
        "qpos",
        "posed_joints",
        "positions",
        "motion",
        "data",
    ]

    arr = None
    for key in candidate_keys:
        if key in payload:
            arr = payload[key]
            break

    if arr is None:
        if len(payload.files) == 1:
            arr = payload[payload.files[0]]
        else:
            raise ValueError(
                "NPZ does not contain a known joint position key. "
                f"Available keys: {payload.files}"
            )

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[1] < joint_count:
        raise ValueError(
            f"NPZ motion has {arr.shape[1]} columns, but robot needs {joint_count} joints."
        )
    return arr[:, :joint_count]


def _resolve_motion_frames(motion_dir: Path, source: str, joint_names: list[str]) -> np.ndarray:
    csv_path = motion_dir / args_cli.csv_file
    npz_path = motion_dir / args_cli.npz_file

    if source == "csv":
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return _load_csv_motion(csv_path, joint_names)

    if source == "npz":
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        return _load_npz_motion(npz_path, len(joint_names))

    if csv_path.exists():
        print(f"Using CSV motion source: {csv_path}")
        return _load_csv_motion(csv_path, joint_names)
    if npz_path.exists():
        print(f"Using NPZ motion source: {npz_path}")
        return _load_npz_motion(npz_path, len(joint_names))

    raise FileNotFoundError(
        f"No motion file found under {motion_dir}. Expected {csv_path.name} or {npz_path.name}."
    )


def _write_joint_positions(robot, joint_pos_tensor: torch.Tensor):
    joint_vel_tensor = torch.zeros_like(joint_pos_tensor)

    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor)
        return

    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(joint_pos_tensor)
        return

    raise RuntimeError(
        "Robot articulation does not expose a supported joint write API "
        "(write_joint_state_to_sim / set_joint_position_target)."
    )


_HANDOVER7_ALIAS = {
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


def _build_mapped_frame(
    motion_joint_names: list[str],
    robot_joint_names: list[str],
    default_joint_pos: torch.Tensor,
    raw_frames: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Return (N_frames, N_robot_joints) tensor mapped by name from HandOver7 source."""
    src_idx = {name: i for i, name in enumerate(motion_joint_names)}
    mapped = torch.from_numpy(np.tile(default_joint_pos.cpu().numpy(), (raw_frames.shape[0], 1))).float()
    count = 0
    for robot_idx, robot_name in enumerate(robot_joint_names):
        src_name = robot_name if robot_name in src_idx else _HANDOVER7_ALIAS.get(robot_name)
        if src_name and src_name in src_idx:
            mapped[:, robot_idx] = torch.from_numpy(raw_frames[:, src_idx[src_name]]).float()
            count += 1
    print(f"Mapped {count}/{len(robot_joint_names)} joints from HandOver7 → G1", flush=True)
    return mapped.to(device)


def main():
    motion_dir = Path(args_cli.motion_dir)

    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset(seed=args_cli.seed)

    env_unwrapped = env.unwrapped
    if not hasattr(env_unwrapped, "scene"):
        raise RuntimeError("Environment does not expose scene on the unwrapped object.")

    try:
        robot = env_unwrapped.scene["robot"]
    except KeyError as exc:
        raise RuntimeError("Could not find 'robot' articulation in environment scene.") from exc

    robot_joint_names = _get_robot_joint_names(robot)
    device = getattr(robot, "device", "cpu")
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()

    # Load raw NPZ with joint_names metadata for name-based mapping
    npz_path = motion_dir / "HandOver7_unitree_g1.npz"
    if not npz_path.exists():
        npz_path = motion_dir / "HandOver7.npz"

    motion_joint_names: list[str] | None = None
    if npz_path.exists():
        payload = np.load(npz_path, allow_pickle=True)
        if "joint_names" in payload:
            motion_joint_names = [str(n) for n in np.asarray(payload["joint_names"]).tolist()]
        raw_key = next(
            (k for k in ("joint_positions", "joint_pos", "joint_position", "posed_joints", "qpos", "positions") if k in payload),
            None,
        )
        if raw_key is None and len(payload.files) == 1:
            raw_key = payload.files[0]
        raw_frames = np.asarray(payload[raw_key], dtype=np.float32) if raw_key else None
    else:
        raw_frames = None

    if raw_frames is not None and motion_joint_names is not None:
        # Name-based mapping via alias table
        frames_tensor = _build_mapped_frame(motion_joint_names, robot_joint_names, torch.from_numpy(default_joint_pos), raw_frames, device)
        frames = frames_tensor.cpu().numpy()
    else:
        # Fallback: index-based (first N columns)
        frames = _resolve_motion_frames(motion_dir, args_cli.source, robot_joint_names)
        print("[WARN] joint_names not in NPZ; using index-based fallback mapping", flush=True)

    if args_cli.max_frames > 0:
        frames = frames[: args_cli.max_frames]

    print(f"Loaded {frames.shape[0]} frames for {frames.shape[1]} joints.", flush=True)

    dt = 1.0 / max(args_cli.fps, 1e-6)

    try:
        while simulation_app.is_running():
            for i in range(frames.shape[0]):
                t0 = time.monotonic()
                frame_tensor = torch.tensor(frames[i], dtype=torch.float32, device=device).unsqueeze(0)
                _write_joint_positions(robot, frame_tensor)

                # Push articulation buffers into PhysX and render one step.
                env_unwrapped.scene.write_data_to_sim()
                env_unwrapped.sim.step()
                env_unwrapped.scene.update(env_unwrapped.physics_dt)

                if (i + 1) % 60 == 0:
                    print(f"  replay frame {i + 1}/{frames.shape[0]}", flush=True)

                elapsed = time.monotonic() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                if not simulation_app.is_running():
                    break

            if not args_cli.loop:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
