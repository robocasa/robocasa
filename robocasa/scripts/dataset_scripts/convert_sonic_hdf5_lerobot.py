"""Convert Robocasa SONIC HDF5 demos to a LeRobot dataset.

This converter is intentionally separate from ``convert_hdf5_lerobot.py``. The
existing converter is for the PandaOmron embodiment, while SONIC G1 demos have a
different robot state and action layout.

The current SONIC Robocasa collector stores the low-level 43D whole-body command
received by the simulator. It does not store the 64D SONIC latent
``action.motion_token`` expected by the official Isaac-GR00T N1.7
``UNITREE_G1_SONIC`` pipeline. This script therefore writes a faithful LeRobot
dataset with ``action.wbc`` / ``action`` as the recorded 43D command and records
that limitation in ``meta/conversion_warnings.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from robocasa.utils.lerobot_utils import (
    LerobotDatasetWrapper,
    calculate_dataset_statistics,
    save_dataset_meta,
)
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils


DEFAULT_TARGET_FPS = 50
THREE_CAMERA_NAMES = [
    "robot0_head_camera",
    "robot0_left_wrist_camera",
    "robot0_right_wrist_camera",
]
THREE_IMAGE_KEYS = ["ego_view", "left_wrist", "right_wrist"]
DEFAULT_CAMERA_NAMES = list(THREE_CAMERA_NAMES)
DEFAULT_IMAGE_KEYS = list(THREE_IMAGE_KEYS)

VIRTUAL_WRIST_CAMERA_SPECS = {
    "robot0_left_wrist_camera": {
        "body": "robot0_left_wrist_yaw_link",
        "pos": "0.08 0.02 0.03",
        "euler": "0 -0.8 -1.57",
        "fovy": "75",
    },
    "robot0_right_wrist_camera": {
        "body": "robot0_right_wrist_yaw_link",
        "pos": "0.08 -0.02 0.03",
        "euler": "0 -0.8 -1.57",
        "fovy": "75",
    },
}

VIDEO_INFO = {
    "video.fps": DEFAULT_TARGET_FPS,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}


def _sorted_demo_keys(hdf5_file: h5py.File) -> list[str]:
    demos = list(hdf5_file["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[-1]))


def _decode_json_attr(value, default=None):
    if default is None:
        default = {}
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return default
    return json.loads(value)


def _task_from_ep_meta(ep_meta) -> str:
    meta = _decode_json_attr(ep_meta)
    return meta.get("lang") or meta.get("task") or "demo"


def _source_fps(raw_file: h5py.File, override: float | None) -> float:
    if override is not None:
        return float(override)

    runtime = _decode_json_attr(raw_file["data"].attrs.get("sonic_runtime"))
    if "control_freq" in runtime:
        return float(runtime["control_freq"])

    env_args = _decode_json_attr(raw_file["data"].attrs.get("env_args"))
    env_kwargs = env_args.get("env_kwargs", {})
    if "control_freq" in env_kwargs:
        return float(env_kwargs["control_freq"])

    return 200.0


def _frame_stride(source_fps: float, target_fps: int) -> int:
    if source_fps <= 0:
        return 1
    return max(1, int(round(source_fps / float(target_fps))))


def _copy_with_jsonable_attrs(attrs) -> dict:
    out = {}
    for key, value in dict(attrs).items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, bytes):
            out[key] = value.decode("utf-8")
        else:
            out[key] = value
    return out


def _camera_names_in_model(model_xml: str) -> set[str]:
    root = ET.fromstring(model_xml)
    return {cam.attrib.get("name") for cam in root.iter("camera") if cam.attrib.get("name")}


def _add_camera_if_missing(root, body_name: str, camera_name: str, pos: str, euler: str):
    if any(cam.attrib.get("name") == camera_name for cam in root.iter("camera")):
        return
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(
            f"Cannot inject camera '{camera_name}': body '{body_name}' not found in model XML."
        )
    ET.SubElement(
        body,
        "camera",
        {
            "name": camera_name,
            "mode": "fixed",
            "pos": pos,
            "euler": euler,
            "fovy": "75",
        },
    )


def _inject_virtual_wrist_cameras(model_xml: str) -> str:
    root = ET.fromstring(model_xml)
    for camera_name, spec in VIRTUAL_WRIST_CAMERA_SPECS.items():
        _add_camera_if_missing(
            root,
            spec["body"],
            camera_name,
            spec["pos"],
            spec["euler"],
        )
    return ET.tostring(root, encoding="unicode")


def _prepare_model_xml(model_xml: str, args) -> str:
    required = set(args.camera_names)
    missing = required - _camera_names_in_model(model_xml)
    if missing and args.inject_virtual_wrist_cameras:
        model_xml = _inject_virtual_wrist_cameras(model_xml)
        missing = required - _camera_names_in_model(model_xml)
    if missing:
        raise ValueError(
            "Camera(s) missing from model XML: "
            + ", ".join(sorted(missing))
            + ". Use --inject-virtual-wrist-cameras for the default wrist cameras, "
            + "or pass camera names present in the dataset model."
        )
    return model_xml


def _whole_body_state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_joint_pos"], dtype=np.float64),
            np.asarray(obs["robot0_left_gripper_qpos"], dtype=np.float64),
            np.asarray(obs["robot0_right_gripper_qpos"], dtype=np.float64),
        ]
    )


def _eef_state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_left_eef_pos"], dtype=np.float64),
            np.asarray(obs["robot0_left_eef_quat"], dtype=np.float64),
            np.asarray(obs["robot0_right_eef_pos"], dtype=np.float64),
            np.asarray(obs["robot0_right_eef_quat"], dtype=np.float64),
        ]
    )


def _root_orientation_from_obs(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_base_quat"], dtype=np.float64)


def _build_task_to_id(paths: list[Path]) -> dict[str, int]:
    task_to_id: dict[str, int] = {}
    for path in paths:
        with h5py.File(path, "r") as raw_file:
            for demo in _sorted_demo_keys(raw_file):
                task = _task_from_ep_meta(raw_file["data"][demo].attrs.get("ep_meta"))
                if task not in task_to_id:
                    task_to_id[task] = len(task_to_id)
    return task_to_id


def _find_first_demo(paths: list[Path]):
    for path in paths:
        with h5py.File(path, "r") as raw_file:
            demos = _sorted_demo_keys(raw_file)
            if demos:
                return path, demos[0]
    raise ValueError("No demos found in input HDF5 dataset(s).")


def _make_env_from_hdf5(path: Path, args):
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=str(path))
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.infer_rewards,
    )


def _reset_to_demo_frame(env, demo_data, frame_index: int, args) -> dict:
    model_xml = _prepare_model_xml(demo_data.attrs["model_file"], args)
    return env.reset_to(
        {
            "states": demo_data["states"][frame_index],
            "model": model_xml,
            "ep_meta": demo_data.attrs.get("ep_meta", None),
        }
    )


def _probe_shapes(paths: list[Path], args):
    first_path, first_demo = _find_first_demo(paths)
    env = _make_env_from_hdf5(first_path, args)
    try:
        with h5py.File(first_path, "r") as raw_file:
            demo_data = raw_file["data"][first_demo]
            obs = _reset_to_demo_frame(env, demo_data, 0, args)
            state_dim = _whole_body_state_from_obs(obs).shape[0]
            eef_dim = _eef_state_from_obs(obs).shape[0]
            root_dim = _root_orientation_from_obs(obs).shape[0]
            action_dim = demo_data["actions"].shape[1]
            action_dict_shapes = {}
            if args.include_action_dict and "action_dict" in demo_data:
                for key, value in demo_data["action_dict"].items():
                    action_dict_shapes[key] = value.shape[1:]
    finally:
        _close_env(env)
    return state_dim, eef_dim, root_dim, action_dim, action_dict_shapes


def _build_features(args, state_dim, eef_dim, root_dim, action_dim, action_dict_shapes):
    image_shape = (args.camera_height, args.camera_width, 3)
    video_info = dict(VIDEO_INFO)
    video_info["video.fps"] = args.target_fps

    features = {}
    for image_key in args.image_keys:
        features[f"observation.images.{image_key}"] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
            "video_info": video_info,
        }

    features.update(
        {
            "observation.state": {
                "dtype": "float64",
                "shape": (state_dim,),
                "names": "sonic_g1_joint_state",
            },
            "observation.eef_state": {
                "dtype": "float64",
                "shape": (eef_dim,),
                "names": "left_right_eef_pose",
            },
            "observation.root_orientation": {
                "dtype": "float64",
                "shape": (root_dim,),
                "names": "base_quat",
            },
            "action.wbc": {
                "dtype": "float64",
                "shape": (action_dim,),
                "names": "sonic_wbc_command",
            },
            "action": {
                "dtype": "float64",
                "shape": (action_dim,),
                "names": "sonic_wbc_command",
            },
            "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
            "next.reward": {"dtype": "float32", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
        }
    )

    if args.add_zero_motion_token:
        features["action.motion_token"] = {
            "dtype": "float64",
            "shape": (64,),
            "names": "missing_motion_token_zero_filled",
        }

    for key, shape in action_dict_shapes.items():
        features[f"action_dict.{key}"] = {
            "dtype": "float64",
            "shape": tuple(shape),
            "names": key,
        }

    return features


def _write_modality_json(output_path: Path, args, state_dim: int, eef_dim: int, action_dim: int):
    modality = {
        "state": {
            "joint_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": state_dim,
            },
            "left_wrist_pose": {
                "original_key": "observation.eef_state",
                "start": 0,
                "end": 7,
            },
            "right_wrist_pose": {
                "original_key": "observation.eef_state",
                "start": 7,
                "end": eef_dim,
            },
            "root_orientation": {
                "original_key": "observation.root_orientation",
                "start": 0,
                "end": 4,
                "rotation_type": "quaternion",
            },
        },
        "action": {
            "wbc_joint_target": {
                "original_key": "action.wbc",
                "start": 0,
                "end": action_dim,
            }
        },
        "video": {
            image_key: {"original_key": f"observation.images.{image_key}"}
            for image_key in args.image_keys
        },
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description"
            }
        },
    }
    if args.add_zero_motion_token:
        modality["action"]["motion_token_zero_filled"] = {
            "original_key": "action.motion_token",
            "start": 0,
            "end": 64,
        }
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)


def _write_embodiment_json(output_path: Path, args):
    embodiment = {
        "robot_name": "SonicG1",
        "robot_type": "SonicG1",
        "record_frequency": float(args.target_fps),
        "body_controller_frequency": float(args.target_fps),
        "hand_controller_frequency": float(args.target_fps),
        "embodiment_tag": "robocasa_sonic_g1_raw_wbc",
    }
    with open(output_path / "meta" / "embodiment.json", "w") as f:
        json.dump(embodiment, f, indent=4)


def _write_camera_metadata(output_path: Path, args):
    cameras = {}
    for camera_name, image_key in zip(args.camera_names, args.image_keys):
        entry = {
            "image_key": image_key,
            "camera_name": camera_name,
            "feature_key": f"observation.images.{image_key}",
        }
        if camera_name in VIRTUAL_WRIST_CAMERA_SPECS:
            entry.update(
                {
                    "injected_when_missing": bool(args.inject_virtual_wrist_cameras),
                    "mujoco_body": VIRTUAL_WRIST_CAMERA_SPECS[camera_name]["body"],
                    "pos": VIRTUAL_WRIST_CAMERA_SPECS[camera_name]["pos"],
                    "euler": VIRTUAL_WRIST_CAMERA_SPECS[camera_name]["euler"],
                    "fovy": VIRTUAL_WRIST_CAMERA_SPECS[camera_name]["fovy"],
                }
            )
        cameras[image_key] = entry
    with open(output_path / "meta" / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=4)


def _write_conversion_warnings(output_path: Path, args, source_fps_values: dict[str, float]):
    warnings = [
        "The source HDF5 stores the 43D low-level SONIC whole-body command, not the "
        "64D SONIC latent action.motion_token expected by the official Isaac-GR00T "
        "N1.7 UNITREE_G1_SONIC pipeline.",
        "Use this dataset as a faithful LeRobot export of Robocasa SONIC demos, or "
        "adapt the GR00T modality/action head before training.",
    ]
    if args.add_zero_motion_token:
        warnings.append(
            "action.motion_token was explicitly zero-filled because "
            "--add-zero-motion-token was set. This is only a compatibility aid."
        )
    metadata = {
        "warnings": warnings,
        "source_fps": source_fps_values,
        "target_fps": args.target_fps,
        "frame_stride_rule": "round(source_fps / target_fps)",
        "camera_names": args.camera_names,
        "image_keys": args.image_keys,
        "inject_virtual_wrist_cameras": args.inject_virtual_wrist_cameras,
    }
    with open(output_path / "meta" / "conversion_warnings.json", "w") as f:
        json.dump(metadata, f, indent=4)


def _save_extra_demo_info(output_path: Path, demo_data, global_ep_idx: int, model_xml: str):
    ep_dir = output_path / "extras" / f"episode_{global_ep_idx:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    payload = {"states": demo_data["states"][:]}
    if "states_integration" in demo_data:
        payload["states_integration"] = demo_data["states_integration"][:]
    np.savez_compressed(ep_dir / "states.npz", **payload)

    ep_meta = _decode_json_attr(demo_data.attrs.get("ep_meta"))
    with open(ep_dir / "ep_meta.json", "w") as f:
        json.dump(ep_meta, f, indent=4)

    root = ET.fromstring(model_xml)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
        f.write(xml_bytes)


def _write_source_metadata(output_path: Path, paths: list[Path]):
    extras = output_path / "extras"
    extras.mkdir(parents=True, exist_ok=True)
    sources = []
    for path in paths:
        with h5py.File(path, "r") as raw_file:
            sources.append(
                {
                    "path": str(path),
                    "data_attrs": _copy_with_jsonable_attrs(raw_file["data"].attrs),
                    "num_demos": len(_sorted_demo_keys(raw_file)),
                }
            )
    with open(extras / "source_datasets.json", "w") as f:
        json.dump(sources, f, indent=4)
    if len(paths) == 1:
        with h5py.File(paths[0], "r") as raw_file:
            save_dataset_meta(output_path, raw_file)


def _close_env(env):
    base_env = getattr(env, "env", None)
    if base_env is not None and hasattr(base_env, "close"):
        base_env.close()


def _action_dict_frame(demo_data, key: str, frame_index: int, shape) -> np.ndarray:
    if "action_dict" not in demo_data or key not in demo_data["action_dict"]:
        return np.zeros(shape, dtype=np.float64)
    return np.asarray(demo_data["action_dict"][key][frame_index], dtype=np.float64)


def _convert_demo(
    dataset,
    env,
    demo_data,
    args,
    task_to_id: dict[str, int],
    action_dict_shapes: dict[str, tuple],
    source_fps: float,
    global_ep_idx: int,
):
    stride = _frame_stride(source_fps, args.target_fps)
    demo_len = demo_data["actions"].shape[0]
    frame_indices = list(range(0, demo_len, stride))
    if args.max_frames_per_demo is not None:
        frame_indices = frame_indices[: args.max_frames_per_demo]
    if not frame_indices:
        return 0

    task = _task_from_ep_meta(demo_data.attrs.get("ep_meta"))
    model_xml = _prepare_model_xml(demo_data.attrs["model_file"], args)
    _save_extra_demo_info(dataset.root, demo_data, global_ep_idx, model_xml)

    for local_i, frame_index in enumerate(frame_indices):
        state = {"states": demo_data["states"][frame_index]}
        if local_i == 0:
            state["model"] = model_xml
            state["ep_meta"] = demo_data.attrs.get("ep_meta", None)
        obs = env.reset_to(state)
        if obs is None:
            obs = env.get_observation()

        action = np.asarray(demo_data["actions"][frame_index], dtype=np.float64)
        frame = {
            "observation.state": _whole_body_state_from_obs(obs),
            "observation.eef_state": _eef_state_from_obs(obs),
            "observation.root_orientation": _root_orientation_from_obs(obs),
            "action.wbc": action,
            "action": action,
            "annotation.human.task_description": np.array(
                [task_to_id[task]], dtype=np.int64
            ),
            "next.reward": np.array(
                [env.get_reward() if args.infer_rewards else 0.0], dtype=np.float32
            ),
            "next.done": np.array([local_i == len(frame_indices) - 1], dtype=bool),
        }

        for camera_name, image_key in zip(args.camera_names, args.image_keys):
            frame[f"observation.images.{image_key}"] = obs[f"{camera_name}_image"]

        if args.add_zero_motion_token:
            frame["action.motion_token"] = np.zeros(64, dtype=np.float64)

        for key, shape in action_dict_shapes.items():
            frame[f"action_dict.{key}"] = _action_dict_frame(
                demo_data, key, frame_index, shape
            )

        dataset.add_frame(frame, task=task)

    dataset.save_episode()
    return len(frame_indices)


def _resolve_output_path(paths: list[Path], output_path: str | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    if len(paths) == 1:
        return paths[0].parent / "lerobot_sonic"
    return paths[0].parent.parent / "lerobot_sonic_merged"


def convert(args):
    raw_paths = [Path(p).expanduser().resolve() for p in args.raw_dataset_paths]
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.three_camera:
        args.camera_names = THREE_CAMERA_NAMES
        args.image_keys = THREE_IMAGE_KEYS
        args.inject_virtual_wrist_cameras = True

    if len(args.camera_names) != len(args.image_keys):
        raise ValueError("--camera-names and --image-keys must have the same length.")

    output_path = _resolve_output_path(raw_paths, args.output_path)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    task_to_id = _build_task_to_id(raw_paths)
    state_dim, eef_dim, root_dim, action_dim, action_dict_shapes = _probe_shapes(
        raw_paths, args
    )

    features = _build_features(
        args, state_dim, eef_dim, root_dim, action_dim, action_dict_shapes
    )
    dataset = LerobotDatasetWrapper.create(
        repo_id=output_path.name,
        root=output_path,
        robot_type="SonicG1",
        fps=args.target_fps,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    _write_source_metadata(output_path, raw_paths)

    source_fps_values = {}
    global_ep_idx = 0
    total_frames = 0

    for path in raw_paths:
        env = _make_env_from_hdf5(path, args)
        try:
            with h5py.File(path, "r") as raw_file:
                source_fps = _source_fps(raw_file, args.source_fps)
                source_fps_values[str(path)] = source_fps
                demos = _sorted_demo_keys(raw_file)
                if args.max_demos is not None:
                    remaining = max(0, args.max_demos - global_ep_idx)
                    demos = demos[:remaining]
                for demo in tqdm(demos, desc=f"Converting {path.name}"):
                    frames = _convert_demo(
                        dataset=dataset,
                        env=env,
                        demo_data=raw_file["data"][demo],
                        args=args,
                        task_to_id=task_to_id,
                        action_dict_shapes=action_dict_shapes,
                        source_fps=source_fps,
                        global_ep_idx=global_ep_idx,
                    )
                    if frames > 0:
                        total_frames += frames
                        global_ep_idx += 1
                    if args.max_demos is not None and global_ep_idx >= args.max_demos:
                        break
        finally:
            _close_env(env)
        if args.max_demos is not None and global_ep_idx >= args.max_demos:
            break

    _write_modality_json(output_path, args, state_dim, eef_dim, action_dim)
    _write_embodiment_json(output_path, args)
    _write_camera_metadata(output_path, args)
    _write_conversion_warnings(output_path, args, source_fps_values)

    if not args.skip_stats:
        parquet_paths = sorted((output_path / "data").glob("*/episode_*.parquet"))
        stats = calculate_dataset_statistics(parquet_paths)
        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    images_dir = output_path / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    print(f"[sonic-lr] wrote {global_ep_idx} episode(s), {total_frames} frame(s)")
    print(f"[sonic-lr] output: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dataset-paths",
        nargs="+",
        required=True,
        help="One or more Robocasa SONIC demo.hdf5 files.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output LeRobot directory. Defaults to lerobot_sonic or lerobot_sonic_merged.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory.")
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override source FPS. Defaults to sonic_runtime/control_freq in the HDF5.",
    )
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_CAMERA_NAMES))
    parser.add_argument("--image-keys", nargs="+", default=list(DEFAULT_IMAGE_KEYS))
    parser.add_argument(
        "--three-camera",
        action="store_true",
        help="Use head + virtual left/right wrist cameras and image keys ego_view/left_wrist/right_wrist.",
    )
    parser.add_argument(
        "--inject-virtual-wrist-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject fixed wrist cameras into each episode XML if missing.",
    )
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument(
        "--infer-rewards",
        action="store_true",
        help="Call env.reward() for each frame. Defaults to zero reward and last-frame done.",
    )
    parser.add_argument(
        "--no-action-dict",
        action="store_false",
        dest="include_action_dict",
        help="Do not export action_dict.* columns even if present in the HDF5.",
    )
    parser.set_defaults(include_action_dict=True)
    parser.add_argument(
        "--add-zero-motion-token",
        action="store_true",
        help="Add a zero-filled 64D action.motion_token column. Not valid supervision.",
    )
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-frames-per-demo", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    convert(get_args())
