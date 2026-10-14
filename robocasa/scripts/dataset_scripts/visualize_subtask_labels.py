"""Render a demo with its per-frame subtask labels burned onto the video.

This is the "see it as a human" tool: it replays a demo, renders each frame,
and overlays the current subtask (a bank atomic-task name), its stage, the
language instruction, the segment index, and a segment-timeline bar. Demos that
fail the sanity checks get a FLAGGED banner so they stand out during review.

Example:
    python -m robocasa.scripts.dataset_scripts.visualize_subtask_labels \\
        --dataset /mnt/.../AddIceCubes/20250829/demo.hdf5 \\
        --demo demo_1 --out /tmp/demo_1_labeled.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
from termcolor import colored

import robocasa  # noqa: F401
from robocasa.scripts.dataset_scripts.label_subtasks import (
    build_env_from_hdf5,
    label_demo,
    reset_to,
)
from robocasa.utils.subtask_labeling import get_labeler, LABELERS

# RGB colors keyed by subtask stage.
STAGE_COLOR = {
    "pick": (90, 170, 255),
    "place": (110, 220, 130),
    "execute": (255, 170, 70),
    "press": (255, 140, 50),
    "wait": (210, 190, 110),
    "tilt": (220, 120, 255),
    "navigate": (255, 230, 100),
    "done": (210, 210, 210),
}
DEFAULT_COLOR = (255, 255, 255)


def segments_of(idx):
    """Run-length-encode subtask_idx into [(start, end, idx), ...]."""
    segs = []
    s = 0
    for t in range(1, len(idx) + 1):
        if t == len(idx) or idx[t] != idx[s]:
            segs.append((s, t - 1, int(idx[s])))
            s = t
    return segs


def draw_overlay(frame, t, T, result, segs, demo_key, issues):
    """Burn subtask info onto a single RGB frame (in place-ish; returns new)."""
    h, w = frame.shape[:2]
    img = np.ascontiguousarray(frame)
    idx = int(result.subtask_idx[t])
    name = str(result.subtask_name[t])
    stage = str(result.subtask_stage[t])
    lang = str(result.subtask_lang[t])
    color = STAGE_COLOR.get(stage, DEFAULT_COLOR)

    # --- top banner (semi-transparent dark strip) ---
    band = img.copy()
    cv2.rectangle(band, (0, 0), (w, 70), (0, 0, 0), -1)
    img = cv2.addWeighted(band, 0.55, img, 0.45, 0)

    # Show "[stage]" for subtasks that split across stages (pick/place,
    # press/wait); single-action subtasks show just the name.
    label = f"{name} [{stage}]" if stage in ("pick", "place", "press", "wait") else name
    cv2.putText(img, f"{demo_key}   frame {t+1}/{T}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"seg {idx}: {label}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, lang, (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- flagged banner ---
    if issues:
        fb = img.copy()
        cv2.rectangle(fb, (0, 74), (w, 96), (120, 0, 0), -1)
        img = cv2.addWeighted(fb, 0.6, img, 0.4, 0)
        cv2.putText(img, "FLAGGED: " + "; ".join(issues)[: 60], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 220), 1, cv2.LINE_AA)

    # --- bottom segment timeline ---
    bar_y0, bar_y1 = h - 26, h - 10
    for (s, e, sidx) in segs:
        x0 = int(w * s / T)
        x1 = int(w * (e + 1) / T)
        c = STAGE_COLOR.get(str(result.subtask_stage[s]), DEFAULT_COLOR)
        cv2.rectangle(img, (x0, bar_y0), (x1, bar_y1), c, -1)
        cv2.rectangle(img, (x0, bar_y0), (x1, bar_y1), (0, 0, 0), 1)
    # playhead
    px = int(w * t / T)
    cv2.line(img, (px, bar_y0 - 4), (px, bar_y1 + 4), (255, 255, 255), 2)
    return img


def visualize(dataset, demo_key, out_path, camera, height, width, fps):
    dataset = Path(dataset)
    env, env_name = build_env_from_hdf5(dataset, offscreen=True)
    if env_name not in LABELERS:
        raise SystemExit(f"No labeler for {env_name}. Have: {sorted(LABELERS)}")
    labeler = get_labeler(env_name)

    with h5py.File(dataset, "r") as f:
        demo_group = f["data"][demo_key]
        states = demo_group["states"][()]
        ep_meta = json.loads(demo_group.attrs["ep_meta"])
        model_xml = demo_group.attrs["model_file"]

        # Compute labels (re-uses the same replay path as the labeler CLI).
        # Done inside the open-file block so label_demo can read the group.
        print(colored(f"Labeling {demo_key} ...", "yellow"))
        result = label_demo(
            env, labeler, demo_group, env_name=env_name, demo_key=demo_key
        )

    issues = labeler.validate(result)
    segs = segments_of(result.subtask_idx)
    status = colored("PASS", "green") if not issues else colored("FLAGGED", "red")
    print(f"  {status}  {len(segs)} segments")
    if issues:
        for it in issues:
            print(colored(f"    - {it}", "red"))

    # Re-load the demo's model once, then render frame by frame.
    reset_to(env, {"model": model_xml, "states": states[0],
                   "ep_meta": json.dumps(ep_meta)})
    T = states.shape[0]
    writer = imageio.get_writer(out_path, fps=fps)
    print(colored(f"Rendering {T} frames ({camera}) ...", "yellow"))
    for t in range(T):
        reset_to(env, {"states": states[t]})
        frame = env.sim.render(height=height, width=width, camera_name=camera)[::-1]
        writer.append_data(
            draw_overlay(frame, t, T, result, segs, demo_key, issues)
        )
    writer.close()
    env.close()
    print(colored(f"  -> {out_path}", "green"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--demo", type=str, required=True, help="demo key, e.g. demo_1")
    p.add_argument("--out", type=str, default=None, help="output mp4 path")
    p.add_argument("--camera", type=str, default="robot0_agentview_center")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    out = args.out or f"/tmp/{args.demo}_subtask_labeled.mp4"
    visualize(args.dataset, args.demo, out, args.camera,
              args.height, args.width, args.fps)


if __name__ == "__main__":
    main()
