"""Offline per-frame subtask labeling for composite task demos.

Replays each demo's sim states one timestep at a time and evaluates task-specific
predicates to produce per-frame subtask labels:
  - subtask_idx: fine-grained index (e.g. per-cube for AddIceCubes)
  - subtask_phase: doc-set phase index
  - subtask_lang: per-frame natural-language string

Labels are written back into the source hdf5 under each demo group as datasets
named "subtask_idx", "subtask_phase", "subtask_lang", plus an attribute
"subtask_phases" on the demo group enumerating the phase strings.

Example:
    python -m robocasa.scripts.dataset_scripts.label_subtasks \\
        --dataset /mnt/.../AddIceCubes/20250829/demo.hdf5 \\
        --n 1 --demo demo_1 --dry-run
"""

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from termcolor import colored
from tqdm import tqdm

import robosuite

import robocasa  # noqa: F401  (registers envs with robosuite)
from robocasa.utils.subtask_labeling import get_labeler, LABELERS
from robocasa.utils.subtask_labeling.manual_overrides import get_override


def reset_to(env, state):
    """Reset env to a flattened sim state and/or a model xml.

    Inlined copy of robocasa.scripts.dataset_scripts.playback_dataset.reset_to so
    that this script does not require the lerobot dependency.
    """
    if "model" in state:
        ep_meta = json.loads(state["ep_meta"]) if state.get("ep_meta") else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        # env.reset() re-samples a fresh random scene whose object placement can
        # occasionally fail (kitchen._load_model retries 50x then raises RuntimeError).
        # reset_from_xml_string() below overrides it with the recorded model, so the
        # sampled scene is irrelevant -- retry until a placeable scene is drawn (flaky
        # RNG, not a bad demo).
        for _reset_attempt in range(30):
            try:
                env.reset()
                break
            except RuntimeError:
                if _reset_attempt == 29:
                    raise
                continue
        # Legacy datasets recorded a mujoco compiler flag <flag sleep="enable"/>;
        # current mujoco rejects the 'sleep' attribute ("Schema violation:
        # unrecognized attribute"). Strip it here so older splits replay (the
        # current <flag multiccd="enable"/> is already present alongside it).
        model = state["model"]
        if isinstance(model, bytes):
            model = model.decode()
        if 'sleep="' in model:
            model = re.sub(r'\s*sleep="[^"]*"', "", model)
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(model)
        else:
            xml = env.edit_model_xml(model)
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return None


def build_env_from_hdf5(hdf5_path: Path, offscreen: bool = False):
    """Create a robosuite env from the hdf5's env_args attribute.

    Args:
        offscreen: if True, enable the offscreen renderer (needed to render
            frames for the subtask-label overlay video).

    Returns the robosuite env and the env_name string.
    """
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    env_kwargs = env_args.get("env_kwargs", env_args)
    env_kwargs = dict(env_kwargs)
    env_kwargs["env_name"] = env_name
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = offscreen
    env_kwargs["use_camera_obs"] = False
    # The dataset was recorded with translucent_robot=true for video purposes;
    # not needed when we just want predicates over sim state.
    env_kwargs["translucent_robot"] = False
    # Labeling only replays sim states (set_state_from_flattened) and never calls
    # env.step(), so the controller is irrelevant. Drop the recorded
    # controller_configs to avoid robocasa<->robosuite version mismatches (e.g.
    # the JOINT_VELOCITY_LEGACY mobile-base controller name); the robot default
    # controller is used instead.
    env_kwargs.pop("controller_configs", None)
    # Legacy split name: some early datasets recorded "train"; current code
    # expects "pretrain"/"target"/None.
    if env_kwargs.get("obj_instance_split") == "train":
        env_kwargs["obj_instance_split"] = "pretrain"
    env = robosuite.make(**env_kwargs)
    return env, env_name


def label_demo(env, labeler, demo_group, env_name=None, demo_key=None):  # noqa: F821
    states = demo_group["states"][()]
    T = states.shape[0]

    # Manual override: a human-reviewed segmentation for a previously flagged
    # demo. Bypasses the auto-resolve (no replay needed) and is marked as
    # manually labeled so it leaves the "investigate" queue.
    override = get_override(env_name, demo_key) if env_name and demo_key else None
    # Overrides are keyed by (env_name, demo_key), but that key collides across
    # the pretrain/target splits, which hold *different* trajectories under the
    # same demo_key. An override's final frame is its trajectory length; if it
    # doesn't match this demo's T, the override belongs to the other split --
    # ignore it and fall back to the auto-resolve.
    if override is not None and int(override[-1][0]) != int(T):
        override = None
    if override is not None:
        labeler.expected_done_idx = len(override) - 1
        result = labeler.build_result(T, override, debug={"manual": True})
        result.review = ["manually labeled (was flagged) -- human-reviewed"]
        return result

    ep_meta = json.loads(demo_group.attrs["ep_meta"])
    model_xml = demo_group.attrs["model_file"]

    # Initial reset to load the demo's specific model + ep_meta.
    initial_state = {
        "model": model_xml,
        "states": states[0],
        "ep_meta": json.dumps(ep_meta),
    }
    reset_to(env, initial_state)
    labeler.setup(env, ep_meta)

    history = []
    for t in range(T):
        reset_to(env, {"states": states[t]})
        history.append(labeler.per_step(env, t))

    return labeler.resolve(history)


SUBTASK_KEYS = ("subtask_idx", "subtask_name", "subtask_stage", "subtask_lang")
SUBTASK_ATTRS = ("subtask_issues", "subtask_review")


def revert_labels(hdf5_path: Path) -> None:
    """Strip every subtask_* dataset/attr added by this script. The script is
    strictly additive (it only creates these keys and never modifies original
    demo data), so deleting them restores the file's pre-labeling state."""
    n_demos = 0
    n_datasets = 0
    n_attrs = 0
    with h5py.File(hdf5_path, "r+") as f:
        for key in sorted(f["data"].keys()):
            g = f["data"][key]
            touched = False
            for dk in SUBTASK_KEYS:
                if dk in g:
                    del g[dk]
                    n_datasets += 1
                    touched = True
            for ak in SUBTASK_ATTRS:
                if ak in g.attrs:
                    del g.attrs[ak]
                    n_attrs += 1
                    touched = True
            if touched:
                n_demos += 1
    print(colored(
        f"Reverted {n_demos} demo(s): removed {n_datasets} dataset(s) and "
        f"{n_attrs} attr(s). Note: hdf5 reclaims space on next h5repack.",
        "green",
    ))


def write_labels(demo_group, result, labeler, issues, overwrite: bool):
    for key in SUBTASK_KEYS:
        if key in demo_group:
            if not overwrite:
                raise RuntimeError(
                    f"{demo_group.name}/{key} exists. Pass --overwrite to replace."
                )
            del demo_group[key]
    demo_group.create_dataset("subtask_idx", data=result.subtask_idx)
    # h5py needs vlen str for arrays of Python strs.
    str_dt = h5py.string_dtype(encoding="utf-8")
    for key in ("subtask_name", "subtask_stage", "subtask_lang"):
        demo_group.create_dataset(
            key, data=getattr(result, key).astype(object), dtype=str_dt
        )
    # Hard issues: a non-empty list means the demo failed sanity checks and
    # needs investigation. Empty == passed.
    demo_group.attrs["subtask_issues"] = json.dumps(issues)
    # Soft review notes: auto-detected-but-valid variants (e.g. tilt-pour) that
    # warrant a light human confirmation rather than investigation.
    demo_group.attrs["subtask_review"] = json.dumps(list(result.review))


# ---------------------------------------------------------------------------
# Safe write pipeline
#
# The live dataset is NEVER opened r+ during labeling. We copy it to a working
# file, mutate the copy, h5repack it (reclaiming freed space so the vlen-string
# heap can't bloat across repeated runs), verify it opens cleanly with every
# selected demo labeled, and only then atomically swap it in -- preserving the
# previous good file as <name>.prev.hdf5. An interrupted or killed run can
# therefore only ever damage the throwaway working copy; the source stays
# intact. This exists because repeated in-place --overwrite of vlen-string
# datasets + a killed mid-write once corrupted a dataset (heap-address overflow:
# metadata pointing past the file's own end).
# ---------------------------------------------------------------------------

def _open_ok(path):
    """Canary: open read-only and touch every demo's core + label-membership so
    structural corruption (e.g. heap-address overflow) surfaces here rather than
    mid-write. Returns (ok, msg)."""
    try:
        with h5py.File(path, "r") as f:
            for k in f["data"].keys():
                g = f["data"][k]
                _ = g["states"].shape
                _ = "subtask_stage" in g  # membership read trips overflow bugs
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def _verify_labeled(path, demo_keys):
    """Confirm a freshly-labeled file opens cleanly and every selected demo
    carries readable labels (forces a vlen-heap read). Returns (ok, msg)."""
    try:
        with h5py.File(path, "r") as f:
            data = f["data"]
            for k in demo_keys:
                g = data[k]
                _ = g["states"].shape
                if "subtask_stage" not in g:
                    return False, f"{k} missing labels after write"
                _ = g["subtask_stage"][()]
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def _find_h5repack():
    exe = shutil.which("h5repack")
    if exe:
        return exe
    guess = os.path.join(sys.prefix, "bin", "h5repack")
    return guess if os.path.exists(guess) else None


def _repack(src, dst):
    """h5repack src -> dst to reclaim freed space. Returns True on success."""
    exe = _find_h5repack()
    if exe is None:
        return False
    try:
        subprocess.run(
            [exe, str(src), str(dst)], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def safe_mutate(src, mutate_fn, verify_fn):
    """Atomically apply mutate_fn to src via a verified, repacked working copy.

    mutate_fn(work_path): performs the in-place edits on the COPY.
    verify_fn(work_path) -> (ok, msg): integrity gate before promotion.

    On success: src holds the new file and <src>.prev.hdf5 is the prior good
    file. On any failure/interrupt: src is untouched and the working copy is
    left behind for inspection.
    """
    src = Path(src)
    work = src.with_name(src.stem + ".inprogress.hdf5")
    packed = src.with_name(src.stem + ".packed.hdf5")
    prev = src.with_name(src.stem + ".prev.hdf5")

    ok, msg = _open_ok(src)
    if not ok:
        raise SystemExit(colored(
            f"[safe] refusing to write: source won't open cleanly ({msg}). "
            f"Recover it before labeling.", "red"))

    for pth in (work, packed):
        if pth.exists():
            pth.unlink()
    print(colored(f"[safe] copying source -> working copy ...", "yellow"))
    shutil.copyfile(src, work)

    mutate_fn(work)

    final = work
    if _repack(work, packed):
        final = packed
        print(colored("[safe] repacked working copy (reclaimed free space)", "yellow"))
    else:
        print(colored("[safe] h5repack unavailable -- promoting unpacked copy "
                      "(safe, but run h5repack periodically)", "yellow"))

    ok, msg = verify_fn(final)
    if not ok:
        raise SystemExit(colored(
            f"[safe] verification FAILED ({msg}); source left UNTOUCHED. "
            f"Working copy kept at {final}", "red"))

    # Back up first (src stays intact the whole time), then atomic same-fs swap.
    print(colored(f"[safe] backing up current -> {prev.name}", "yellow"))
    if prev.exists():
        prev.unlink()
    shutil.copyfile(src, prev)
    os.replace(final, src)  # atomic on the same filesystem
    for pth in (work, packed):
        if pth.exists():
            try:
                pth.unlink()
            except OSError:
                pass
    print(colored(f"[safe] promoted -> {src.name}   (backup: {prev.name})", "green"))


def print_segments(result, max_segments: int = 20):
    """Pretty-print run-length segments of subtask_idx for quick inspection."""
    idx = result.subtask_idx
    name = result.subtask_name
    stage = result.subtask_stage
    lang = result.subtask_lang
    T = len(idx)
    segments = []
    start = 0
    for t in range(1, T + 1):
        if t == T or idx[t] != idx[start]:
            segments.append(
                (start, t - 1, int(idx[start]), str(name[start]),
                 str(stage[start]), str(lang[start]))
            )
            start = t
    print(colored(f"  {len(segments)} segment(s):", "cyan"))
    for s in segments[:max_segments]:
        # "[stage]" only for pick/place phases; single-action subtasks show name.
        label = f"{s[3]} [{s[4]}]" if s[4] in ("pick", "place") else s[3]
        print(f"    t={s[0]:4d}..{s[1]:4d}  idx={s[2]:2d}  {label}  {s[5]}")
    if len(segments) > max_segments:
        print(f"    ... ({len(segments) - max_segments} more)")


# ---------------------------------------------------------------------------
# Parallel compute
#
# Labeling is CPU-bound and embarrassingly parallel across demos: each demo is
# an independent replay + predicate eval. We fan the COMPUTE out over a process
# pool (each worker builds its own env once and reads the source read-only),
# then the main process does the (hdf5-unsafe) WRITE serially via the safe
# pipeline. Workers read the untouched source; the write goes to the working
# copy -- never the same file concurrently. maxtasksperchild recycles a worker
# periodically so the known per-demo env-rebuild leak can't grow unbounded.
# ---------------------------------------------------------------------------

_WORKER: dict = {}
_WORKER_MAXTASKS = 25


def _init_worker(dataset_path_str: str, env_name: str) -> None:
    global _WORKER
    env, _ = build_env_from_hdf5(Path(dataset_path_str))
    _WORKER = {
        "env": env,
        "labeler": get_labeler(env_name),
        "env_name": env_name,
        "data": h5py.File(dataset_path_str, "r"),
    }


def _worker_compute(demo_key: str):
    """Compute (result, issues, review) for one demo in a pool worker."""
    w = _WORKER
    g = w["data"]["data"][demo_key]
    last_err = None
    for _ in range(3):
        try:
            result = label_demo(
                w["env"], w["labeler"], g, env_name=w["env_name"], demo_key=demo_key
            )
            issues = w["labeler"].validate(result)
            return demo_key, result, issues, list(result.review)
        except RuntimeError as e:
            # Robosuite intermittently fails to sample a valid object placement
            # on reset ("Ran _load_model() N times but could not initialize
            # task!"). It is RNG/timing-dependent (a different demo each run, and
            # the same data labels fine on a fresh attempt), so rebuild the env
            # and retry rather than crashing the whole batch.
            last_err = e
            try:
                env, _ = build_env_from_hdf5(Path(w["data"].filename))
                w["env"] = env
            except Exception:
                pass
    raise last_err


def parallel_compute(dataset_path, env_name, demo_keys, workers):
    """Compute labels for all demos across `workers` processes (read-only on the
    source). Returns {demo_key: (result, issues, review)}, computed identically
    to the serial path. Falls back to serial on import-pickling problems."""
    ctx = multiprocessing.get_context("spawn")
    results = {}
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(str(dataset_path), env_name),
        maxtasksperchild=_WORKER_MAXTASKS,
    ) as pool:
        for key, result, issues, review in tqdm(
            pool.imap(_worker_compute, demo_keys, chunksize=1),
            total=len(demo_keys),
            desc=f"labeling ({workers} workers)",
        ):
            results[key] = (result, issues, review)
    return results


def _label_loop(data, demo_keys, env, env_name, labeler, write, overwrite,
                results=None):
    """Run the per-demo label/validate/(write) loop against an open `data` group.
    Returns (passed, review_q, flagged) and prints a summary.

    If `results` is given ({key: (result, issues, review)} from parallel_compute),
    those precomputed labels are used instead of replaying here -- env/labeler may
    then be None (write_labels does not use the labeler)."""
    # Three outcomes: PASS (clean), REVIEW (auto-detected-but-valid variant,
    # light confirmation), FLAGGED (hard sanity failure, investigate).
    passed, review_q, flagged = [], [], []
    for key in demo_keys:
        print(colored(f"\n[{key}]", "yellow"))
        if results is not None:
            result, issues, review = results[key]
        else:
            result = label_demo(
                env, labeler, data[key], env_name=env_name, demo_key=key
            )
            issues = labeler.validate(result)
            review = list(result.review)
        print_segments(result)

        if issues:
            flagged.append(key)
            print(colored("  FLAGGED (investigate):", "red"))
            for issue in issues:
                print(colored(f"    - {issue}", "red"))
        elif review:
            review_q.append(key)
            print(colored("  REVIEW (confirm auto-detected variant):", "magenta"))
            for note in review:
                print(colored(f"    - {note}", "magenta"))
        else:
            passed.append(key)
            print(colored("  PASS (sanity checks ok)", "green"))

        if write:
            write_labels(data[key], result, labeler, issues, overwrite=overwrite)
            print(colored("  -> wrote labels", "green"))

    # Summary: which demos need a human to look at them.
    total = len(passed) + len(review_q) + len(flagged)
    print(colored(f"\n{'=' * 60}", "yellow"))
    print(
        colored(
            f"SUMMARY: {len(passed)}/{total} passed, "
            f"{len(review_q)}/{total} for review (confirm), "
            f"{len(flagged)}/{total} flagged (investigate)",
            "yellow",
        )
    )
    if review_q:
        print(colored(f"  review demos: {', '.join(review_q)}", "magenta"))
    if flagged:
        print(colored(f"  flagged demos: {', '.join(flagged)}", "red"))
    return passed, review_q, flagged


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to aggregated demo.hdf5",
    )
    p.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Specific demo key to label (e.g. demo_1). Default: all demos.",
    )
    p.add_argument(
        "--n", type=int, default=None, help="Limit to first N demos."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute labels and print summary but do not write back.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subtask_* datasets (only relevant with "
             "--unsafe-inplace; the safe pipeline always replaces).",
    )
    p.add_argument(
        "--revert",
        action="store_true",
        help="Remove every subtask_* dataset/attr added by this script and exit. "
             "The script is strictly additive, so this restores the original file.",
    )
    p.add_argument(
        "--unsafe-inplace",
        action="store_true",
        help="Write directly into the live file instead of the safe "
             "copy-verify-swap pipeline. Faster for quick single-demo iteration; "
             "do NOT use for bulk runs (an interrupt can corrupt the file).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 2)),
        help="Processes for the per-demo compute (each holds its own mujoco env, "
             "~1-3GB). Default auto-scales to the box; 1 = serial. Reduce if RAM "
             "is tight. The hdf5 write stays serial via the safe pipeline.",
    )
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    assert dataset_path.exists(), dataset_path

    if args.revert:
        # Even revert goes through the atomic pipeline by default -- an
        # interrupted delete shouldn't be able to damage the live file.
        if args.unsafe_inplace:
            revert_labels(dataset_path)
        else:
            safe_mutate(dataset_path, revert_labels, _open_ok)
        return

    # env_name is in the hdf5 attrs -- read it cheaply (don't build the heavy env
    # in the main process when the work is going to the worker pool).
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        all_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    env_name = env_args["env_name"]
    print(colored(f"env_name = {env_name}", "yellow"))
    if env_name not in LABELERS:
        raise SystemExit(
            f"No subtask labeler registered for {env_name}. "
            f"Available: {sorted(LABELERS)}"
        )

    if args.demo is not None:
        demo_keys = [args.demo]
    else:
        demo_keys = all_keys
    if args.n is not None:
        demo_keys = demo_keys[: args.n]

    # Fan the compute out across processes when there is more than one demo to do
    # and more than one worker requested. Each worker computes the same labels the
    # serial path would; only the (hdf5-unsafe) write stays in the main process.
    use_pool = args.workers > 1 and len(demo_keys) > 1
    if use_pool:
        results = parallel_compute(dataset_path, env_name, demo_keys, args.workers)
        env = labeler = None  # workers held their own; main does no replay
    else:
        env, env_name = build_env_from_hdf5(dataset_path)
        labeler = get_labeler(env_name)
        results = None

    try:
        if args.dry_run:
            with h5py.File(dataset_path, "r") as f:
                _label_loop(f["data"], demo_keys, env, env_name, labeler,
                            write=False, overwrite=False, results=results)
            return

        def _mutate(work_path):
            # Always overwrite within the fresh working copy; h5repack afterwards
            # reclaims any space the overwrite frees, so no bloat accumulates.
            with h5py.File(work_path, "r+") as f:
                _label_loop(f["data"], demo_keys, env, env_name, labeler,
                            write=True, overwrite=True, results=results)

        if args.unsafe_inplace:
            _mutate(dataset_path)
        else:
            safe_mutate(dataset_path, _mutate,
                        lambda w: _verify_labeled(w, demo_keys))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
