"""Augment an existing LeRobot dataset in place with per-frame subtask labels.

Adds four per-frame int64 columns to every episode parquet, mirroring the four
per-frame subtask_* arrays the hdf5 stores:
  - annotation.human.subtask        -> subtask_lang  (index into meta/tasks.jsonl)
  - annotation.human.subtask_name   -> subtask_name  (index into meta/tasks.jsonl)
  - annotation.human.subtask_stage  -> subtask_stage (index into meta/tasks.jsonl)
  - subtask_idx                     -> the raw per-frame segment index
The three annotation.* columns index meta/tasks.jsonl (their strings are
registered there); subtask_idx is a raw numeric column. All are declared in
meta/info.json, and the three string annotations in meta/modality.json.

Subtask labels are computed self-contained from each episode's own
extras/episode_*/states.npz + ep_meta.json + model.xml.gz, replayed through the
registered subtask labeler. No hdf5 join and no video re-render -- the 166 MB of
video and every other column are untouched.

Reversible:
  --apply   backs up data/ + meta/ to <lerobot>/_subtask_backup/ first, then
            writes the new column + metadata.
  --revert  restores data/ + meta/ from the backup.
  --dry-run computes labels and prints what would change; writes nothing.

Example:
    python -m robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot \\
        --lerobot /mnt/.../AddIceCubes/20250829/lerobot --dry-run
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from termcolor import colored
from tqdm import tqdm

import robosuite

import robocasa  # noqa: F401  (registers envs)
from robocasa.scripts.dataset_scripts.label_subtasks import reset_to
from robocasa.utils.subtask_labeling import get_labeler, LABELERS
from robocasa.utils.subtask_labeling.manual_overrides import get_lerobot_override

ANNOT_KEY = "annotation.human.subtask"          # per-frame subtask_lang (string idx)
NAME_KEY = "annotation.human.subtask_name"      # per-frame subtask_name (string idx)
STAGE_KEY = "annotation.human.subtask_stage"    # per-frame subtask_stage (string idx)
IDX_KEY = "subtask_idx"                          # per-frame segment index (raw int64)
MODALITY_KEY = "human.subtask"
NAME_MODALITY = "human.subtask_name"
STAGE_MODALITY = "human.subtask_stage"
# annotation.* columns index meta/tasks.jsonl; subtask_idx is a raw numeric column.
STR_ANNOTS = ((ANNOT_KEY, MODALITY_KEY), (NAME_KEY, NAME_MODALITY), (STAGE_KEY, STAGE_MODALITY))
BACKUP_DIRNAME = "_subtask_backup"


def copy_tree_contents(src: Path, dst: Path):
    """Recursively copy file *contents* only (no metadata). shutil.copytree uses
    copy2/copystat, which fails with 'Operation not permitted' on some mounted
    drives (can't set ownership/utime); copying bytes only avoids that."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.rglob("*")):
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def build_env(env_args: dict):
    """Construct a robosuite env from the dataset_meta env_args. The controller
    is irrelevant (we only replay sim states), so drop controller_configs to
    avoid robocasa<->robosuite version mismatches."""
    ek = dict(env_args.get("env_kwargs", env_args))
    ek["env_name"] = env_args["env_name"]
    ek["has_renderer"] = False
    ek["has_offscreen_renderer"] = False
    ek["use_camera_obs"] = False
    ek["translucent_robot"] = False
    ek.pop("controller_configs", None)
    # Legacy split name: some early datasets recorded "train"; current code
    # expects "pretrain"/"target"/None.
    if ek.get("obj_instance_split") == "train":
        ek["obj_instance_split"] = "pretrain"
    return robosuite.make(**ek)


def load_episode(ep_dir: Path):
    states = np.load(ep_dir / "states.npz")["states"]
    ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
    with gzip.open(ep_dir / "model.xml.gz", "rt") as f:
        model_xml = f.read()
    return states, ep_meta, model_xml


def label_episode(env, labeler, states, ep_meta, model_xml):
    reset_to(
        env,
        {"model": model_xml, "states": states[0], "ep_meta": json.dumps(ep_meta)},
    )
    labeler.setup(env, ep_meta)
    history = []
    for t in range(states.shape[0]):
        reset_to(env, {"states": states[t]})
        history.append(labeler.per_step(env, t))
    return labeler.resolve(history)


def episode_parquets(lerobot: Path):
    return sorted((lerobot / "data").rglob("episode_*.parquet"))


def ep_index(parquet_path: Path) -> int:
    return int(parquet_path.stem.split("_")[1])


def load_tasks(lerobot: Path):
    """Return (list_of_task_strings_by_index, str->index map)."""
    by_index = {}
    path = lerobot / "meta" / "tasks.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        by_index[int(d["task_index"])] = d["task"]
    str_to_idx = {v: k for k, v in by_index.items()}
    return by_index, str_to_idx


# ---------------------------------------------------------------------------
# Parallel compute (pass 1): each episode is an independent replay. Fan out
# across processes (each builds its own env once, reads episodes from disk);
# only the per-frame instruction strings come back. The parquet write (pass 2)
# is fast file I/O and stays serial. Identical labels to the serial path.
# ---------------------------------------------------------------------------

_LR_WORKER: dict = {}
_LR_MAXTASKS = 25


def _lr_init_worker(lerobot_str: str, env_args: dict, env_name: str) -> None:
    global _LR_WORKER
    _LR_WORKER = {
        "lerobot": Path(lerobot_str),
        "env": build_env(env_args),
        "labeler": get_labeler(env_name),
        "env_name": env_name,
    }


def _lr_label_one(env, labeler, env_name, lerobot, ei, n_rows):
    """Label one episode (or apply its manual override). Returns subtask_lang."""
    override = get_lerobot_override(env_name, ei)
    # The override's final frame is its trajectory length; if it doesn't match
    # this episode's, the override is for a different split/episode -- ignore it.
    if override is not None and int(override[-1][0]) != int(n_rows):
        override = None
    if override is not None:
        labeler.expected_done_idx = len(override) - 1
        result = labeler.build_result(n_rows, override, debug={"manual": True})
        result.review = ["manually labeled (was flagged) -- human-reviewed"]
    else:
        states, ep_meta, model_xml = load_episode(
            lerobot / "extras" / f"episode_{ei:06d}"
        )
        result = label_episode(env, labeler, states, ep_meta, model_xml)
    issues = labeler.validate(result)
    return (
        np.asarray(result.subtask_lang), np.asarray(result.subtask_name),
        np.asarray(result.subtask_stage), np.asarray(result.subtask_idx, dtype=np.int64),
        issues, list(result.review),
    )


def _lr_worker(task):
    ei, n_rows = task
    w = _LR_WORKER
    lang, name, stage, idx, issues, review = _lr_label_one(
        w["env"], w["labeler"], w["env_name"], w["lerobot"], ei, n_rows
    )
    return ei, lang, name, stage, idx, issues, review


def _states_hash(states) -> str:
    """Stable fingerprint of a trajectory's flattened sim states. The LeRobot
    episode states are byte-identical to the matching hdf5 demo's, so this gives
    a robust 1:1 match across the two surfaces regardless of their ordering."""
    return hashlib.sha1(
        np.ascontiguousarray(states, dtype=np.float64).tobytes()
    ).hexdigest()


def labels_from_hdf5(hdf5_path: Path, lerobot: Path, tasks):
    """#2: copy per-frame labels straight from the already-labeled hdf5 instead
    of replaying every episode. Episodes are matched to demos by states hash
    (byte-identical across surfaces), so this is order-independent and carries the
    hdf5's manual overrides for free. Returns
    {ei: (subtask_lang, subtask_name, subtask_stage, subtask_idx, issues, review)}.
    Raises with guidance to fall back to replay if any episode has no match."""
    def _str_col(g, key):
        return np.array([
            x.decode() if isinstance(x, bytes) else str(x) for x in g[key][()]
        ])

    by_hash = {}
    with h5py.File(hdf5_path, "r") as f:
        for k in f["data"]:
            g = f["data"][k]
            for req in ("subtask_lang", "subtask_name", "subtask_stage", "subtask_idx"):
                if req not in g:
                    raise SystemExit(
                        f"{hdf5_path.name}:{k} has no {req} -- label the hdf5 "
                        f"first (this mode copies from it)."
                    )
            lang = _str_col(g, "subtask_lang")
            name = _str_col(g, "subtask_name")
            stage = _str_col(g, "subtask_stage")
            idx = np.asarray(g["subtask_idx"][()], dtype=np.int64)
            issues = json.loads(g.attrs.get("subtask_issues", "[]"))
            review = json.loads(g.attrs.get("subtask_review", "[]"))
            by_hash[_states_hash(g["states"][()])] = (lang, name, stage, idx, issues, review)
    out = {}
    for ei, n_rows in tqdm(tasks, desc="copying labels from hdf5"):
        states = load_episode(lerobot / "extras" / f"episode_{ei:06d}")[0]
        h = _states_hash(states)
        if h not in by_hash:
            raise SystemExit(colored(
                f"episode {ei}: no hdf5 demo with matching states (states not "
                f"byte-identical). Re-run WITHOUT --from-hdf5 to replay.", "red"))
        lang, name, stage, idx, issues, review = by_hash[h]
        if len(lang) != n_rows:
            raise SystemExit(
                f"episode {ei}: {len(lang)} hdf5 labels != {n_rows} parquet rows"
            )
        out[ei] = (lang, name, stage, idx, issues, review)
    return out


def compute_episode_labels(lerobot, env_args, env_name, tasks, workers):
    """tasks = [(ei, n_rows), ...]. Returns {ei: (subtask_lang, issues, review)}.
    Parallel across `workers` processes when >1 and more than one episode."""
    if workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("spawn")
        out = {}
        with ctx.Pool(
            workers,
            initializer=_lr_init_worker,
            initargs=(str(lerobot), env_args, env_name),
            maxtasksperchild=_LR_MAXTASKS,
        ) as pool:
            for ei, lang, name, stage, idx, issues, review in tqdm(
                pool.imap(_lr_worker, tasks, chunksize=1),
                total=len(tasks),
                desc=f"labeling ({workers} workers)",
            ):
                out[ei] = (lang, name, stage, idx, issues, review)
        return out
    # serial
    env = build_env(env_args)
    labeler = get_labeler(env_name)
    out = {}
    for ei, n_rows in tqdm(tasks, desc="labeling (serial)"):
        out[ei] = _lr_label_one(env, labeler, env_name, lerobot, ei, n_rows)
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot", required=True, help="Path to the lerobot dataset dir")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write the labels (default).")
    mode.add_argument("--revert", action="store_true", help="Restore from backup.")
    mode.add_argument("--dry-run", action="store_true", help="Compute + report only.")
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 2)),
        help="Processes for the per-episode replay (each holds its own mujoco "
             "env, ~1-3GB). Default auto-scales; 1 = serial. The parquet write "
             "stays serial.",
    )
    p.add_argument(
        "--from-hdf5",
        type=str,
        default=None,
        help="Copy per-frame labels from this already-labeled demo.hdf5 instead "
             "of replaying (episodes matched to demos by states hash). Seconds "
             "instead of minutes, and carries the hdf5's manual overrides "
             "automatically. Falls back-with-error if states aren't byte-identical.",
    )
    args = p.parse_args()

    lerobot = Path(args.lerobot)
    assert lerobot.exists(), lerobot
    backup = lerobot / BACKUP_DIRNAME

    # --- revert ---
    if args.revert:
        if not backup.exists():
            raise SystemExit(f"No backup at {backup}; nothing to revert.")
        for sub in ("data", "meta"):
            tgt = lerobot / sub
            if tgt.exists():
                shutil.rmtree(tgt)
            copy_tree_contents(backup / sub, tgt)
        print(colored(f"Reverted data/ + meta/ from {backup}", "green"))
        return

    apply = not args.dry_run  # default action is apply

    # --- identify env + labeler ---
    meta = json.loads((lerobot / "extras" / "dataset_meta.json").read_text())
    env_args = meta["env_args"]
    env_name = env_args["env_name"]
    print(colored(f"env_name = {env_name}", "yellow"))
    if env_name not in LABELERS:
        raise SystemExit(f"No labeler for {env_name}. Have: {sorted(LABELERS)}")

    parquets = episode_parquets(lerobot)
    print(colored(f"{len(parquets)} episode parquet(s)", "yellow"))

    # --- pass 1: label every episode (replay), cache per-frame instructions ---
    by_index, str_to_idx = load_tasks(lerobot)
    next_idx = (max(by_index) + 1) if by_index else 0
    tasks = [(ep_index(pqp), pq.read_metadata(pqp).num_rows) for pqp in parquets]
    if args.from_hdf5:
        # #2: copy from the labeled hdf5 (no replay, no env).
        ep_results = labels_from_hdf5(Path(args.from_hdf5), lerobot, tasks)
    else:
        ep_results = compute_episode_labels(
            lerobot, env_args, env_name, tasks, args.workers
        )

    cached = {}  # episode_index -> (lang, name, stage, idx) per-frame arrays
    passed = review = flagged = 0
    for ei, n_rows in tasks:
        lang, name, stage, idx, issues, review_notes = ep_results[ei]
        if len(lang) != n_rows:
            raise SystemExit(
                f"episode {ei}: {len(lang)} labels != {n_rows} rows"
            )
        cached[ei] = (lang, name, stage, idx)
        # Register the lang, name and stage strings in the shared task vocab.
        for arr in (lang, name, stage):
            for s in {str(v) for v in np.unique(arr)}:
                if s not in str_to_idx:
                    str_to_idx[s] = next_idx
                    by_index[next_idx] = s
                    next_idx += 1
        if issues:
            flagged += 1
        elif review_notes:
            review += 1
        else:
            passed += 1

    # New tasks.jsonl entries = those whose index is beyond the original max.
    orig_by_index, _ = load_tasks(lerobot)
    orig_max = max(orig_by_index) if orig_by_index else -1
    new_strings = [(i, by_index[i]) for i in sorted(by_index) if i > orig_max]

    print(colored(f"\nstrategy split: {passed} PASS, {review} REVIEW, "
                  f"{flagged} FLAGGED", "cyan"))
    print(colored(f"new subtask strings to register ({len(new_strings)}):", "cyan"))
    for i, s in new_strings:
        print(f"    task_index {i}: {s}")

    if not apply:
        print(colored("\n[dry-run] no files written.", "yellow"))
        return

    # --- backup before mutating (content-only; the mounted FS rejects copystat) ---
    if backup.exists():
        shutil.rmtree(backup)
    for sub in ("data", "meta"):
        copy_tree_contents(lerobot / sub, backup / sub)
    print(colored(f"backed up data/ + meta/ -> {backup}", "green"))

    # --- pass 2: add the subtask columns to each parquet ---
    # annotation.human.subtask{,_name,_stage} are int indices into tasks.jsonl;
    # subtask_idx is the raw per-frame segment index.
    def _idxcol(arr):
        return pa.array(np.array([str_to_idx[str(s)] for s in arr], dtype=np.int64), type=pa.int64())

    for pq_path in parquets:
        ei = ep_index(pq_path)
        lang, name, stage, idx = cached[ei]
        table = pq.read_table(pq_path)
        for key in (ANNOT_KEY, NAME_KEY, STAGE_KEY, IDX_KEY):
            if key in table.column_names:
                table = table.drop([key])
        table = table.append_column(ANNOT_KEY, _idxcol(lang))
        table = table.append_column(NAME_KEY, _idxcol(name))
        table = table.append_column(STAGE_KEY, _idxcol(stage))
        table = table.append_column(
            IDX_KEY, pa.array(np.asarray(idx, dtype=np.int64), type=pa.int64())
        )
        pq.write_table(table, pq_path)
    print(colored(
        f"added {ANNOT_KEY}, {NAME_KEY}, {STAGE_KEY}, {IDX_KEY} to "
        f"{len(parquets)} parquet(s)", "green"))

    # --- append new strings to tasks.jsonl ---
    tasks_path = lerobot / "meta" / "tasks.jsonl"
    with open(tasks_path, "a") as f:
        for i, s in new_strings:
            f.write(json.dumps({"task_index": i, "task": s}) + "\n")
    print(colored(f"appended {len(new_strings)} entries to tasks.jsonl", "green"))

    # --- declare features in info.json ---
    info_path = lerobot / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in (ANNOT_KEY, NAME_KEY, STAGE_KEY, IDX_KEY):
        info["features"][key] = {"dtype": "int64", "shape": [1]}
    info_path.write_text(json.dumps(info, indent=4))
    print(colored("declared subtask features in info.json", "green"))

    # --- declare the string annotations in modality.json ---
    mod_path = lerobot / "meta" / "modality.json"
    mod = json.loads(mod_path.read_text())
    mod.setdefault("annotation", {})
    for key, modk in STR_ANNOTS:
        mod["annotation"][modk] = {"original_key": key}
    mod_path.write_text(json.dumps(mod, indent=4))
    print(colored("declared subtask annotations in modality.json", "green"))

    print(colored("\nDONE. Revert with --revert.", "green"))


if __name__ == "__main__":
    main()
