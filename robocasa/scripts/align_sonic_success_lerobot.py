#!/usr/bin/env python3
"""Align replay-validated raw SONIC episodes with a LeRobot dataset.

The raw collector and VLA exporter save the same attempt independently.  Raw
episodes carry the authoritative RoboCasa task-success result, while LeRobot
episodes carry the policy latent (``action.motion_token``) used for training.
This tool joins those two records by their save-completion timestamps and only
selects LeRobot episodes whose raw HDF5 replay passed.

The command is fail-closed: every raw HDF5 must have a replay result, every
passed raw episode must map uniquely, and ambiguous timestamp matches abort.
With ``--apply`` it delegates the actual non-destructive copy, reindexing,
video filtering, statistics rebuild, and verification to
``clean_sonic_dataset.py``.  The source dataset is never modified.
Replay decisions remain in the external operational report; no success labels
or selection report are added to the derived dataset's metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


EPISODE_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
STATE_PATTERN = re.compile(r"^state_(?P<seconds>\d+)_(?P<fraction>\d+)\.npz$")
TIMESTAMPED_TASK_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_sonic$"
)
DATASET_TASK_PATTERN = re.compile(
    r"^robocasa_(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_g1_3cam(?:_[A-Za-z0-9_-]+)?$"
)
REPORT_SCHEMA_VERSION = 1
DEFAULT_MAX_LAG_SECONDS = 15.0
DEFAULT_AMBIGUITY_MARGIN_SECONDS = 1.0


class AlignmentError(RuntimeError):
    """Raised when inputs cannot produce a safe, unique alignment."""


@dataclass(frozen=True)
class RawEpisode:
    """One raw RoboCasa recording attempt."""

    task: str
    episode: str
    episode_dir: Path
    hdf5_path: Path | None
    end_timestamp: float | None
    end_timestamp_source: str | None


@dataclass(frozen=True)
class LeRobotEpisode:
    """One source LeRobot episode and its required files."""

    episode_index: int
    parquet_path: Path
    parquet_mtime: float
    video_paths: tuple[Path, ...]
    existing_discarded: bool

    @property
    def files_complete(self) -> bool:
        return self.parquet_path.is_file() and all(
            path.is_file() for path in self.video_paths
        )


@dataclass(frozen=True)
class Match:
    """A unique raw-to-LeRobot timestamp match."""

    raw: RawEpisode
    lerobot: LeRobotEpisode
    lag_seconds: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AlignmentError(f"JSON root must be an object: {path}")
    return value


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AlignmentError(
                    f"{path}:{line_number}: JSONL record must be an object"
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_from_parent(episode_dir: Path) -> str | None:
    for parent in episode_dir.parents:
        match = TIMESTAMPED_TASK_PATTERN.fullmatch(parent.name)
        if match is not None:
            return match.group("task")
        if parent.name.endswith("_sonic"):
            task = parent.name[: -len("_sonic")]
            if task:
                return task
    return None


def _timestamp_from_state_file(path: Path) -> float | None:
    match = STATE_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return float(f"{match.group('seconds')}.{match.group('fraction')}")


def _raw_end_timestamp(
    episode_dir: Path,
    hdf5_path: Path | None,
) -> tuple[float | None, str | None]:
    state_timestamps = [
        value
        for path in episode_dir.glob("state_*.npz")
        if (value := _timestamp_from_state_file(path)) is not None
    ]
    if state_timestamps:
        return max(state_timestamps), "state_filename"
    if hdf5_path is not None:
        return hdf5_path.stat().st_mtime, "hdf5_mtime_fallback"
    return None, None


def discover_raw_episodes(raw_root: Path, task: str) -> list[RawEpisode]:
    """Discover all raw attempts for *task* without following symlinks."""

    if not raw_root.is_dir():
        raise AlignmentError(f"raw root does not exist: {raw_root}")
    episodes: list[RawEpisode] = []
    for current_root, directory_names, _file_names in os.walk(
        raw_root, followlinks=False
    ):
        current = Path(current_root)
        kept: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            if EPISODE_PATTERN.fullmatch(name):
                if path.is_symlink() or not path.is_dir():
                    continue
                task_hint = _task_from_parent(path)
                if task_hint != task:
                    continue
                candidate_hdf5 = path / "ep_demo.hdf5"
                hdf5_path = candidate_hdf5 if candidate_hdf5.is_file() else None
                end_timestamp, source = _raw_end_timestamp(path, hdf5_path)
                episodes.append(
                    RawEpisode(
                        task=task,
                        episode=name,
                        episode_dir=path.resolve(),
                        hdf5_path=(hdf5_path.resolve() if hdf5_path else None),
                        end_timestamp=end_timestamp,
                        end_timestamp_source=source,
                    )
                )
                continue
            if not path.is_symlink():
                kept.append(name)
        directory_names[:] = kept
    episodes.sort(
        key=lambda episode: (
            episode.end_timestamp if episode.end_timestamp is not None else float("inf"),
            str(episode.episode_dir),
        )
    )
    if not episodes:
        raise AlignmentError(f"no raw episodes found for task {task!r} in {raw_root}")
    names = [episode.episode for episode in episodes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AlignmentError(
            f"raw episode names are not unique for task {task}: {duplicates}"
        )
    return episodes


def task_from_dataset_name(dataset: Path) -> str:
    match = DATASET_TASK_PATTERN.fullmatch(dataset.name)
    if match is None:
        raise AlignmentError(
            "cannot infer task from LeRobot dataset name; pass --task explicitly: "
            f"{dataset.name}"
        )
    return match.group("task")


def _format_dataset_path(
    dataset: Path,
    pattern: str,
    episode_index: int,
    chunks_size: int,
    *,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    try:
        return dataset / pattern.format(**values)
    except (KeyError, ValueError) as exc:
        raise AlignmentError(f"invalid dataset path template {pattern!r}: {exc}") from exc


def discover_lerobot_episodes(
    dataset: Path,
) -> tuple[dict[str, Any], list[LeRobotEpisode]]:
    """Load source episode paths and timestamps from a LeRobot v2.1 dataset."""

    dataset = dataset.resolve()
    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise AlignmentError(f"not a LeRobot dataset: {dataset}")
    info = _read_json(info_path)
    latent = info.get("features", {}).get("action.motion_token")
    if not isinstance(latent, dict) or latent.get("shape") != [64]:
        raise AlignmentError(
            f"{dataset}: action.motion_token must exist with shape [64]"
        )
    chunks_size = int(info.get("chunks_size", 1000))
    data_pattern = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    video_pattern = str(
        info.get(
            "video_path",
            "videos/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    video_keys = [
        str(key)
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    discarded = {int(index) for index in info.get("discarded_episode_indices", [])}
    records = _read_jsonlines(episodes_path)
    episodes: list[LeRobotEpisode] = []
    seen: set[int] = set()
    for record in records:
        index = int(record["episode_index"])
        if index in seen:
            raise AlignmentError(f"duplicate LeRobot episode index: {index}")
        seen.add(index)
        parquet = _format_dataset_path(
            dataset, data_pattern, index, chunks_size
        ).resolve()
        if not parquet.is_file():
            raise AlignmentError(f"missing LeRobot parquet: {parquet}")
        videos = tuple(
            _format_dataset_path(
                dataset,
                video_pattern,
                index,
                chunks_size,
                video_key=video_key,
            ).resolve()
            for video_key in video_keys
        )
        episodes.append(
            LeRobotEpisode(
                episode_index=index,
                parquet_path=parquet,
                parquet_mtime=parquet.stat().st_mtime,
                video_paths=videos,
                existing_discarded=index in discarded,
            )
        )
    expected = set(range(len(records)))
    if seen != expected:
        raise AlignmentError(
            "LeRobot episode indices must be contiguous from zero: "
            f"missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}"
        )
    incomplete = [episode.episode_index for episode in episodes if not episode.files_complete]
    if incomplete:
        raise AlignmentError(
            f"LeRobot episodes have missing parquet/video files: {incomplete}"
        )
    return info, sorted(episodes, key=lambda episode: episode.parquet_mtime)


def load_replay_statuses(
    report_paths: Sequence[Path],
    task: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Load the latest replay status for one task and report provenance.

    Incremental replay can legitimately evaluate an episode more than once.
    Resolve those results by their evaluation completion time rather than by
    report argument order.  This mirrors the raw organizer's precedence:
    ``result.isolation.completed_at``, then ``result.completed_at``, then the
    report-level ``completed_at``.  Conflicting historical results are allowed;
    only conflicting statuses at the latest completion time are ambiguous.
    """

    candidates: dict[str, list[tuple[dt.datetime, str, Path, int]]] = {}
    provenance: list[dict[str, Any]] = []
    for raw_path in report_paths:
        path = raw_path.expanduser().resolve()
        report = _read_json(path)
        provenance.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "success_mode": report.get("success_mode"),
                "status": report.get("status"),
                "completed_at": report.get("completed_at"),
            }
        )
        for result_index, result in enumerate(report.get("results", [])):
            if not isinstance(result, dict) or result.get("task") != task:
                continue
            episode = str(result.get("episode", ""))
            status = str(result.get("status", ""))
            if not episode or not status:
                continue
            isolation = result.get("isolation")
            if isinstance(isolation, dict) and isolation.get("completed_at"):
                completed_at = isolation["completed_at"]
            else:
                completed_at = result.get("completed_at") or report.get(
                    "completed_at"
                )
            label = f"{path}: result[{result_index}] {task}/{episode}"
            if not isinstance(completed_at, str) or not completed_at:
                raise AlignmentError(f"{label}: missing completion timestamp")
            try:
                parsed_completed_at = dt.datetime.fromisoformat(completed_at)
            except ValueError as exc:
                raise AlignmentError(
                    f"{label}: invalid completion timestamp {completed_at!r}"
                ) from exc
            if parsed_completed_at.tzinfo is None:
                raise AlignmentError(
                    f"{label}: completion timestamp has no timezone"
                )
            candidates.setdefault(episode, []).append(
                (parsed_completed_at, status, path, result_index)
            )
    if not provenance:
        raise AlignmentError("at least one --replay-report is required")

    statuses: dict[str, str] = {}
    for episode, evidence in candidates.items():
        latest_time = max(item[0] for item in evidence)
        latest = [item for item in evidence if item[0] == latest_time]
        latest_statuses = {item[1] for item in latest}
        if len(latest_statuses) != 1:
            details = [
                {
                    "report": str(item[2]),
                    "result_index": item[3],
                    "status": item[1],
                }
                for item in latest
            ]
            raise AlignmentError(
                f"conflicting latest replay statuses for {task}/{episode} "
                f"at {latest_time.isoformat()}: {details}"
            )
        statuses[episode] = latest[0][1]
    return statuses, provenance


def align_by_completion_time(
    raw_episodes: Sequence[RawEpisode],
    lerobot_episodes: Sequence[LeRobotEpisode],
    *,
    max_lag_seconds: float,
    ambiguity_margin_seconds: float,
) -> tuple[list[Match], list[int]]:
    """Match each LeRobot save to the nearest preceding raw completion."""

    unused = {
        episode.episode: episode
        for episode in raw_episodes
        if episode.end_timestamp is not None
    }
    matches: list[Match] = []
    unmatched_lerobot: list[int] = []
    for lerobot in sorted(lerobot_episodes, key=lambda episode: episode.parquet_mtime):
        candidates = sorted(
            (
                (lerobot.parquet_mtime - raw.end_timestamp, raw)
                for raw in unused.values()
                if raw.end_timestamp is not None
                and 0 <= lerobot.parquet_mtime - raw.end_timestamp <= max_lag_seconds
            ),
            key=lambda item: (item[0], item[1].episode),
        )
        if not candidates:
            unmatched_lerobot.append(lerobot.episode_index)
            continue
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= ambiguity_margin_seconds
        ):
            raise AlignmentError(
                f"ambiguous timestamp match for LeRobot episode "
                f"{lerobot.episode_index}: "
                f"{candidates[0][1].episode} lag={candidates[0][0]:.3f}s, "
                f"{candidates[1][1].episode} lag={candidates[1][0]:.3f}s"
            )
        lag, raw = candidates[0]
        matches.append(Match(raw=raw, lerobot=lerobot, lag_seconds=lag))
        del unused[raw.episode]
    return matches, sorted(unmatched_lerobot)


def build_selection_plan(
    *,
    task: str,
    dataset: Path,
    raw_episodes: Sequence[RawEpisode],
    lerobot_episodes: Sequence[LeRobotEpisode],
    matches: Sequence[Match],
    unmatched_lerobot: Sequence[int],
    replay_statuses: dict[str, str],
    replay_provenance: list[dict[str, Any]],
    max_lag_seconds: float,
    ambiguity_margin_seconds: float,
) -> dict[str, Any]:
    """Build a fail-closed source-episode selection plan."""

    hdf5_episodes = [episode for episode in raw_episodes if episode.hdf5_path]
    missing_replay = sorted(
        episode.episode
        for episode in hdf5_episodes
        if episode.episode not in replay_statuses
    )
    if missing_replay:
        raise AlignmentError(
            "raw HDF5 episodes are missing replay results: "
            f"{missing_replay}"
        )
    match_by_raw = {match.raw.episode: match for match in matches}
    passed_raw = [
        episode
        for episode in hdf5_episodes
        if replay_statuses[episode.episode] == "passed"
    ]
    unmatched_passed = sorted(
        episode.episode
        for episode in passed_raw
        if episode.episode not in match_by_raw
    )
    if unmatched_passed:
        raise AlignmentError(
            "replay-passed raw episodes have no unique LeRobot match: "
            f"{unmatched_passed}"
        )

    successful_indices = sorted(
        match_by_raw[episode.episode].lerobot.episode_index
        for episode in passed_raw
    )
    existing_discarded = sorted(
        episode.episode_index
        for episode in lerobot_episodes
        if episode.existing_discarded
    )
    successful_but_discarded = sorted(
        set(successful_indices) & set(existing_discarded)
    )
    kept_indices = sorted(set(successful_indices) - set(existing_discarded))
    if not kept_indices:
        raise AlignmentError("no replay-passed LeRobot episodes remain to clean")
    all_indices = {episode.episode_index for episode in lerobot_episodes}
    excluded_indices = sorted(all_indices - set(kept_indices))
    extra_discard_indices = sorted(set(excluded_indices) - set(existing_discarded))

    match_records = []
    for match in sorted(matches, key=lambda value: value.lerobot.episode_index):
        status = replay_statuses.get(match.raw.episode)
        if match.raw.hdf5_path is None:
            reason = "raw_missing_hdf5"
        elif status != "passed":
            reason = f"raw_replay_{status or 'missing'}"
        elif match.lerobot.existing_discarded:
            reason = "source_already_discarded"
        else:
            reason = "keep_replay_passed"
        match_records.append(
            {
                "raw_episode": match.raw.episode,
                "raw_episode_dir": str(match.raw.episode_dir),
                "raw_hdf5": (
                    str(match.raw.hdf5_path) if match.raw.hdf5_path else None
                ),
                "raw_end_timestamp": match.raw.end_timestamp,
                "raw_end_timestamp_source": match.raw.end_timestamp_source,
                "replay_status": status,
                "lerobot_episode_index": match.lerobot.episode_index,
                "lerobot_parquet": str(match.lerobot.parquet_path),
                "completion_lag_seconds": match.lag_seconds,
                "source_existing_discarded": match.lerobot.existing_discarded,
                "selection": reason,
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "task": task,
        "source_dataset": str(dataset.resolve()),
        "policy": {
            "success_authority": "raw_hdf5_robocasa_state_replay",
            "keep_rule": (
                "unique timestamp match AND raw ep_demo.hdf5 exists AND "
                "replay status is passed AND source is not already discarded"
            ),
            "latent_feature": "action.motion_token",
            "max_completion_lag_seconds": max_lag_seconds,
            "ambiguity_margin_seconds": ambiguity_margin_seconds,
            "fail_closed": True,
        },
        "replay_reports": replay_provenance,
        "summary": {
            "raw_attempts": len(raw_episodes),
            "raw_hdf5_episodes": len(hdf5_episodes),
            "raw_replay_passed": len(passed_raw),
            "source_lerobot_episodes": len(lerobot_episodes),
            "timestamp_matches": len(matches),
            "kept_source_episode_count": len(kept_indices),
            "excluded_source_episode_count": len(excluded_indices),
        },
        "kept_source_episode_indices": kept_indices,
        "excluded_source_episode_indices": excluded_indices,
        "extra_discard_episode_indices": extra_discard_indices,
        "successful_but_source_discarded": successful_but_discarded,
        "unmatched_lerobot_episode_indices": list(unmatched_lerobot),
        "matches": match_records,
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cleaner_python_path(path: Path) -> Path:
    """Return an absolute venv entry path without dereferencing its symlink."""

    return path.expanduser().absolute()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--lerobot-dataset", required=True, type=Path)
    parser.add_argument(
        "--replay-report",
        required=True,
        action="append",
        type=Path,
        help="Raw replay JSON report; repeat to merge reports.",
    )
    parser.add_argument("--task", help="Task name; inferred from dataset by default.")
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=DEFAULT_MAX_LAG_SECONDS,
    )
    parser.add_argument(
        "--ambiguity-margin-seconds",
        type=float,
        default=DEFAULT_AMBIGUITY_MARGIN_SECONDS,
    )
    parser.add_argument(
        "--output-suffix",
        default="_replay_cleaned",
        help="Suffix passed to clean_sonic_dataset.py.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="External alignment report path (default: sibling of source dataset).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Build and verify the non-destructive cleaned dataset.",
    )
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--cleaner-python",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Python interpreter containing LeRobot, pandas, AV, and tyro "
            "(default: the interpreter running this script)."
        ),
    )
    parser.add_argument(
        "--processor-path",
        type=Path,
        default=os.environ.get("SONIC_DATASET_PROCESSOR"),
        help=(
            "GEAR-SONIC gear_sonic/scripts/process_dataset.py passed to the "
            "cleaner with --apply (or set SONIC_DATASET_PROCESSOR)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_lag_seconds <= 0:
        raise AlignmentError("--max-lag-seconds must be positive")
    if args.ambiguity_margin_seconds < 0:
        raise AlignmentError("--ambiguity-margin-seconds cannot be negative")
    if not args.output_suffix:
        raise AlignmentError("--output-suffix cannot be empty")
    if args.apply and args.processor_path is None:
        raise AlignmentError("--apply requires --processor-path or SONIC_DATASET_PROCESSOR")

    dataset = args.lerobot_dataset.expanduser().resolve()
    task = args.task or task_from_dataset_name(dataset)
    raw_episodes = discover_raw_episodes(args.raw_root.expanduser().resolve(), task)
    _info, lerobot_episodes = discover_lerobot_episodes(dataset)
    replay_statuses, replay_provenance = load_replay_statuses(
        args.replay_report, task
    )
    matches, unmatched_lerobot = align_by_completion_time(
        raw_episodes,
        lerobot_episodes,
        max_lag_seconds=args.max_lag_seconds,
        ambiguity_margin_seconds=args.ambiguity_margin_seconds,
    )
    plan = build_selection_plan(
        task=task,
        dataset=dataset,
        raw_episodes=raw_episodes,
        lerobot_episodes=lerobot_episodes,
        matches=matches,
        unmatched_lerobot=unmatched_lerobot,
        replay_statuses=replay_statuses,
        replay_provenance=replay_provenance,
        max_lag_seconds=args.max_lag_seconds,
        ambiguity_margin_seconds=args.ambiguity_margin_seconds,
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else dataset.with_name(f"{dataset.name}_raw_replay_alignment.json")
    )
    for source in (dataset, args.raw_root.expanduser().resolve()):
        if report_path == source or source in report_path.parents:
            raise AlignmentError(f"report must be outside the source dataset: {report_path}")
    if report_path.exists() or report_path.is_symlink():
        raise AlignmentError(f"refusing to overwrite report: {report_path}")
    _atomic_write_json(report_path, plan)

    summary = plan["summary"]
    print(f"Task: {task}")
    print(f"Raw HDF5 replay passed: {summary['raw_replay_passed']}")
    print(f"LeRobot source episodes: {summary['source_lerobot_episodes']}")
    print(f"Selected source episodes: {plan['kept_source_episode_indices']}")
    print(f"Excluded source episodes: {plan['excluded_source_episode_indices']}")
    print(f"Alignment report: {report_path}")

    if not args.apply:
        print("Plan only; pass --apply to build the cleaned dataset.")
        return 0

    # Do not resolve this path: a venv's ``bin/python`` is commonly a symlink,
    # and executing the resolved base interpreter loses the venv site-packages.
    cleaner_python = cleaner_python_path(args.cleaner_python)
    if not cleaner_python.is_file():
        raise AlignmentError(f"cleaner Python does not exist: {cleaner_python}")
    cleaner_script = Path(__file__).resolve().with_name("clean_sonic_dataset.py")
    cleaner_args = [
        str(cleaner_python),
        str(cleaner_script),
        str(dataset),
        "--output-suffix",
        args.output_suffix,
        "--processor-path",
        str(args.processor_path.expanduser().resolve()),
    ]
    for index in plan["extra_discard_episode_indices"]:
        cleaner_args.extend(["--extra-discard-episode", str(index)])
    if args.replace_existing:
        cleaner_args.append("--replace-existing")
    plan["cleaner_command"] = cleaner_args
    _atomic_write_json(report_path, plan)
    result = subprocess.run(cleaner_args, check=False)
    if result.returncode != 0:
        return result.returncode

    output = dataset.with_name(f"{dataset.name}{args.output_suffix}")
    cleanup_report = output / "meta" / "cleanup_report.json"
    if not cleanup_report.is_file():
        raise AlignmentError(f"cleaner did not produce report: {cleanup_report}")
    plan["cleaned_dataset"] = str(output)
    plan["cleaner_report"] = str(cleanup_report)
    plan["cleaner_report_sha256"] = _sha256(cleanup_report)
    _atomic_write_json(report_path, plan)
    print(f"Verified cleaned dataset: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlignmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
