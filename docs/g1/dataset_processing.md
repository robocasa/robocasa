# Dataset processing and release metadata

Process a new collection as two related streams: raw simulation recordings establish task-state replay evidence, while SONIC exporter data supplies the latent-action training records. The processing tools create reports and derived outputs; preserve the original recordings, file timestamps, and a backup before preparing a new release.

The published benchmark is already frozen at 40 tasks × 50 paired demonstrations. Download its fixed revision and mapping from the [dataset guide](datasets.md). The workflow below prepares **new collections**. It does not redefine the accepted demonstrations or regenerate the existing release's success annotations.

## Tools and environments

Run these commands from the RoboCasa repository root, using the Python environment indicated below.

| Tool | Purpose | Required environment |
| --- | --- | --- |
| `robocasa/scripts/evaluate_sonic_raw.py` | Recorded-state replay using each task's success function | RoboCasa, compatible robosuite, MuJoCo, assets, `h5py`, NumPy |
| `robocasa/scripts/replay_sonic_dataset.py` | Parquet/video structural checks and synchronized video preview | NumPy, pandas, PyArrow, OpenCV |
| `robocasa/scripts/clean_sonic_dataset.py` | Remove discarded episodes and stale pose frames into a new LeRobot copy | SONIC exporter environment, PyAV, pandas, PyArrow, LeRobot v2.1 interfaces, compatible SONIC `process_dataset.py` |
| `robocasa/scripts/align_sonic_success_lerobot.py` | Plan raw-to-exporter pairing; optionally build a cleaned copy | Python for planning; the cleaner environment for `--apply` |
| `robocasa/scripts/dataset_scripts/convert_sonic_hdf5_lerobot.py` | Derive a LeRobot dataset of recorded WBC commands and re-rendered images | Simulator and LeRobot conversion dependencies |
| `python -m robocasa_eval.dataset_release` | Build or verify release manifests, mappings, and SHA-256 digests | Python standard library |

The cleaner expects the LeRobot `lerobot.common.datasets` interfaces used by the compatible SONIC processor. Use that verified environment rather than assuming every LeRobot version has the same API.

The LeRobot replay tool inspects recorded videos and tabular features. It does not reconstruct a MuJoCo scene or determine task success. Use raw recorded-state replay for success checks.

Set paths for one newly collected task. `RAW_TASK` must identify the actual session or task directory printed by the collector; `VLA_TASK` identifies the corresponding unfiltered exporter dataset.

```bash
export ROBOCASA_ROOT="$(pwd)"
export SIM_PYTHON="/absolute/path/to/simulation-env/bin/python"
export SONIC_ROOT="/absolute/path/to/GR00T-WholeBodyControl"
export DATA_PYTHON="$SONIC_ROOT/.venv_data_collection/bin/python"
export RAW_TASK="/absolute/path/to/collection/raw/atomic_tasks/<session>_CloseDishwasher_sonic"
export VLA_TASK="/absolute/path/to/collection/lerobot/atomic_tasks/robocasa_CloseDishwasher_g1_3cam"
export REVIEW_ROOT="$ROBOCASA_ROOT/evaluation_runs/data_review/session_01"
mkdir -p "$REVIEW_ROOT"
```

Use a new review directory for each validation run. Replay/alignment reports are written outside the source dataset, and the commands reject an existing report path rather than overwriting earlier evidence.

## 1. Inspect the raw recording and task success

Start with a small read-only check in the simulator environment:

```bash
MUJOCO_GL=egl "$SIM_PYTHON" robocasa/scripts/evaluate_sonic_raw.py \
  --dataset "$RAW_TASK" \
  --limit 2 \
  --success-mode any \
  --isolate-episodes \
  --report "$REVIEW_ROOT/raw_replay_sample.json"
```

Then cover every episode before pairing:

```bash
MUJOCO_GL=egl "$SIM_PYTHON" robocasa/scripts/evaluate_sonic_raw.py \
  --dataset "$RAW_TASK" \
  --success-mode any \
  --isolate-episodes \
  --report "$REVIEW_ROOT/raw_replay_all.json"
```

The tool restores each recording's model, scene metadata, and states, then calls the environment's `_check_success()`. It prefers the complete raw NPZ state trajectory when present, including the true terminal state. Inspect the report's state source, errors, success mode, and terminal-success result. A missing dependency or replay exception requires investigation; it is not a demonstrated task failure.

`--success-mode any` accepts success at some checked point in the trajectory. `--success-mode final` accepts only terminal success. Select and document this policy before curating a new dataset. Saving an attempt in the collector is a separate operator action and does not establish either result. A state-replay result also does not test whether replaying commands reproduces the same dynamics.

For the frozen v1 benchmark, retain the historical `any` acceptance policy with terminal-success priority: 1,984 terminal-success episodes and 16 earlier-only episodes. The release's `DATA_NOTES.md` records those qualifications and the `CloseStandMixerHead` evidence limitation. Keep release success annotations in Markdown, and leave the recorded data and prompts unchanged.

## 2. Check the exporter data

Use an environment with the video and Parquet dependencies:

```bash
"$DATA_PYTHON" robocasa/scripts/replay_sonic_dataset.py "$VLA_TASK" \
  --all --check-only \
  --output-dir "$REVIEW_ROOT/lerobot_checks"
```

Inspect `replay_report.json` for missing files, shapes, non-finite values, frame counts, timestamps, and video reliability. `--check-only` still decodes videos for validation; it does not write preview videos. To create a synchronized preview, run the command with a new output directory and omit `--check-only`.

Confirm the dataset includes a real 64-dimensional `action.motion_token` and two 7-dimensional hand-command fields. Shape validity alone cannot prove these are meaningful SONIC targets; retain the exporter/controller provenance and inspect representative trajectories. The baseline [state and action mapping](datasets.md#lerobot-schema) differs from the HDF5 converter's WBC representation.

## 3. Plan the pairing before applying cleanup

Run alignment on the original recording files or on a copy that preserves their timestamps. This tool matches raw episode completion times from state filenames against exporter Parquet modification times. Defaults allow up to 15 seconds of lag and require a 1-second ambiguity margin. It requires replay evidence for every raw HDF5 and a unique match for every accepted raw episode.

```bash
"$DATA_PYTHON" robocasa/scripts/align_sonic_success_lerobot.py \
  --raw-root "$RAW_TASK" \
  --lerobot-dataset "$VLA_TASK" \
  --replay-report "$REVIEW_ROOT/raw_replay_all.json" \
  --report "$REVIEW_ROOT/alignment_plan.json"
```

Review the proposed pairs and exclusions before constructing a derived copy. Investigate ambiguous or missing matches using the original files and logs. Do not adjust match tolerances merely to make an uncertain join pass. Hugging Face downloads can have new modification times; for the published benchmark, use its explicit `episode_mapping` files instead of reconstructing pairing from download timestamps.

Create the derived LeRobot dataset with the same inputs and a new report path:

```bash
"$DATA_PYTHON" robocasa/scripts/align_sonic_success_lerobot.py \
  --raw-root "$RAW_TASK" \
  --lerobot-dataset "$VLA_TASK" \
  --replay-report "$REVIEW_ROOT/raw_replay_all.json" \
  --report "$REVIEW_ROOT/alignment_applied.json" \
  --apply \
  --cleaner-python "$DATA_PYTHON" \
  --processor-path "$SONIC_ROOT/gear_sonic/scripts/process_dataset.py" \
  --output-suffix _replay_cleaned
```

The output is a sibling named `<source>_replay_cleaned`; its parent must be writable. Alignment delegates copy, reindexing, video filtering, statistics rebuilding, and verification to the cleaner. Source data is preserved. Keep the alignment report and cleanup report, including their source indices, when preparing the final mapping.

The alignment report stays in the external review directory. Replay outcomes and selection reasons are not copied into the derived dataset's metadata. Describe published success qualifications in Markdown, and keep the released episode mappings limited to episode identities and paths.

### Inspect cleanup separately

For a dataset whose pairing has already been established, inspect cleanup without writing a copy:

```bash
"$DATA_PYTHON" robocasa/scripts/clean_sonic_dataset.py "$VLA_TASK" --dry-run
```

To create a sibling `_cleaned` dataset:

```bash
"$DATA_PYTHON" robocasa/scripts/clean_sonic_dataset.py "$VLA_TASK" \
  --processor-path "$SONIC_ROOT/gear_sonic/scripts/process_dataset.py" \
  --output-suffix _cleaned
```

The cleaner removes explicitly discarded episodes and stale SMPL pose frames in POSE / POSE_PAUSE modes (1 and 4). Zero SMPL values in PLANNER or OFF modes have valid meanings and are preserved. Review `meta/cleanup_report.json` and verify the output before selecting it for a release. Filtering frames can re-encode video, so record this as a new derived dataset; do not use this step to rewrite the frozen benchmark's videos.

## 4. Select and aggregate a task release

Select paired episodes only after the checks above. Preserve the association between the original raw ID, original exporter episode index, cleaned index where applicable, and released index. For a benchmark in the published format:

1. Choose 50 accepted pairs for each of the 25 atomic and 15 seen composite tasks, prioritizing terminal success and then scene, object, instruction, and trajectory diversity. Exclude demonstration data for the 10 unseen tasks.
2. Copy the selected HDF5 files byte-for-byte, retaining the original `ep_*` names.
3. Build each LeRobot task directory with contiguous released episode indices. Rebuild `episode_index`, `frame_index`, global `index`, `task_index`, and the required subset metadata consistently; keep non-index sample values and instructions unchanged by selection. Reuse encoded videos when selecting whole episodes without trimming.
4. Save the explicit pairing list for each task and compare each selected pair against its source evidence. Do not concatenate Parquet files or rename videos without updating their index and metadata references.

The release metadata command below inventories a prepared release. It does not choose the 50 pairs, copy or reindex data, run simulation, or infer a raw/exporter correspondence. A selection or aggregation step must first produce consistent data and its explicit pairing lists.

## 5. Build and verify release metadata

Copy `examples/datasets/release.json` to a working configuration and set:

- `dataset_root`: the prepared directory with `lerobot/` and `hdf5/` subdirectories in the [published layout](datasets.md#directory-layout-and-episode-identity).
- `pairings_dir`: the directory containing `<Task>.json` pairing lists with original and released episode indices and raw IDs.
- `release_id` and `source_revision`: identifiers for this release and its source revision.
- `episodes_per_task` and `fps`: normally 50 and 50. The default task manifest fixes the benchmark to 40 released tasks, 10 unseen tasks, 50 pairs per released task, and 50 Hz. An explicit custom `task_manifest` supports a separate small validation sample; it is labeled as a custom release.

```bash
python -m robocasa_eval.dataset_release build \
  --config ./evaluation_runs/release.json \
  --output ./evaluation_runs/release_metadata

python -m robocasa_eval.dataset_release verify \
  --config ./evaluation_runs/release.json \
  --output ./evaluation_runs/release_metadata
```

`build` writes a new metadata directory containing `release_manifest.json`, `episode_mapping/<Task>.json`, and `checksums/<Task>.json`. `verify` reads that directory and checks it against the configured dataset and the original pairing lists, which must remain available. Configuration paths are resolved from the current working directory. Manifest references to mapping/checksum files are relative to the metadata directory; data paths are relative to `dataset_root`.

Review the task counts, one-to-one pair mapping, declared index coverage, expected files, and SHA-256 results. This command performs structural and byte-integrity checks. `source_revision` is a provenance label supplied by the release author, not an independently verified Git revision. The command does not decode all videos, inspect every Parquet value, establish meaningful latent supervision, or assign task-success labels. Keep the replay and exporter validation reports as separate evidence, and publish the human-readable qualifications in Markdown.

## Optional: derive WBC LeRobot data from HDF5

For analysis or a model trained on low-level joint commands, the converter can merge multiple HDF5 inputs and re-render observations into a separate LeRobot directory:

```bash
MUJOCO_GL=egl "$SIM_PYTHON" \
  robocasa/scripts/dataset_scripts/convert_sonic_hdf5_lerobot.py \
  --raw-dataset-paths /absolute/path/to/session/demo.hdf5 \
  --output-path ./evaluation_runs/derived_wbc/CloseDishwasher \
  --source-fps 200 --target-fps 50 \
  --three-camera
```

At 200 Hz source and 50 Hz target, the converter selects every fourth recorded action-aligned state and re-renders images. Check the source recording rate before overriding it, especially if collection used a non-default `--record-freq`. Camera metadata records virtual wrist-camera injection when the original model lacks those cameras.

The converter writes `action.wbc` / `action` as the recorded 43-dimensional command, with its own state layout and `meta/conversion_warnings.json`. It does not recover the real 64-dimensional latent or reproduce the live exporter's complete schema. Its optional zero-filled motion-token field is a placeholder and is unsuitable as SONIC latent supervision. Use the paired live exporter records for the 78-dimensional baseline action target.

## Review before publication

A prepared release should have 40 task directories in each format, 50 unique pairs per task, zero unseen demonstration directories, verified source-to-output indices, readable camera previews, checksums, and a Markdown account of selection and replay evidence. Record the data revision separately from simulator, robosuite, SONIC, and decoder revisions so a reader can identify which evidence belongs to which artifact.
