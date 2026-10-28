# Datasets

The RoboCasa-G1 benchmark publishes **40 tasks with 50 paired demonstrations per task**: 25 atomic tasks and 15 composite tasks, for 2,000 demonstrations. Each demonstration has a LeRobot export and its original HDF5 recording. The 10 unseen composite tasks have no demonstration data in either format. See the [task catalog](tasks.md) for the split.

The dataset is [Dawn-star/robocasa-sonic](https://huggingface.co/datasets/Dawn-star/robocasa-sonic). Use this fixed release for the directory layout documented here:

| Field | Value |
| --- | --- |
| Tag | `benchmark40-v1-hdf5-20261022` |
| Commit | `ee964e43b35aed036319d494219d7377e58c7669` |
| LeRobot | v2.1; 3,211,854 frames; 6,000 camera videos |
| HDF5 | 2,000 files; 285,076,349,296 bytes (285.08 GB / 265.50 GiB) |

The October 22 release adds HDF5 and moves the previously published LeRobot files under `lerobot/`. It preserves the same demonstration selection and file contents. The earlier tag `benchmark40-v1-20261013` uses top-level `Atomic Tasks/` and `Composite Tasks/` directories and contains only LeRobot data and pairing metadata.

## Download

Choose `lerobot`, `hdf5`, or both. The following example downloads LeRobot plus the release metadata; downloading HDF5 as well requires at least the additional space listed above.

```python
from huggingface_hub import snapshot_download

formats = ["lerobot"]  # Use ["hdf5"] or ["lerobot", "hdf5"] as needed.
snapshot_download(
    repo_id="Dawn-star/robocasa-sonic",
    repo_type="dataset",
    revision="ee964e43b35aed036319d494219d7377e58c7669",
    local_dir="./datasets/robocasa-sonic",
    allow_patterns=[
        *(f"{name}/**" for name in formats),
        "metadata/**",
        "README.md",
        "DATA_NOTES.md",
    ],
)
```

For an initial inspection, download one task in both formats instead:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Dawn-star/robocasa-sonic",
    repo_type="dataset",
    revision="ee964e43b35aed036319d494219d7377e58c7669",
    local_dir="./datasets/robocasa-sonic",
    allow_patterns=[
        "lerobot/Atomic Tasks/robocasa_CheesyBread_g1_3cam/**",
        "hdf5/Atomic Tasks/CheesyBread_sonic/**",
        "metadata/**",
        "README.md",
        "DATA_NOTES.md",
    ],
)
```

## Directory layout and episode identity

```text
lerobot/
  Atomic Tasks/robocasa_<Task>_g1_3cam/{data,meta,videos}/...
  Composite Tasks/robocasa_<Task>_g1_3cam/{data,meta,videos}/...
hdf5/
  Atomic Tasks/<Task>_sonic/episodes/<raw_episode>/ep_demo.hdf5
  Composite Tasks/<Task>_sonic/episodes/<raw_episode>/ep_demo.hdf5
metadata/benchmark40_v1/
  release_manifest.json
  episode_mapping/<Task>.json
  checksums/<Task>.json
DATA_NOTES.md
```

Paths containing `Atomic Tasks` or `Composite Tasks` need quotes in shell commands. A LeRobot loader should receive one task's directory, such as `lerobot/Atomic Tasks/robocasa_CheesyBread_g1_3cam`.

`episode_mapping/<Task>.json` is the pairing authority:

| Field | Meaning |
| --- | --- |
| `output_episode_index` | Released LeRobot episode index, 0–49 within each task |
| `source_episode_index` | LeRobot index before the 50-episode selection |
| `raw_episode` | Original raw episode directory ID |
| `lerobot_data_path` | Released Parquet path relative to the repository root |
| `hdf5_path` | Paired HDF5 path relative to the repository root |

Do not pair files by their position in a directory listing or assume the source and released indices match. In each task, `meta/info.json` defines the Parquet and video path templates. `meta/tasks.jsonl` maps `task_index` to the sampled instruction; a single environment class can have many instruction strings.

## LeRobot schema

The published dataset contains the state and action fields used by the shared G1 evaluation protocol, together with collection telemetry. Read each task's `meta/info.json` for stored shapes and types, and `meta/modality.json` for named features. The evaluation protocol uses **46 state values** and **78 action values**.

### State

The 43-element `observation.state` stores joint positions in radians in this order. Slice ends are exclusive.

| Model feature | Source | Slice | Dimensions |
| --- | --- | --- | ---: |
| `state.left_leg` | `observation.state` | `0:6` | 6 |
| `state.right_leg` | `observation.state` | `6:12` | 6 |
| `state.waist` | `observation.state` | `12:15` | 3 |
| `state.left_arm` | `observation.state` | `15:22` | 7 |
| `state.left_hand` | `observation.state` | `22:29` | 7 |
| `state.right_arm` | `observation.state` | `29:36` | 7 |
| `state.right_hand` | `observation.state` | `36:43` | 7 |
| `state.projected_gravity` | `observation.projected_gravity` | `0:3` | 3 |

Projected gravity is the unit downward vector expressed in the robot base frame. The 46-dimensional evaluation state combines the named joint fields with projected gravity; `observation.state` alone is 43-dimensional. Additional stored values, including wrist poses and root orientation, are available for analysis.

### Actions

| Model feature | Stored source | Dimensions | Meaning |
| --- | --- | ---: | --- |
| `action.motion_token` | `action.motion_token` | 64 | SONIC latent motion command |
| `action.left_hand_joints` | `teleop.left_hand_joints` | 7 | Left hand absolute joint targets, radians |
| `action.right_hand_joints` | `teleop.right_hand_joints` | 7 | Right hand absolute joint targets, radians |

The latent is produced by the live SONIC pipeline and decoded into body control during evaluation. `action.wbc`, also present in the export, is the 43-dimensional low-level joint command and is a different action representation.

Hand state and action ordering differ. Each hand's **state** is index finger (2 joints), middle finger (2), then thumb (3). Each hand's **action** uses the SONIC motor order: thumb (3), index finger (2), then middle finger (2). Converting a motor-order vector to the dataset's state order uses indices `[3, 4, 5, 6, 0, 1, 2]`. The evaluator implements this conversion in `robocasa_eval/sim_backend.py`.

### Images, instructions, and rates

The released videos are 640 × 480 RGB. Their camera keys correspond to these canonical evaluation names:

| Dataset camera key | Evaluation `camera_names` entry |
| --- | --- |
| `observation.images.ego_view` | `ego_view` |
| `observation.images.left_wrist` | `left_wrist` |
| `observation.images.right_wrist` | `right_wrist` |

The evaluation configuration defaults to `camera_names: ["ego_view"]`. Add `left_wrist` and `right_wrist` to receive those views, and map the resulting observations to the inference service's inputs. The task text is read through `task_index` and `meta/tasks.jsonl`. See the [observation contract](evaluation.md#observation-contract) for a three-camera configuration and the evaluation message format.

| Clock | Rate | Interpretation |
| --- | ---: | --- |
| MuJoCo / low-level collection loop | 200 Hz | Physics and WBC joint commands |
| Live camera publisher | 30 Hz by default | New rendered camera samples |
| LeRobot export and video timeline | 50 Hz | Synchronized dataset rows and encoded frames |
| Evaluation latent / hand commands | 50 Hz | Four 200 Hz simulation steps per command |

The exporter samples the latest available observations at 50 Hz. A 50 fps dataset can therefore contain repeated images from the 30 Hz publisher. Encoded video rate does not imply 50 independently rendered images per second. The HDF5-only converter can instead re-render recorded simulation states; that produces a separate derived dataset.

## HDF5 scope

The download contains selected `ep_demo.hdf5` files with recorded simulation states, actions, model and episode metadata. It preserves the original HDF5 bytes and raw episode IDs. Separate raw NPZ, XML, and JSON sidecars from the collection directory are not part of this upload; LeRobot camera videos are under `lerobot/`.

State replay uses the recorded environment model and compatible simulator assets. For terminal-state audits, preserve the full local raw episode directory when available: the raw NPZ trajectory can include the true terminal state beyond the action-aligned HDF5 states. Report the state source used by a replay rather than silently treating those two sources as equivalent.

## Acceptance and checksums

The frozen selection reuses the previously accepted recorded-state replay evidence: success at any point in the trajectory, with terminal-success episodes prioritized before scene, object, instruction, and trajectory diversity. **1,984 episodes have historical terminal-success evidence; 16 succeeded earlier without retaining success at the final state.** The affected episodes and the historical evidence limitation for `CloseStandMixerHead` are documented in the release's [DATA_NOTES.md](https://huggingface.co/datasets/Dawn-star/robocasa-sonic/blob/ee964e43b35aed036319d494219d7377e58c7669/DATA_NOTES.md).

These qualifications are recorded only in Markdown. No success labels are added to HDF5, Parquet, videos, or task instructions. The original selection rebuilt the four LeRobot index columns and required subset metadata while preserving non-index values and encoded videos. The October 22 layout update preserves those released bytes.

Each `checksums/<Task>.json` contains SHA-256 digests. Its `lerobot` paths are relative to the task's `root` in the manifest; its `hdf5` paths are relative to `hdf5_root`. After downloading one task in both formats, this example resolves the first pair and verifies its Parquet and HDF5 bytes:

```python
import hashlib
import json
from pathlib import Path

root = Path("./datasets/robocasa-sonic")
metadata = root / "metadata/benchmark40_v1"
manifest = json.loads((metadata / "release_manifest.json").read_text())
task = next(item for item in manifest["tasks"] if item["task"] == "CheesyBread")
pair = json.loads((root / task["episode_mapping"]).read_text())[0]
checksums = json.loads((root / task["checksums"]).read_text())

for format_name, root_key, path_key in [
    ("lerobot", "root", "lerobot_data_path"),
    ("hdf5", "hdf5_root", "hdf5_path"),
]:
    path = root / pair[path_key]
    relative = path.relative_to(root / task[root_key]).as_posix()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checksums[format_name][relative]:
        raise ValueError(f"Checksum mismatch: {path}")
    print(f"Verified {format_name}: {path}")
```

This checks one pair; verify every downloaded file, including videos and metadata, before a full experiment. The [processing guide](dataset_processing.md) describes validation and metadata generation for a new release; rebuilding metadata does not establish a new simulation-success result.
