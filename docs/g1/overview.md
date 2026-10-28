# Unitree G1

RoboCasa for Unitree G1 provides kitchen manipulation tasks, paired
demonstrations, a SONIC collection workflow, and a shared evaluation interface
for VLA models. The robot uses a 29-joint body and two seven-joint Dex3 hands.
Policies receive camera images, robot state, and the episode instruction, and
produce SONIC motion tokens with hand joint targets.

## Benchmark

| Task group | Tasks | Released demonstrations | Evaluation episodes | Episode limit |
| --- | ---: | ---: | ---: | ---: |
| Seen atomic | 25 | 1,250 | 750 | 40 s |
| Seen composite | 15 | 750 | 450 | 180 s |
| Unseen composite | 10 | 0 | 300 | 180 s |

The release contains **2,000 paired demonstrations across 40 seen tasks**.
Standard evaluation covers all **50 tasks with 30 episodes per task**. Limits
are simulation time. The [task reference](tasks.md) defines the split,
instructions, and implemented success conditions.

## Choose a workflow

- [G1 installation](installation.md): prepare a new simulation environment, assets, and SONIC decoder.
- [Datasets](datasets.md): download the released data and inspect its schema.
- [Policy evaluation](evaluation.md): connect a model server and run the benchmark.
- [Data collection](data_collection.md): collect paired HDF5 and LeRobot recordings.
- [Dataset processing](dataset_processing.md): validate, clean, pair, and describe new collections.
- [Release provenance](reproducibility.md): match simulation dependency versions and validate a deployment.

After following the installation guide, connect an existing inference service
to the simulator through
**one HTTP port, 8000 by default**. Configure its endpoint, the simulation Python
interpreter, and the SONIC decoder. The service receives images, a 46-dimensional
robot state, and the task instruction; it returns a 64-dimensional SONIC motion
token and two seven-joint hand targets. Implement the shared HTTP protocol, or
use the optional Python adapter to convert model inputs and outputs. Multiple
service replicas and concurrent simulation episodes use the same endpoint.

## Release artifacts

| Artifact | Location and contents |
| --- | --- |
| Dataset | [Dawn-star/robocasa-sonic](https://huggingface.co/datasets/Dawn-star/robocasa-sonic/tree/benchmark40-v1-hdf5-20261022), tag `benchmark40-v1-hdf5-20261022` |
| LeRobot demonstrations | Dataset `lerobot/`, grouped into `Atomic Tasks/` and `Composite Tasks/` |
| Original HDF5 recordings | Dataset `hdf5/`, with the same task groups and paired raw episode IDs |
| Pair mappings and checksums | Dataset `metadata/benchmark40_v1/`; selection qualifications in `DATA_NOTES.md` |
| Evaluation and data-tool examples | [Examples index](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/README.md) |

The [download guide](datasets.md#download) provides fixed-revision commands for
one task or the full dataset. HDF5 stores recorded simulation states and WBC
commands; the paired live LeRobot export contains the SONIC motion-token
training targets. Use the documented [state and action schema](datasets.md#lerobot-schema)
when preparing a model.

## Repository layout

| Path | Purpose |
| --- | --- |
| `docs/g1/` | User guides, task reference, and release provenance |
| `docs/developer/` | Collection transport and debugging details |
| `examples/evaluation/` | Inference-service and evaluator configurations and simulation dependency pins |
| `examples/datasets/` | Release metadata configuration |
| `examples/sonic/` | Companion collection patch for the pinned SONIC source |
| `robocasa_eval/` | Shared model service, simulator evaluation, reporting, and release metadata tools |
| `robocasa/scripts/` | Collection, replay, cleaning, alignment, and conversion entry points |

The [streaming developer notes](../developer/sonic_vla_streaming.md) cover
collection message formats and recording synchronization. General simulator
usage and asset references remain available in the RoboCasa sections of the
documentation.
