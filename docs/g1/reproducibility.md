# Release provenance and validation

The dataset selection, simulator, inference adapter, and task scoring form one
reproducible experiment. Record their revisions along with the checkpoint and
the generated run manifest. This page distinguishes the completed historical
baseline from validation of the public evaluation entry point.

## Dataset versions

The public paired release is `Dawn-star/robocasa-sonic` at tag
`benchmark40-v1-hdf5-20261022`, commit
`ee964e43b35aed036319d494219d7377e58c7669`. It contains `lerobot/`, `hdf5/`, and
`metadata/benchmark40_v1/`. The selection is 40 seen tasks with 50 paired
demonstrations each; the 10 unseen tasks have no released demonstrations.

The earlier `benchmark40-v1-20261013` tag points to
`0e216ce0e4555dc3233d08900503b9be59fd1b7b`. It uses the original top-level
LeRobot directory layout and contains no HDF5 upload. The later release moves
the same LeRobot files and adds their paired HDF5 recordings. See
[Datasets](datasets.md) for fixed-revision download commands and file scopes.

## Historical runtime revisions

These revisions were recorded during the 2026-10-20 real-model validation and
the completed baseline. They are also available as the machine-readable
[runtime version record](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/runtime_versions.json).

| Component | Repository | Commit or revision |
| --- | --- | --- |
| RoboCasa task/runtime source | `robocasa/robocasa` | `de7fff3e0a5a1a8fd76f861f1a93c6dc2b0f3b54` |
| SonicG1 robosuite | `Xucxname/robosuite` | `67c18a4c66fa7bca02258a8a6aee2abf29e5427f` |
| Robot assets | `ARISE-Initiative/robosuite_models` | `2d63a6b4319bda772420daf10594ce0de5bd793d` |
| SONIC simulator/decoder dependencies | `NVlabs/GR00T-WholeBodyControl` | `a9d20b2ac0949244d94461a1a3263f38c5027c4a` |
| SONIC ONNX model | HF `nvidia/GEAR-SONIC` | `6733128a3d8a523b1418b06bca3cdf61c8b0987f` |
| CycloneDDS | `eclipse-cyclonedds/cyclonedds` | `9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd` (`0.10.2`) |

The decoder filename is `model_decoder.onnx`, size **40,900,688 bytes**, with
SHA-256 `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`.
The evaluation decoder runs through CPU ONNX Runtime. TensorRT belongs to a
separately prepared deployment environment when that deployment requires it;
it is not required by the shared evaluation decoder.

The SONIC revision above identifies the historical evaluation dependencies.
Live [data collection](data_collection.md) additionally requires the exporter
camera, recording-event, and instruction-handshake integration described in
that guide. Verify those capabilities in the collection checkout before
recording; an evaluation dependency pin alone does not establish them.

To prepare robosuite at the recorded revision in a new checkout:

```bash
git clone https://github.com/Xucxname/robosuite.git
cd robosuite
git checkout 67c18a4c66fa7bca02258a8a6aee2abf29e5427f
python -m pip install -e .
```

Use the current RoboCasa-G1 checkout for the public `robocasa_eval` package.
The historical RoboCasa commit in the table identifies the earlier task and
simulator source; it predates the public model-serving CLI. The evaluator
fingerprints the source actually imported by `sim_python`, the decoder, and
the serving model instead of assuming they match this table.

## GROOT model environment

The SDK checkout inspected for the public entry point is
`NVIDIA/Isaac-GR00T` at
`ab88b50c718f6528e1df9dcbaf75865d1b604760`. Its working tree was clean and the
commit includes `UNITREE_G1_SONIC`. Training implementation and model-specific
configuration belong in that SDK; `robocasa_eval/adapters/groot.py` supplies
the interface to this repository's shared evaluation protocol.

The checkpoint uses `nvidia/Cosmos-Reason2-2B` revision
`9ce19a195e423419c349abfc86fd07178b230561`. Its full offline snapshot includes
the 4,877,470,304-byte `model.safetensors` file as well as processor/tokenizer
files. These are required during SDK initialization even when the fine-tuned
checkpoint is available. See [model environment setup](evaluation.md) for the
workspace-local loading view and cache requirements. The fine-tuned checkpoint
itself is a separate model artifact, not part of the dataset release.

## Completed GROOT baseline

The final report was updated on **2026-10-20 14:22:32 Asia/Shanghai**. It used
the trained GROOT N1.7 Flow checkpoint at step 100,000 and the earlier direct
adapter. The run completed all 1,500 episodes with no runtime errors.

| Group | Success / planned | Success rate |
| --- | ---: | ---: |
| All 50 tasks | 485 / 1,500 | 32.33% |
| Seen 40 tasks | 445 / 1,200 | 37.08% |
| Unseen 10 composite tasks | 40 / 300 | 13.33% |
| Seen 25 atomic tasks | 312 / 750 | 41.60% |
| Seen 15 composite tasks | 133 / 450 | 29.56% |

The [baseline result record](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/baseline_results.json)
contains the run identity, original timestamp, all 50 per-task results, and
the execution protocol. The model used the current ego image, current 46-D
state, and sampled task instruction; it predicted 40 action steps and executed
up to 20 before replanning. Its 50 Hz action clock and 200 Hz physics clock
are separate from inference wall time.

Seven successes occurred at the first physics step (0.005 simulation seconds):
six `PrepareBroilingStation` episodes (1, 11, 18, 21, 24, 27) and
`DumpLeftovers` episode 21. These remain included in the frozen native score.
See [task predicate notes](tasks.md) for the state conditions; these observations
do not establish that the model performed the requested manipulation in 5 ms.
Interpretation notes stay in Markdown, while the original result counts and
data remain unchanged.

## Public data-tool validation

On 2026-10-28, the public tools were checked against the released `CheesyBread`
episode 0 and its paired raw recording `ep_1785323382_3171391` at the October 22
dataset revision. The 155,527,560-byte HDF5 matched its published SHA-256:
`3c7ba2aec2054f79298d9e589a31bed7b3fd556b95fc90bd34081fd93c7fbb02`.

- Recorded-state replay inspected 6,506 HDF5 states at 652 checkpoints and found
  both an earlier success and success at the last available state. Its source
  was `ep_demo.hdf5/aligned_states`; the original NPZ terminal-state sidecar was
  unavailable, so this is not a check of the collection's complete terminal state.
- LeRobot validation checked 1,573 rows, 31 fields, and decoded all three
  640 × 480, 50 fps videos.
- A separate, explicitly custom one-task / one-pair sample passed metadata
  `build` and `verify`, covering 10 files and 164,479,616 bytes.
- The 10 input files retained their SHA-256 values and modification times.

This validates one real pair and the public metadata workflow. It does not
replace validation of all 2,000 demonstrations or test action-dynamics replay.
The companion [SONIC collection patch](https://github.com/Xucxname/robocasa/tree/agent/sonic-g1-followups/examples/sonic)
also applied successfully to a full clean snapshot of its pinned source and
passed 16 focused tests. A live PICO collection session remains a separate
station check.

## Validate a new deployment

Follow [Evaluation](evaluation.md) from a new checkout and a prepared local GPU
environment. Record the checkout commit, model ID and service fingerprint,
decoder hash, simulator identity, device type, and output directory.

1. Run `plan` and `check` with your configured paths and a running model service.
2. Run a short single-GPU smoke with multiple simulator environments.
3. Run the full-horizon single-GPU smoke covering atomic and composite tasks.
4. Inspect episode errors, service logs, simulator logs, and the smoke receipt.
5. Expand model replicas and client workers, then run the desired benchmark.

Automated CPU transport tests and the historical baseline serve different
purposes. Report a new HTTP-service or hardware validation only with its own
completed run artifacts. A missing checkpoint or unavailable runtime is a
deployment prerequisite, not a measured policy failure. Publishing the GROOT
checkpoint location and compatible training SDK/configuration remains part of
the model release handoff.
