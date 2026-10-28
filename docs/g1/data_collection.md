# Data collection

RoboCasa-G1 collection runs four cooperating processes: the RoboCasa simulator, SONIC controller, PICO manager, and SONIC LeRobot exporter. One operator attempt produces two records:

- The RoboCasa collector saves raw simulation states and 43-dimensional WBC joint commands in HDF5, with local episode sidecars.
- The SONIC exporter saves synchronized images, robot state, the real 64-dimensional motion token, hand commands, and the sampled task instruction in LeRobot format.

Keep both records and their provenance. The HDF5 converter alone cannot reconstruct the motion-token training target. See the [dataset schema](datasets.md#lerobot-schema) and [processing guide](dataset_processing.md).

## Prepare the station

Use a PICO headset and controllers with the ankle trackers configured for the SONIC teleoperation setup, an interactive display for MuJoCo, and the simulation and SONIC environments installed for this release. Confirm tracker connectivity and complete the PICO manager's calibration before recording. The controller's startup elastic band is a simulation stabilizer, controlled with `b` in the collector.

The public simulator and raw-recorder entry point is
`robocasa/scripts/collect_sonic_demos.py`.

The other three entry points belong to the SONIC checkout: `gear_sonic_deploy/deploy.sh`, `gear_sonic/scripts/pico_manager_thread_server.py`, and `gear_sonic/scripts/run_data_exporter.py`. Use SONIC commit `a9d20b2ac0949244d94461a1a3263f38c5027c4a` with the [RoboCasa collection patch](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/sonic/robocasa_collection.patch). That base commit already supports 50 Hz export and wrist cameras; the patch supplies the sampled-instruction and recording-event handshake. The [patch notes](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/sonic/README.md) describe its scope and validation.

Set the following variables in **each terminal**, replacing the two installation paths. Start from the RoboCasa repository root. All collection outputs in this example live under a single configurable directory.

```bash
export ROBOCASA_ROOT="$(pwd)"
export SONIC_ROOT="/absolute/path/to/GR00T-WholeBodyControl"
export ROBOCASA_PYTHON="/absolute/path/to/simulation-env/bin/python"
export COLLECTION_ROOT="$ROBOCASA_ROOT/evaluation_runs/collection"
export TASK="CloseDishwasher"
export CATEGORY="atomic_tasks"
mkdir -p "$COLLECTION_ROOT/raw/$CATEGORY" "$COLLECTION_ROOT/lerobot/$CATEGORY"
```

Use `composite_tasks` for a composite task. The task class names and categories are listed in the [task catalog](tasks.md).

Prepare a dedicated SONIC checkout at the fixed revision and apply the companion patch:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$SONIC_ROOT"
git -C "$SONIC_ROOT" checkout --detach a9d20b2ac0949244d94461a1a3263f38c5027c4a
cd "$SONIC_ROOT"
test "$(git rev-parse HEAD)" = a9d20b2ac0949244d94461a1a3263f38c5027c4a &&
git apply --check "$ROBOCASA_ROOT/examples/sonic/robocasa_collection.patch" &&
git apply "$ROBOCASA_ROOT/examples/sonic/robocasa_collection.patch"
```

For an existing dedicated checkout, start with the revision check after placing it at the pinned commit. Install the controller and PICO prerequisites from that SONIC revision and prepare `.venv_teleop`. The exporter requires `gear_sonic[data_collection]`, including PyZMQ, Tyro, LeRobot, PyAV, OpenCV, and Pinocchio; the upstream `install_scripts/install_data_collection.sh` prepares `.venv_data_collection` and replaces an existing environment of that name. Check the exporter's `--help` output for `--data-collection-frequency`, `--record-wrist-cameras`, and `--sync-robocasa-instruction`. Record both the SONIC base commit and patch SHA-256 with the session. The patch passed isolated application, syntax checks, and 16 instruction-protocol tests; a live four-process collection session still requires validation on the prepared station.

## Start the four processes

### 1. RoboCasa simulator and raw collector

The interactive viewer needs a display. Unset a headless EGL override for this terminal.

```bash
cd "$ROBOCASA_ROOT"
unset MUJOCO_GL
"$ROBOCASA_PYTHON" robocasa/scripts/collect_sonic_demos.py \
  --environment "$TASK" \
  --robot SonicG1 \
  --layout -1 \
  --render-camera robot0_head_camera \
  --out "$COLLECTION_ROOT/raw/$CATEGORY" \
  --vla-stream \
  --vla-camera-names robot0_head_camera robot0_left_wrist_camera robot0_right_wrist_camera \
  --vla-camera-keys ego_view left_wrist right_wrist \
  --vla-camera-width 640 --vla-camera-height 480 \
  --vla-camera-hz 30 --no-vla-camera-flip \
  --rtf-log
```

`--layout -1` samples compatible test layouts. To repeat a scene configuration, set a concrete `--layout` and `--style`; the sampled object instances and instruction can still vary on reset. Record the actual episode metadata rather than assuming the command-line task name fully describes the scene.

### 2. SONIC controller

```bash
cd "$SONIC_ROOT/gear_sonic_deploy"
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager --output-type zmq sim
```

Wait for controller initialization (`Init done`) before recording. The simulator publishes robot state over DDS and receives the controller's joint targets and gains. The controller's ZMQ output supplies the exporter with SONIC telemetry and configuration. RoboCasa owns the simulation clock; this workflow uses its collector as the simulator.

### 3. PICO manager

```bash
cd "$SONIC_ROOT"
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

Confirm that headset, controller, and tracker movements produce the intended robot motion. Use the manager's calibration and engagement procedure for the installed SONIC version. Keep the manager running throughout collection and episode transitions.

### 4. LeRobot exporter

```bash
cd "$SONIC_ROOT"
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK" \
  --dataset-name "robocasa_${TASK}_g1_3cam" \
  --root-output-dir "$COLLECTION_ROOT/lerobot/$CATEGORY" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --sync-robocasa-instruction \
  --no-text-to-speech
```

The task name is only a fallback prompt. A compatible exporter locks the simulator's sampled instruction for each recording attempt. This matters when the instruction names randomly selected objects or destinations.

## Record one complete attempt

1. Wait for a stable robot and live SONIC commands. Recording is blocked until the collector has received valid controller gains.
2. Read the green `Instruction:` line for the new episode. Confirm the exporter receives the same instruction and episode identity.
3. Press `b` in the collector to release the startup elastic band once the robot is balancing. Confirm stable balance before starting the demonstration.
4. Press `c` in the collector terminal to start both recordings. Confirm the exporter matches the episode's start metadata and reports the task text it will record.
5. Perform the sampled instruction. Press `k` to keep the attempt, or `x` to discard it.
6. Wait for both saving and simulator reset to finish. Check the next instruction before recording again.

| Collector key | Effect |
| --- | --- |
| `c` | Start recording while idle |
| `k` | Save the current attempt and reset |
| `x` | Discard the current attempt and reset |
| `b` | Toggle the startup elastic band |

The PICO manager also sends record/save and abort events. Use one control path for an event, and confirm both recorder logs agree. Saving records the operator's choice to keep an attempt; **it does not certify task success**. Raw recorded-state replay establishes the task's success result later.

For `NavigateKitchen`, the viewer shows the target outline and label. These decorations guide the operator and are absent from the exported camera images and recorded scene state.

## Instruction and timing checks

A compatible exporter reports the ready episode, a matched start attempt, and the instruction used for recording. Match their episode ID and text to the collector. If it records only the fallback task name when the collector has a detailed instruction, fix the instruction integration before collecting more demonstrations.

The simulator and WBC loop run at 200 Hz by default. The camera publisher renders at 30 Hz; the exporter writes a 50 Hz timeline using the latest samples. Repeated video frames are therefore expected. Do not use video fps as a measure of simulator control frequency.

Watch the collector's `[real-time]` output with `--rtf-log`: sustained collection should stay near real-time factor 1 and the requested control rate. Save/reset transitions have separate `[sonic-timing]` logs. The collector resynchronizes its clock after reset. Its PGS solver avoids sparse Newton-factorization failures in scenes with dense contacts; leave this recorded setting consistent when replaying.

## Accept the first pair before a longer session

After one save, inspect the files under the printed raw session directory and LeRobot task directory:

- Raw has `episodes/ep_*/ep_demo.hdf5` plus the full local episode sidecars. Preserve the sidecars and timestamps for replay and pairing.
- LeRobot has one corresponding Parquet episode, three camera videos, and metadata with `fps = 50` and the expected 43-state / 64-token / two 7-hand features.
- The sampled instruction agrees between raw episode metadata and the LeRobot task mapping.
- The three videos are correctly oriented and show the same attempt; a short preview of the beginning and end confirms their alignment.
- A [read-only replay and alignment check](dataset_processing.md) can locate the pair without an ambiguous timestamp match.

Keep the raw session, original exporter dataset, software revisions, controller/decoder identifiers, and validation reports together. Process into new output directories so the collection evidence remains available for later review.

The [streaming developer notes](../SONIC_VLA_STREAMING.md) describe transport ports and lifecycle messages when debugging the integration.
