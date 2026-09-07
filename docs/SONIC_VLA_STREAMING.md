# RoboCasa SONIC VLA Streaming

This path lets the RoboCasa SONIC collector produce the same ZMQ streams that
`gear_sonic/scripts/run_data_exporter.py` expects during real-world VLA
collection. RoboCasa remains the owner of the MuJoCo clock and the DDS-driven
SONIC controller remains unchanged.

## What it publishes

Running `robocasa/scripts/collect_sonic_demos.py --vla-stream` adds four VLA
integration pieces:

- A camera stream on port `5555` using SONIC's `ImageMessageSchema`.
- A subscriber for VR/PICO `manager_state` toggles on port `5556`.
- A keyboard publisher on port `5580` so local collector hotkeys can keep
  `run_data_exporter.py` in sync.
- A state-style metadata publisher on port `5581` that forwards the exact
  instruction sampled for the current RoboCasa episode.

The instruction is captured once after each successful reset. A `ready` state
is repeated on port `5581`; pressing record changes it to a versioned `start`
attempt carrying the same episode ID and instruction. The exporter waits for
that start metadata before entering `RECORDING`, then locks the text before the
episode's first frame.
For object-dependent tasks such as `LoadDishwasher`, a generated instruction
like `Pick up the cup and bowl ...` therefore becomes the LeRobot task label
automatically. `--task-prompt` remains a fallback for an older collector, a
missing/blank task instruction, or non-RoboCasa collection.

Default camera settings match the real VLA path: `robot0_head_camera` is
published as `ego_view` at `640x480`, `30 Hz`, with the MuJoCo image vertically
flipped. Collision geoms are hidden and visual geoms are rendered.

The image renderer runs in a separate process. The 200 Hz collection loop only
copies the latest MuJoCo state into a bounded queue, so image streaming is not in
the control-loop critical path.

During startup, the controller startup band remains enabled, but its orientation
reference is the reset/spawn pelvis pose so fixture-facing spawn yaw is not
pulled back toward the world frame.
Recording remains blocked until that real command arrives, so saved demos still
contain real SONIC gains and q-star targets.

## VR workflow terminals

Run these in separate terminals for RoboCasa VR collection. This mirrors the
manual VR workflow in `dc_ft_docs.md`, except RoboCasa replaces
`gear_sonic/scripts/run_sim_loop.py` as the MuJoCo simulator and camera
publisher. Do not start both sim loops at the same time.

### 1. Start RoboCasa collection with VLA streaming

Use an interactive display for real collection. Do not set `MUJOCO_GL=egl` for
normal VR collection because the collector viewer is interactive.

```bash
cd /home/amaddukuri/Projects/robocasa-dev-sonic-vla
/home/amaddukuri/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python \
  robocasa/scripts/collect_sonic_demos.py \
  --environment Kitchen \
  --layout -1 \
  --robot SonicG1 \
  --out /tmp/sonic_robocasa_demos \
  --vla-stream
```

This process owns the RoboCasa MuJoCo sim, publishes DDS lowstate/odometry to
the SONIC controller, receives DDS lowcmd actions back from the controller, and
publishes simulated ego camera frames for the exporter on port `5555`.

### 2. Start the SONIC controller

This follows the C++ deployment terminal from `dc_ft_docs.md`: source the deploy
environment, use `zmq_manager` input, and pass `sim` as the final mode argument.

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

The `zmq_manager` input subscribes to the PICO manager stream on port `5556`;
the trailing `sim` selects the sim/DDS backend instead of the real robot backend.

### 3. Start the PICO manager

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

The PICO manager publishes VR pose/planner inputs and `manager_state` recording
toggles on port `5556`. Keep this process running for both teleop control and
episode start/save/discard events.

Each streamed `pose` message remains a complete sliding window of five PICO
frames. SONIC merges that window by `frame_index`; latest-only transport means
the newest complete window, not a one-frame pose. The exporter uses independent
latest-only subscribers for `pose`, `planner`, and `manager_state`, so those
topics cannot replace one another in a shared conflated queue.

`manager_state` repeats a manager session ID, monotonic event counters, and the
last event timestamps. This allows the collector and exporter to discard stale
continuous state after a long save/reset while still observing a one-frame
start/save/discard button edge. These fields affect recording lifecycle only;
they are not part of SONIC's policy observation.

### 4. Start the VLA exporter

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "<fallback task prompt>" \
  --dataset-name <dataset_name> \
  --root-output-dir /tmp/sonic_vla_exports \
  --no-text-to-speech
```

The exporter consumes the RoboCasa `ego_view` camera stream on port `5555`, the
VR/PICO stream on port `5556`, SONIC state/config streams from the controller
on port `5557`, keyboard sync on port `5580`, and RoboCasa episode instructions
on port `5581`.

Before recording, confirm that the exporter prints both lines below. The second
line is the task text that will be written to each frame in that episode:

```text
[EpisodeInstruction] ready: <session:sequence> attempt=0 -> <sampled instruction>
[EpisodeInstruction] start matched: <session:sequence> attempt=1 -> <sampled instruction>
[EpisodeInstruction] recording task from RoboCasa <session:sequence>: <sampled instruction>
```

Common overrides:

```bash
--vla-camera-name robot0_head_camera
--vla-camera-key ego_view
--vla-camera-width 640
--vla-camera-height 480
--vla-camera-hz 30
--no-vla-camera-flip
--no-vla-keyboard-sync
--vla-instruction-port 5581
--no-vla-instruction-sync
```

Exporter-side overrides are `--instruction-zmq-host`,
`--instruction-zmq-port`, `--instruction-wait-timeout`, and
`--no-sync-robocasa-instruction`. Use the last option with an older RoboCasa
collector that does not publish port `5581`; the existing `--task-prompt` is
then used immediately.

## Episode controls

Local collector hotkeys still work:

- `c`: start recording in RoboCasa and notify the exporter.
- `k`: save the RoboCasa episode and notify the exporter to stop/save.
- `x`: discard the RoboCasa episode and notify the exporter to abort.
- `b`: toggle the startup elastic band after SONIC is balancing.

VR/PICO `manager_state` toggles are also consumed:

- `toggle_data_collection`: start when idle, save when recording.
- `toggle_data_abort`: discard the current episode.

After `k` or `x`, `[sonic-timing]` lines split the transition into finalize,
environment reset, and source/instruction phases. The collection clock is
resynchronized when the new episode is ready, so the 200 Hz loop does not try
to catch up wall-clock deadlines that expired during a hard reset. Camera
publication remains `640x480` at `30 Hz`.

## Real-time check

The renderer was tested on a hard scene with `DivideBuffetTrays`, layout `25`,
style `11`, at `200 Hz` control. Baseline paced collection and VLA-streamed
collection both held `RTF = 1.0` and achieved `200 Hz`; the VLA process
published frames at the requested camera rate without blocking the sim loop.

## Smoke tests

Focused unit tests:

```bash
cd /home/amaddukuri/Projects/robocasa-dev
MUJOCO_GL=egl /home/amaddukuri/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m pytest tests/test_sonic_vla_streaming.py -q
```

Exporter smoke test shape:

1. Start mock VR/PICO `manager_state` messages on port `5556`.
2. Start `gear_sonic/scripts/run_data_exporter.py`.
3. Start `collect_sonic_demos.py --vla-stream`.
4. Toggle start/save and verify the exporter writes
   `observation.images.ego_view` videos.

The latest local smoke test wrote two exporter episodes with `98` total frames
and `observation.images.ego_view` videos.
