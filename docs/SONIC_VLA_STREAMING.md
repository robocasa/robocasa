# RoboCasa SONIC VLA Streaming

This path lets the RoboCasa SONIC collector produce the same ZMQ streams that
`gear_sonic/scripts/run_data_exporter.py` expects during real-world VLA
collection. RoboCasa remains the owner of the MuJoCo clock and the DDS-driven
SONIC controller remains unchanged.

## What it publishes

Running `robocasa/scripts/collect_sonic_demos.py --vla-stream` adds three VLA
integration pieces:

- A camera stream on port `5555` using SONIC's `ImageMessageSchema`.
- A subscriber for VR/PICO `manager_state` toggles on port `5556`.
- A keyboard publisher on port `5580` so local collector hotkeys can keep
  `run_data_exporter.py` in sync.

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

## Terminals

Run these in separate terminals.

### 1. Start the SONIC controller

Use the same controller command used for normal SONIC RoboCasa collection. For
example, from the SONIC deploy tree:

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check
```

### 2. Start the VLA exporter

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "<task prompt>" \
  --dataset-name <dataset_name> \
  --root-output-dir /tmp/sonic_vla_exports \
  --no-text-to-speech
```

The exporter consumes the RoboCasa `ego_view` camera stream on port `5555`, the
VR/PICO manager stream on port `5556`, SONIC state/config streams from the
controller, and keyboard sync on port `5580`.

### 3. Start RoboCasa collection with VLA streaming

Use an interactive display for real collection. Do not set `MUJOCO_GL=egl` for
normal VR collection because the collector viewer is interactive.

```bash
cd /home/amaddukuri/Projects/robocasa-dev
/home/amaddukuri/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python \
  robocasa/scripts/collect_sonic_demos.py \
  --environment Kitchen \
  --layout 1 \
  --robot SonicG1 \
  --out /tmp/sonic_robocasa_demos \
  --vla-stream
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
```

## Episode controls

Local collector hotkeys still work:

- `c`: start recording in RoboCasa and notify the exporter.
- `k`: save the RoboCasa episode and notify the exporter to stop/save.
- `x`: discard the RoboCasa episode and notify the exporter to abort.
- `b`: toggle the startup elastic band after SONIC is balancing.

VR/PICO `manager_state` toggles are also consumed:

- `toggle_data_collection`: start when idle, save when recording.
- `toggle_data_abort`: discard the current episode.

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
