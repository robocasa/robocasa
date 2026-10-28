# RoboCasa SONIC VLA streaming: developer notes

The [G1 collection guide](g1/data_collection.md) contains the operator workflow and four terminal commands. This page describes the transport and recording lifecycle for debugging that integration.

RoboCasa owns the MuJoCo clock. Its collector publishes robot state over DDS, receives SONIC joint commands and gains, and streams rendered images without moving rendering into the 200 Hz control loop. The SONIC exporter records the latent-action representation alongside those observations.

## Transport

`robocasa/scripts/collect_sonic_demos.py --vla-stream` enables the following integration:

| Port | Publisher | Consumer | Contents |
| ---: | --- | --- | --- |
| 5555 | RoboCasa | SONIC exporter | Cameras using SONIC's `ImageMessageSchema` |
| 5556 | PICO manager | SONIC controller, collector, exporter | Pose/planner inputs and `manager_state` events |
| 5557 | SONIC controller | SONIC exporter | State and configuration streams |
| 5580 | RoboCasa | SONIC exporter | Keyboard recording synchronization |
| 5581 | RoboCasa | SONIC exporter | Sampled episode instruction and recording attempt metadata |

The controller must publish its exporter telemetry (the documented command uses `--output-type zmq`) in addition to the simulation DDS interface. The exporter must implement the RoboCasa instruction protocol and the requested camera schema. Check the installed SONIC version's CLI and capabilities before a collection session.

The default single camera is `robot0_head_camera`, published as `ego_view` at 640 × 480 and 30 Hz. The collector currently defaults to **no vertical flip**; the public three-camera command makes that setting explicit. Collision geoms are hidden and visual geoms are rendered. Inspect a preview whenever changing a renderer or camera configuration.

A separate image process renders camera frames. The collector copies the latest MuJoCo state into a bounded queue, keeping rendering outside the control-loop critical path. Camera publication is independent of the exporter's 50 Hz sampling and video timeline.

## Instruction handshake

After a successful reset, the collector captures the sampled episode metadata and instruction once. A `ready` message is repeated on port 5581. A record event creates a versioned `start` attempt carrying the same episode ID and instruction. The exporter waits for the matching start metadata and locks the text before recording the first frame.

For an object-dependent task such as `LoadDishwasher`, the instruction can name the actual sampled objects. The exporter should use that text as the LeRobot task label. `--task-prompt` is a fallback for a missing instruction or an exporter configuration without live synchronization.

Check that the compatible exporter reports the episode ID and matched instruction at ready/start/recording transitions. Example log messages from the integration are:

```text
[EpisodeInstruction] ready: <session:sequence> attempt=0 -> <sampled instruction>
[EpisodeInstruction] start matched: <session:sequence> attempt=1 -> <sampled instruction>
[EpisodeInstruction] recording task from RoboCasa <session:sequence>: <sampled instruction>
```

Collector overrides include:

```text
--vla-camera-name robot0_head_camera
--vla-camera-key ego_view
--vla-camera-names robot0_head_camera robot0_left_wrist_camera robot0_right_wrist_camera
--vla-camera-keys ego_view left_wrist right_wrist
--vla-camera-width 640
--vla-camera-height 480
--vla-camera-hz 30
--no-vla-camera-flip
--no-vla-keyboard-sync
--vla-instruction-port 5581
--no-vla-instruction-sync
```

Supply the plural camera-name and key lists together, with equal lengths and unique keys. The singular settings describe the single-camera case. Exporter versions with the instruction integration expose `--instruction-zmq-host`, `--instruction-zmq-port`, `--instruction-wait-timeout`, and `--sync-robocasa-instruction`; consult that checkout's help for the supported interface.

## Recording events and resets

The local collector keys are `c` to start, `k` to save, `x` to discard, and `b` to toggle the startup elastic band. VR/PICO sends `toggle_data_collection` (start when idle, save when recording) and `toggle_data_abort` (discard). The collector's keyboard forwarding translates local events into the exporter's recording lifecycle.

`manager_state` includes a manager session ID, monotonic event counters, and event timestamps. This lets consumers discard stale continuous state after a long save/reset while retaining a short start/save/discard button edge. These fields control recording lifecycle; they are not model observations.

Each streamed `pose` message contains a complete sliding window of five PICO frames. SONIC merges the window by `frame_index`. Independent latest-only subscribers for `pose`, `planner`, and `manager_state` prevent one topic from replacing another in a conflated queue.

The startup elastic band's orientation reference is the reset/spawn pelvis pose, so a fixture-facing spawn is not pulled toward world-frame yaw. Recording waits for a real SONIC command and valid gains. Press `b` to release the band once the controller balances the robot.

After save/discard, `[sonic-timing]` logs separate finalization, environment reset, and source/instruction setup. The collector resynchronizes the wall-clock deadline when the next episode is ready. It does not try to catch up deadlines that elapsed during a hard reset. Saving an episode retains it for later validation; it does not evaluate task success.

## Navigation annotation

`NavigateKitchen` shows a target outline, arrow, and name in the interactive viewer by default. Pass `--no-nav-target-marker` to disable it. The marker follows the sampled target after resets and viewer recreation.

The decoration exists only in the passive viewer's scene and overlay. It does not change model XML, simulation state, or the camera publisher's render scene. Exported videos remain unannotated.

Task success uses `NavigateKitchen._check_success()`: for SonicG1, the pelvis must be within 25 cm of the target base placement in the horizontal plane, with heading cosine at least 0.98 (yaw error at most about 11.48 degrees). The distance is to the target base placement, not the fixture surface. This criterion is shared by collection, evaluation, and recorded-state replay.

## Validation

Run the focused RoboCasa streaming tests from the repository root in the simulation environment:

```bash
MUJOCO_GL=egl python -m pytest tests/test_sonic_vla_streaming.py -q
```

For an integration test, start the four processes using the public collection guide, record one short attempt, and verify the raw HDF5, three exporter videos, metadata schema, matched instruction, and raw/exporter pairing. Check sustained `[real-time]` output near RTF 1 at the requested control frequency, and inspect the transition logs around save/reset. A unit-test result alone does not verify the installed SONIC exporter, hardware input, or live timing.
