# RoboCasa collection integration for SONIC

`robocasa_collection.patch` adds the sampled-instruction and recording-event
handshake used by `docs/g1/data_collection.md`. Apply it to
[`NVlabs/GR00T-WholeBodyControl` at `a9d20b2ac0949244d94461a1a3263f38c5027c4a`](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/a9d20b2ac0949244d94461a1a3263f38c5027c4a).
The base commit already supports 50 Hz export and wrist cameras.

Use a dedicated clean checkout at that commit. With `ROBOCASA_ROOT` pointing to
this repository and `SONIC_ROOT` to the prepared SONIC checkout:

```bash
cd "$SONIC_ROOT"
test "$(git rev-parse HEAD)" = a9d20b2ac0949244d94461a1a3263f38c5027c4a &&
git apply --check "$ROBOCASA_ROOT/examples/sonic/robocasa_collection.patch" &&
git apply "$ROBOCASA_ROOT/examples/sonic/robocasa_collection.patch"
```

The patch changes two existing scripts and adds an instruction subscriber and
its tests. It locks the sampled prompt to a recording attempt, waits for matching
start metadata, gives abort events priority, and retains record/abort counters
across conflated PICO manager updates. It does not add a training implementation
or change deployment thresholds, ROS2 build configuration, pose-frame selection,
or the timing monitor.

Install the pinned SONIC repository's controller and teleoperation prerequisites.
The exporter uses the `gear_sonic[data_collection]` extra, including PyZMQ, Tyro,
LeRobot, PyAV, OpenCV, and Pinocchio. Its upstream
`install_scripts/install_data_collection.sh` creates `.venv_data_collection`;
that setup script replaces an existing environment with the same name. The PICO
process uses the separately prepared `.venv_teleop` environment. Apply the patch
before running the documented four processes.

The focused tests require Python 3.10 or newer, `pytest`, and `pyzmq`:

```bash
cd "$SONIC_ROOT"
PYTHONPATH="$SONIC_ROOT" python -m pytest -q \
  gear_sonic/tests/test_episode_instruction_subscriber.py
```

Validation for this bundle applied the patch to an isolated snapshot of the
pinned source, checked Python syntax, and passed all 16 focused tests. It did not
validate a live PICO headset, compile the controller, or run a complete four-process
recording session. Preserve the base commit and patch SHA-256 with collection
provenance.

This patch was extracted from the local RoboCasa collection integration on
2026-10-28. It is distributed as a RoboCasa companion patch; the full integration
has not been established as an upstream SONIC commit. Source code remains under
the upstream Apache-2.0 terms described in its
[`LICENSE`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/a9d20b2ac0949244d94461a1a3263f38c5027c4a/LICENSE).
Existing source headers are retained. No model weights or binaries are included.
