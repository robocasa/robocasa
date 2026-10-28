# RoboCasa-G1 examples

Use these configurations with the [evaluation](../docs/g1/evaluation.md),
[dataset processing](../docs/g1/dataset_processing.md), and
[data collection](../docs/g1/data_collection.md) guides. Run commands from the
repository root; relative configuration paths use that working directory.
For a new simulation environment, follow [G1 installation](../docs/g1/installation.md).

## Inference services and evaluation

| File | Purpose |
| --- | --- |
| [evaluation/http.json](evaluation/http.json) | Connect simulation workers to an existing HTTP inference service; defaults to 50 tasks × 30 episodes, with 40-second atomic and 180-second composite simulation limits. |
| [evaluation/server_custom.json](evaluation/server_custom.json) | Optionally wrap a Python `module:function` adapter in the shared HTTP service. |
| [evaluation/custom.json](evaluation/custom.json) | Load a custom Python adapter directly inside evaluation workers. |
| [evaluation/runtime_versions.json](evaluation/runtime_versions.json) | Recorded simulator, decoder, asset, and dataset revisions. |
| [evaluation/requirements-sim.txt](evaluation/requirements-sim.txt) | Pinned runtime dependencies for the dedicated G1 simulation environment. |

Create a working evaluator configuration:

```sh
mkdir -p evaluation_runs/configs
cp examples/evaluation/http.json evaluation_runs/configs/evaluation.json
```

Set `sim_python` and `decoder_path` to the prepared simulation environment and
SONIC decoder. Set `policy.kwargs.endpoint` to the running service, or override
it with `--endpoint http://127.0.0.1:8000`. Configure outputs with `workspace`.

Inspect the schedule without loading the simulator:

```sh
python -m robocasa_eval plan --config evaluation_runs/configs/evaluation.json
```

Then follow the evaluation guide to run `check`, full-horizon `smoke`, and the
benchmark. The service must implement the documented observation, action, and
HTTP contracts.

For a model that needs a wrapper, the custom configurations reference
[robocasa_eval/examples/custom_policy.py](../robocasa_eval/examples/custom_policy.py).
This template demonstrates lifecycle and action shapes. Replace its placeholder
predictions with inference, preprocessing, and conversion to SONIC motion tokens
and hand targets, then serve it with:

```sh
python -m robocasa_eval serve --config examples/evaluation/server_custom.json \
  --host 127.0.0.1 --port 8000 --devices cuda:0
```

Set `policy.factory` to your importable `module:function`, use `policy.kwargs`
for its constructor options, and list its actual model artifacts in
`policy.artifacts` when they should be content-fingerprinted. Configure more
GPU devices to serve multiple replicas through the same port.

## Paired dataset releases

| File | Purpose |
| --- | --- |
| [datasets/release.json](datasets/release.json) | Build or verify manifests, episode mappings, and checksums for a prepared paired HDF5/LeRobot release. |

Copy the configuration and set `dataset_root`, `pairings_dir`, `release_id`,
and `source_revision`. The example specifies 50 episodes per released task
at 50 Hz. The [processing guide](../docs/g1/dataset_processing.md) covers
replay, pairing, selection, and the `dataset_release build` and `verify`
commands.

## SONIC collection integration

| File | Purpose |
| --- | --- |
| [sonic/README.md](sonic/README.md) | Pinned SONIC checkout, patch application, dependencies, and focused validation commands. |
| [sonic/robocasa_collection.patch](sonic/robocasa_collection.patch) | Sampled-instruction and recording-event handshake between RoboCasa collection and the SONIC exporter. |

Apply the patch to the revision specified in its README, then use the
[collection guide](../docs/g1/data_collection.md) to run the simulator,
controller, PICO manager, and LeRobot exporter.
