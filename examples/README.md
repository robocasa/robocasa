# RoboCasa-G1 examples

Use these configurations with the [evaluation](../docs/g1/evaluation.md),
[dataset processing](../docs/g1/dataset_processing.md), and
[data collection](../docs/g1/data_collection.md) guides. Run commands from the
repository root; relative configuration paths use that working directory.

## Model serving and evaluation

| File | Purpose |
| --- | --- |
| [evaluation/server_groot.json](evaluation/server_groot.json) | Serve a GROOT N1.7 Flow checkpoint through one HTTP port, with configurable GPU replicas and batching. |
| [evaluation/http.json](evaluation/http.json) | Connect simulation workers to the model service; defaults to 50 tasks × 30 episodes, with 40-second atomic and 180-second composite simulation limits. |
| [evaluation/server_custom.json](evaluation/server_custom.json) | Serve the custom Python adapter template through the same HTTP protocol. |
| [evaluation/groot.json](evaluation/groot.json) | Load GROOT directly inside evaluation workers. |
| [evaluation/custom.json](evaluation/custom.json) | Load a custom Python adapter directly inside evaluation workers. |
| [evaluation/baseline_results.json](evaluation/baseline_results.json) | Frozen historical GROOT baseline: protocol, run identity, and aggregate results for all 50 tasks. |
| [evaluation/runtime_versions.json](evaluation/runtime_versions.json) | Recorded simulator, decoder, asset, dataset, and GROOT inference dependency revisions. |

The custom configurations reference
[robocasa_eval/examples/custom_policy.py](../robocasa_eval/examples/custom_policy.py),
which demonstrates the adapter lifecycle and action shapes. Replace its
placeholder predictions with the model's inference and action conversion.

Copy the server and client examples into working configurations:

```sh
mkdir -p evaluation_runs/configs
cp examples/evaluation/server_groot.json evaluation_runs/configs/server.json
cp examples/evaluation/http.json evaluation_runs/configs/evaluation.json
```

Set the server's `policy.kwargs.checkpoint` and `policy.kwargs.repo` to the local
checkpoint and GR00T SDK checkout. Set the evaluator's `sim_python` and
`decoder_path` to the prepared simulation environment and SONIC decoder. Use
the same immutable `policy.id` in both files, and point the evaluator's
`policy.kwargs.endpoint` at the service. Configure outputs with `workspace`.

Inspect the resolved task schedule without loading a model or simulator:

```sh
python -m robocasa_eval plan --config evaluation_runs/configs/evaluation.json
```

Then follow the evaluation guide to start the service, run `check` and
full-horizon `smoke`, and launch the benchmark.

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
