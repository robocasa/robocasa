# Release provenance and validation

Record the dataset selection, simulator, inference-service fingerprint, and
task-scoring configuration with the generated run manifest. This page records
dataset versions, simulation dependencies, and deployment verification steps.

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

These simulation dependency revisions were recorded on 2026-10-20. They are
also available as the machine-readable
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

## Validate a new deployment

Follow [Evaluation](evaluation.md) from a new checkout and a prepared local GPU
environment. Record the checkout commit and service fingerprint,
decoder hash, simulator identity, device type, and output directory.

1. Run `plan` and `check` with your configured paths and a running model service.
2. Run a short single-GPU smoke with multiple simulator environments.
3. Run the full-horizon single-GPU smoke covering atomic and composite tasks.
4. Inspect episode errors, service logs, simulator logs, and the smoke receipt.
5. Expand model replicas and client workers, then run the desired benchmark.

Keep the verification reports with the evaluation run. Include the serving
implementation and artifact revisions, runtime revisions, task manifest, seeds,
and scoring protocol when sharing benchmark results.
