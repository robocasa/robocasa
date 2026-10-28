<h1 align="center">RoboCasa for Unitree G1 Evaluation</h1>

Collect paired demonstrations and evaluate vision-language-action (VLA) models
on **Unitree G1 with SONIC** in RoboCasa. This repository provides task
definitions, collection and data-processing tools, and a shared evaluation
interface for model services.

## Benchmark and dataset

| Task group | Tasks | Released demonstrations | Evaluation episodes |
| --- | ---: | ---: | ---: |
| Seen atomic | 25 | 1,250 | 750 |
| Seen composite | 15 | 750 | 450 |
| Unseen composite | 10 | 0 | 300 |
| **Total** | **50** | **2,000** | **1,500** |

Each of the **40 seen tasks has 50 paired HDF5 and LeRobot demonstrations**.
The 10 unseen tasks are reserved for evaluation. The default evaluation runs
30 episodes per task, with simulation-time limits of 40 seconds for atomic
tasks and 180 seconds for composite tasks.

Download the fixed dataset release from
[Dawn-star/robocasa-sonic at `benchmark40-v1-hdf5-20261022`](https://huggingface.co/datasets/Dawn-star/robocasa-sonic/tree/benchmark40-v1-hdf5-20261022).
It contains `hdf5/`, `lerobot/`, and pairing metadata. Both data formats retain
the atomic/composite task grouping; the [dataset guide](docs/g1/datasets.md)
describes downloads, observations, actions, and episode alignment.

## Guides

For a new setup, start with [G1 installation](docs/g1/installation.md).
The [release overview](docs/g1/overview.md) describes the benchmark and workflows.

| Guide | Contents |
| --- | --- |
| [G1 installation](docs/g1/installation.md) | Simulation environment, compatible dependencies, assets, and SONIC decoder |
| [Tasks](docs/g1/tasks.md) | All 50 tasks, split membership, instructions, and success predicates |
| [Datasets](docs/g1/datasets.md) | Fixed dataset version, download commands, and paired data schemas |
| [Data collection](docs/g1/data_collection.md) | PICO/SONIC collection and synchronized HDF5/LeRobot recording |
| [Data processing](docs/g1/dataset_processing.md) | Replay, validation, selection, alignment, and release tools |
| [Policy evaluation](docs/g1/evaluation.md) | Model service, shared VLA interface, concurrent evaluation, and reports |
| [Release provenance](docs/g1/reproducibility.md) | Simulation dependency revisions and deployment validation |

The [examples index](examples/README.md) lists the evaluation configurations,
dataset release configuration, and SONIC collection patch.

## Prepare evaluation

Connect an existing inference service through HTTP. The service receives images,
46-dimensional robot state, and the task instruction, and returns SONIC motion
tokens plus two seven-joint hand targets. Its input preprocessing and output
conversion must implement the [shared protocol](docs/g1/evaluation.md#http-communication-protocol).

Follow [G1 installation](docs/g1/installation.md) to prepare the simulation
environment, compatible dependencies, assets, and SONIC decoder. Then use the
interpreter and decoder paths in the evaluation commands below.

## Evaluation quickstart

From the repository root, create a working evaluator configuration:

```sh
mkdir -p evaluation_runs/configs
cp examples/evaluation/http.json evaluation_runs/configs/evaluation.json
```

Set `sim_python` and `decoder_path` to the prepared simulator interpreter and
SONIC decoder. Set `policy.kwargs.endpoint` to the running inference service,
or pass `--endpoint` on the command line. The default address is
`http://127.0.0.1:8000`.

```sh
python -m robocasa_eval plan --config evaluation_runs/configs/evaluation.json
python -m robocasa_eval check --config evaluation_runs/configs/evaluation.json
python -m robocasa_eval smoke --config evaluation_runs/configs/evaluation.json \
  --envs 4 --full-episodes
python -m robocasa_eval run --config evaluation_runs/configs/evaluation.json
```

For a configuration-free invocation, pass the three settings directly:

```sh
python -m robocasa_eval plan \
  --endpoint http://127.0.0.1:8000 \
  --sim-python ./evaluation_runs/envs/sim/bin/python \
  --decoder-path ./evaluation_runs/models/sonic/model_decoder.onnx
```

Use the same options with `check`, `smoke`, and `run`, replacing the local paths
with your prepared interpreter and decoder.

The service uses **one port**, 8000 by default. Evaluation workers connect to the
same endpoint for concurrent episodes. Reports, logs, and caches use the
configurable `workspace`, defaulting to `./evaluation_runs/`; interrupted runs
can be resumed. `plan` resolves the schedule without loading the simulator.

If a model needs an HTTP wrapper, implement the `module:function` factory shown
in [the adapter guide](docs/g1/evaluation.md#model-adapters) and configure
`examples/evaluation/server_custom.json`. The optional `serve` command supports
batching and multiple GPU replicas behind one endpoint:

```sh
python -m robocasa_eval serve --config examples/evaluation/server_custom.json \
  --host 127.0.0.1 --port 8000 --devices cuda:0
```

Replace the example factory's placeholder actions with the model's inference and
action conversion before evaluation. Choose model devices with `--devices` and
set concurrent environments with `--num-workers N --envs-per-worker 1`.
The [evaluation guide](docs/g1/evaluation.md) covers concurrency,
full-horizon validation, scoring, errors, and resume behavior.

## Credits

This project builds on [RoboCasa](https://robocasa.ai),
[RoboSuite](https://github.com/ARISE-Initiative/robosuite),
[NVIDIA SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl), and the
[Unitree G1](https://www.unitree.com/g1) robot platform. See the
[upstream RoboCasa documentation](https://robocasa.ai/docs/introduction/overview.html)
for general simulator features. The SONIC collection integration and its
upstream license are described in [the companion patch guide](examples/sonic/README.md).

## License

Code: [MIT License](https://opensource.org/license/mit)

Assets and Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

## Citation

**RoboCasa365:**

```bibtex
@inproceedings{robocasa365,
  title={RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author={Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**RoboCasa (Original Release):**

```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```
