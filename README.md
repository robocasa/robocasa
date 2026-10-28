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

Start with the [release overview](docs/g1/overview.md).

| Guide | Contents |
| --- | --- |
| [Tasks](docs/g1/tasks.md) | All 50 tasks, split membership, instructions, and success predicates |
| [Datasets](docs/g1/datasets.md) | Fixed dataset version, download commands, and paired data schemas |
| [Data collection](docs/g1/data_collection.md) | PICO/SONIC collection and synchronized HDF5/LeRobot recording |
| [Data processing](docs/g1/dataset_processing.md) | Replay, validation, selection, alignment, and release tools |
| [Policy evaluation](docs/g1/evaluation.md) | Model service, shared VLA interface, concurrent evaluation, and reports |
| [Release provenance](docs/g1/reproducibility.md) | Runtime revisions, model dependencies, and the completed GROOT baseline |

The [examples index](examples/README.md) lists the evaluation configurations,
dataset release configuration, and SONIC collection patch.

## Prepare the evaluation environments

Use separate environments for the **model service** and **simulation client**.
The model environment needs the selected VLA's dependencies and checkpoint.
The simulation environment needs RoboSuite, MuJoCo, the G1 assets, SONIC, and
its ONNX decoder. Install the evaluation package from this checkout with
`python -m pip install -e '.[eval]'` in the simulation environment.

The SonicG1-compatible RoboSuite dependency is
[`Xucxname/robosuite` at `67c18a4c66fa7bca02258a8a6aee2abf29e5427f`](https://github.com/Xucxname/robosuite/tree/67c18a4c66fa7bca02258a8a6aee2abf29e5427f).
Follow [environment preparation](docs/g1/evaluation.md#prepare-the-two-environments)
and [runtime revisions](docs/g1/reproducibility.md) for the remaining dependency,
decoder, and offline model-cache requirements.

## Evaluation quickstart

From the repository root, create working configurations:

```sh
mkdir -p evaluation_runs/configs
cp examples/evaluation/server_groot.json evaluation_runs/configs/server.json
cp examples/evaluation/http.json evaluation_runs/configs/evaluation.json
```

Set the checkpoint and GR00T SDK paths in `server.json`; set `sim_python` and
`decoder_path` in `evaluation.json`. Use the same `policy.id` in both files.
The [evaluation guide](docs/g1/evaluation.md) describes all settings.

Start the GROOT N1.7 model service in the model environment:

```sh
python -m robocasa_eval serve --config evaluation_runs/configs/server.json \
  --host 127.0.0.1 --port 8000 --devices cuda:0
```

In another terminal, use the simulation environment to connect and evaluate:

```sh
python -m robocasa_eval plan --config evaluation_runs/configs/evaluation.json
python -m robocasa_eval check --config evaluation_runs/configs/evaluation.json
python -m robocasa_eval smoke --config evaluation_runs/configs/evaluation.json \
  --envs 4 --full-episodes
python -m robocasa_eval run --config evaluation_runs/configs/evaluation.json
```

The model exposes **one port**, 8000 by default. Set `--devices cuda:{0..15}`
in Bash to serve 16 GPU replicas behind it, and configure client worker and
environment counts to run concurrent episodes. The service batches inference
requests. Reports, logs, and caches use the configurable `workspace`, defaulting
to `./evaluation_runs/`; interrupted evaluations can be resumed.

GROOT N1.7 has a built-in adapter. Other VLAs can implement the shared Python
adapter or HTTP protocol, producing SONIC motion tokens and G1 hand joint
targets. `plan` works without loading a model or simulator.

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
