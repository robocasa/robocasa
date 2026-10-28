# Policy Learning Algorithms

Use the [shared Unitree G1 policy evaluator](../g1/evaluation.md) to connect an
existing inference service over HTTP, on port 8000 by default. The service
implements the common image/state/language input and SONIC action protocol;
the evaluator manages the task schedule, simulation budgets, and reports.

```sh
# Simulation environment: replace the interpreter and decoder paths.
python -m robocasa_eval plan \
  --endpoint http://127.0.0.1:8000 \
  --sim-python ./evaluation_runs/envs/sim/bin/python \
  --decoder-path ./evaluation_runs/models/sonic/model_decoder.onnx
```

The same three options apply to `check`, `smoke`, and `run`. For repeated use,
set these values in [the HTTP example](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/http.json),
then run:

```sh
python -m robocasa_eval check --config examples/evaluation/http.json
python -m robocasa_eval smoke --config examples/evaluation/http.json --envs 4 --full-episodes
python -m robocasa_eval run --config examples/evaluation/http.json
python -m robocasa_eval summarize --config examples/evaluation/http.json
```

Run commands from the repository root. The default evaluation covers 50 tasks ×
30 episodes, with 40-second atomic and 180-second composite limits measured in
simulation time. The optional `serve` wrapper accepts a Python `module:function`
adapter and can place multiple GPU replicas behind the same endpoint. See the
[adapter and concurrency guide](../g1/evaluation.md#model-adapters) for input
preprocessing, action conversion, and lifecycle requirements.

The remaining sections preserve the upstream RoboCasa training references:
[Diffusion Policy](https://github.com/robocasa-benchmark/diffusion_policy),
[Openpi](https://github.com/robocasa-benchmark/openpi), and
[GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T). Their original training
recipes and robot embodiments are separate from the G1 evaluation protocol.

-------
## Diffusion Policy
We fork the official Diffusion Policy code base, hosted at [https://github.com/robocasa-benchmark/diffusion_policy](https://github.com/robocasa-benchmark/diffusion_policy).
### Recommended system specs
For training we recommend a GPU with at least 24 Gb of memory, but 48 Gb+ is prefered.
For inference we recommend a GPU with at least 8 Gb of memory.

### Installation
```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy
pip install -e .
```

### Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- G1 evaluation: [shared policy evaluator](../g1/evaluation.md)

### Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

```

After training a compatible G1 SONIC policy, implement its inference adapter
and use the shared evaluator above.


-------
## Openpi
We fork the official Openpi code base, hosted at [https://github.com/robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi). Our fork support training for **pi0**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/openpi
cd openpi
pip install -e .
pip install -e packages/openpi-client/
```

### Key files
- Training: [scripts/train.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/train.py)
- G1 evaluation: [shared policy evaluator](../g1/evaluation.md)
- Setting up configs: [src/openpi/training/config.py](https://github.com/robocasa-benchmark/openpi/blob/main/src/openpi/training/config.py)

### Experiment workflow
```
# train model
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
<dataset-soup> \
--exp-name=<exp-name>

```

A pi inference service needs a Python adapter or a bridge implementing the
shared evaluator's HTTP contract; changing only a native server URL is not
sufficient.

-------
## GR00T
We fork the official GR00T code base, hosted at [https://github.com/robocasa-benchmark/Isaac-GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T). Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- G1 evaluation: [shared HTTP policy evaluator](../g1/evaluation.md)

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

```

The upstream training recipe above is retained as a reference. Serving its
outputs in G1 evaluation requires the shared observation and action contract.
