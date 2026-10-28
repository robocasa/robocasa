# Policy Learning Algorithms

This checkout evaluates models using the
[shared Unitree G1 policy evaluator](../g1/evaluation.md). Run a model server
on port 8000 in the model environment, then connect the simulation environment
to that port. Both sides use a common observation/action protocol, task schedule,
simulation-time budget, success definition, and result format.

```sh
# Model environment: start the service and leave it running.
python -m robocasa_eval serve --config examples/evaluation/server_groot.json \
  --host 127.0.0.1 --port 8000 --devices cuda:0

# Simulation environment, in a second terminal.
python -m robocasa_eval plan --config examples/evaluation/http.json
python -m robocasa_eval check --config examples/evaluation/http.json
python -m robocasa_eval smoke --config examples/evaluation/http.json --envs 4 --full-episodes
python -m robocasa_eval run --config examples/evaluation/http.json
python -m robocasa_eval summarize --config examples/evaluation/http.json
```

Fill in the model and simulator paths in the
[example configurations](https://github.com/Xucxname/robocasa/tree/agent/sonic-g1-followups/examples/evaluation/) and set matching `policy.id`
values before execution. Run these commands from the repository root. The
default protocol is 50 tasks × 30 episodes, with 40 seconds for atomic tasks
and 180 seconds for composite tasks, measured in simulation time. After a
single-GPU full-horizon smoke, restart the service with `--devices cuda:{0..15}`
in Bash to use 16 model GPUs behind the same port.

The built-in GR00T adapter targets N1.7 Flow trained for G1 SONIC. Other VLA
architectures, including Diffusion Policy and pi, can provide a Python factory
to the shared server or implement the HTTP protocol. Models produce the same
G1 SONIC action representation. The service uses JSON HTTP; existing native
Openpi WebSocket servers need the corresponding protocol adapter. See the
[adapter contract](../g1/evaluation.md#model-adapters).

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
- G1 evaluation: [shared policy evaluator](../g1/evaluation.md), including the N1.7 Flow adapter

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

```

The upstream N1.5 training recipe above is retained as a reference. It is not
the N1.7 G1 checkpoint format expected by the built-in evaluation adapter.
