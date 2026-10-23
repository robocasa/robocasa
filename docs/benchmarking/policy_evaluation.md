# Unitree G1 Policy Evaluation

This checkout provides a shared evaluator for VLA models controlling **Unitree
G1 with SONIC** in RoboCasa. The task scheduler, simulator, success criterion,
timeouts, and result files are independent of the model architecture. A model
adapter handles loading, image processing, normalization, temporal history,
and conversion to the required G1 action representation.

The default benchmark is **50 tasks × 30 episodes = 1,500 episodes**:

| Task group | Tasks | Episodes | Simulation-time limit per episode |
| --- | ---: | ---: | ---: |
| Seen atomic | 25 | 750 | 40 seconds |
| Seen composite | 15 | 450 | 180 seconds |
| Unseen composite | 10 | 300 | 180 seconds |

The task split comes from the frozen benchmark manifest in
[`robocasa_eval/tasks50.json`](../../robocasa_eval/tasks50.json).
Panda, end-effector-action, and other robot checkpoints require an
explicit embodiment/action adapter and compatible training.

## Status and migration

The previous GROOT N1.7-specific evaluator completed a 1,500-episode evaluation.
Its simulation and SONIC control logic form the basis of this shared runner.
That historical result does **not** validate newly introduced model adapters
or the shared runner as a whole: the refactor has CPU/interface checks, and
each configured real model must pass the new full-horizon smoke test before
a formal run.

The public entry point is now `python -m robocasa_eval`. The former
`robocasa/utils/eval_utils.py` command dispatches to the same CLI instead of
launching random actions. Its `create_eval_env` and `run_random_rollouts`
functions remain available for older raw-RoboSuite callers. Random rollouts
are an environment demonstration, not a model evaluation.

## Installation and runtime

Run the examples below from the repository root. Planning needs only Python's
standard library; it does not import RoboCasa, MuJoCo, NumPy, a model framework,
or model weights:

```sh
python -m robocasa_eval plan --config examples/evaluation/groot.json
```

For execution, use a Python environment with this checkout and the dependencies
required by your chosen adapter. The evaluator's optional dependencies are
available through `pip install -e '.[eval]'`. The existing RoboCasa installation
dependencies remain unchanged; the evaluation extra does not add PyTorch.
Install a model framework and its checkpoint-specific dependencies separately.

The inference process and simulation workers may use different Python
environments. Set `sim_python` to the interpreter with RoboCasa, RoboSuite,
MuJoCo, ONNX Runtime, SONIC, its SDK/reset dependencies, the G1 robot assets,
and the selected rendering backend. `decoder_path` must point to the compatible
SONIC `model_decoder.onnx`. With `sim_root` set, the evaluator uses the pinned
runtime checkouts and source manifest under that directory. With `sim_root`
set to `null`, use packages already installed in `sim_python`'s environment.
The runtime must provide `SonicG1` and the benchmark's registered task classes.

CPU rendering with `renderer: "osmesa"` keeps model GPUs available for
inference. The availability of EGL rendering depends on the selected runtime
and devices; validate it with `check` and `smoke` if selected.

## Configure a run

Start from one of the checked-in configurations:

| Example | Model integration |
| --- | --- |
| [`groot.json`](../../examples/evaluation/groot.json) | GROOT N1.7 Flow trained for Unitree G1 SONIC |
| [`http.json`](../../examples/evaluation/http.json) | External service implementing this project's HTTP protocol |
| [`custom.json`](../../examples/evaluation/custom.json) | Template for a custom Python adapter |

```sh
mkdir -p xuchengxin/evaluation
cp examples/evaluation/groot.json xuchengxin/evaluation/model.json
```

Edit `model.json` before `check`. Evaluator paths and built-in GROOT checkpoint/
repository paths are resolved from the current working directory, not from the
JSON file's directory. Custom factory kwargs pass through unchanged; use absolute
paths for model-specific file arguments, since workers run inside their output
directory. Set an explicit,
descriptive `policy.id` for the model version; do not reuse an ID for changed
remote weights. List additional local model files or directories in
`policy.artifacts` to include their content in the model fingerprint.

GROOT checkpoint files, resolved processor/tokenizer configuration, and the
configured GR00T SDK source and key inference dependency versions are fingerprinted
automatically. For external services,
the configured model revision is a declared identity; the client cannot verify
the server's weight bytes. Keep the serving model fixed for the run. Simulator
runtime fingerprints cover imported source/configuration files (`.py`, `.json`,
`.yaml`, and `.yml`), lightweight robot/scene descriptions (`.xml`, `.urdf`, and
`.srdf`, including those under asset directories), and recorded package versions.
They do not hash large meshes or textures; retain the prepared asset version
separately.

The principal configuration fields are:

| Field | Meaning |
| --- | --- |
| `workspace` | Dedicated writable directory; examples use `./xuchengxin` |
| `run_name` | Unique name for a configuration/model/protocol combination |
| `policy.factory` | `groot`, `http`, or `importable.module:function` |
| `policy.id` | Explicit model version identity |
| `policy.kwargs` | Model-specific constructor options |
| `policy.artifacts` | Optional local files/directories for content fingerprinting |
| `total_gpus` | Number of inference workers across all nodes |
| `envs_per_gpu` | Concurrent simulation workers sharing each model instance |
| `episodes_per_task` | Default 30 |
| `atomic_seconds`, `composite_seconds` | Default 40 and 180 simulation seconds |
| `execution_horizon` | Maximum action steps executed before requesting new inference |
| `sim_python`, `sim_root`, `decoder_path` | Simulation interpreter, optional pinned runtime, and SONIC decoder |
| `renderer`, `camera_names` | Rendering backend and model observation cameras |
| `task_manifest` | Optional task manifest path; `null` selects the bundled benchmark |

Logs, reports, worker directories, temporary files, and caches belong inside
`workspace`. The default workspace capacity limit is **2,000,000,000,000 bytes**;
preflight and periodic checks stop evaluation when the configured limit is
exceeded. They do not delete existing datasets or checkpoints. Custom adapters
must also keep their generated files and caches in the supplied workspace.

## Check, smoke, run, and summarize

```sh
# Resolve the schedule without loading a model or simulator.
python -m robocasa_eval plan --config xuchengxin/evaluation/model.json

# Verify configuration, identities, simulation imports, decoder, and rendering.
python -m robocasa_eval check --config xuchengxin/evaluation/model.json

# Short integration probe; this does not authorize a formal evaluation.
python -m robocasa_eval smoke --config xuchengxin/evaluation/model.json \
  --envs 4 --steps 100

# Single-GPU test using complete 40-second / 180-second horizons.
python -m robocasa_eval smoke --config xuchengxin/evaluation/model.json \
  --envs 4 --full-episodes \
  --tasks CloseDishwasher OpenDrawer CoolBakedCake CupcakeCleanup

# Launch the configured workers after the full-horizon smoke passes.
python -m robocasa_eval run --config xuchengxin/evaluation/model.json

# Rebuild reports from saved episode results.
python -m robocasa_eval summarize --config xuchengxin/evaluation/model.json
```

`check` does not launch the formal benchmark. The full-horizon smoke exercises
real model inference and simulation, including atomic and composite tasks and
multiple simultaneous environments. It is an integration check, not an estimate
of benchmark success rate. A task timeout is a valid completed smoke episode;
an integration error is not.

Changing the model, adapter code, runtime, decoder, camera observations, task
manifest, or scoring protocol invalidates the smoke receipt. Repeat the smoke
after such changes. An earlier receipt from the private GROOT script cannot
authorize this new runner.

Alternative entry points accept the same arguments:

```sh
bash scripts/evaluate_g1.sh plan --config examples/evaluation/groot.json
python robocasa/scripts/evaluate.py plan --config examples/evaluation/groot.json
robocasa-eval plan --config examples/evaluation/groot.json
```

The console command is installed with the package. Use `EVAL_PYTHON` to select
the inference interpreter when invoking the shell wrapper. The direct-file and
`robocasa_eval` entry points avoid importing the heavy `robocasa` package for
planning; `python -m robocasa.scripts.evaluate` remains available in an installed
simulation environment.

### One GPU and 16 GPUs

For a local one-GPU run, use `--total-gpus 1` and a distinct `--run-name`, keeping
`envs_per_gpu` at a value your CPU and model can sustain. After `check` and
full-horizon `smoke` on `model.json`, launch with:

```sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_g1.sh run \
  --config xuchengxin/evaluation/model.json \
  --total-gpus 1 --run-name g1-sonic50-model-1gpu
```

For one node with 16 GPUs, set `total_gpus` to `16` in `model.json`, as in the
examples. For example, four environments per GPU
provide up to 64 concurrent simulations, with one model instance per GPU:

```sh
NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0 \
  bash scripts/evaluate_g1.sh run --config xuchengxin/evaluation/model.json
```

For four nodes with four GPUs each, use the same configuration and shared
workspace on every node, with `NNODES=4`, `GPUS_PER_NODE=4`, and distinct
`NODE_RANK` values `0` through `3`. These are evaluation workers, not distributed
training: each GPU owns a model replica, and ready simulation observations form
dynamic inference batches. Increasing concurrency requires sufficient CPU,
memory, and inference capacity; it does not guarantee higher GPU utilization.

To resume an interrupted run, use the identical configuration and `run_name`.
Completed records with matching identity are reused. Changed configurations
require a new run name; results are never silently mixed across checkpoints.

## Model adapters

The factory receives the selected worker device, seed, and dedicated workspace:

```python
def create_policy(*, device, seed, workspace, **kwargs):
    return MyPolicy(device=device, seed=seed, workspace=workspace, **kwargs)
```

The returned object must implement:

```python
def reset(self, context): ...
def predict_batch(self, observations, contexts): ...
def end_episode(self, context): ...
def close(self): ...
```

`context` is an immutable `EpisodeContext` with `episode_id`, `task`, `episode`,
`seed`, `rank`, and `slot`. It remains stable for the entire episode. Keep
history, recurrent state, action queues, and per-episode RNG keyed by
`context.episode_id`. Batch size and order change between calls, and slots are
reused after episodes end; neither batch position nor slot alone identifies a
trajectory. `reset` initializes only that episode, and `end_episode` releases
only its state. `close` releases remaining model or transport resources.

Return one action mapping per observation, in the same order. Adapters may
return a single action step or a chunk. The runner executes at most the returned
chunk length, `execution_horizon`, and the episode's remaining steps. The
[example adapter](../../robocasa_eval/examples/custom_policy.py) illustrates
state ownership; its placeholder zero outputs are **not a trained policy** and
must not be presented as a model result.

### Observation contract

Each observation contains:

| Key | Representation |
| --- | --- |
| `video` | Camera-name mapping to HWC, RGB, `uint8` images |
| `state.left_leg`, `state.right_leg` | 6 values each |
| `state.waist` | 3 values |
| `state.left_arm`, `state.right_arm` | 7 values each |
| `state.left_hand`, `state.right_hand` | 7 values each, published exporter state order |
| `state.projected_gravity` | 3 values |
| `language` | Plain task instruction string sampled for the episode |

State arrays are unnormalized `float32`. The default view is `ego_view`;
available calibrated aliases also include `left_wrist` and `wrist_view`.
Model-specific resizing, history stacking, language tokenization, and
normalization belong in the adapter. Privileged object state and success
signals are not model observations.

### Action contract

Each output mapping must contain exactly these finite arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `motion_token` | `(T, 64)` | Unnormalized SONIC motion latent |
| `left_hand_joints` | `(T, 7)` | Absolute hand joint targets in SONIC/DDS motor-slot order |
| `right_hand_joints` | `(T, 7)` | Absolute hand joint targets in SONIC/DDS motor-slot order |

All three arrays share the same positive `T`. The adapter must undo training
normalization. State hand ordering and action hand ordering differ by the
dataset/exporter convention; do not copy the state reorder onto action slots.
Unsupported action keys, empty chunks, mismatched lengths, NaNs, and infinities
are errors, not benchmark timeouts.

SONIC runs at **50 Hz**, with four **200 Hz** physics steps per token. Its
decoder history and closed-loop motor observations remain inside each isolated
simulation worker. The model-specific inference rate can vary through chunk
length without changing the physical scoring horizon.

### Built-in GROOT adapter

Set `policy.factory` to `"groot"` and provide `checkpoint`, `repo`, and
`inference_steps` in `policy.kwargs`. This adapter validates the N1.7 Flow,
UNITREE_G1_SONIC processor, normalization statistics, and trained action
horizon. Model files and preprocessing assets must already be available; it
uses workspace-local caches and does not download missing assets during a run.

The current GROOT adapter uses process-seeded Flow noise. Scene seeds are
stable, but dynamic batch membership can change model RNG draws. Do not claim
bitwise-identical trajectories across different worker/concurrency settings.

### HTTP bridge

Set `policy.factory` to `"http"`. Its kwargs are `endpoint`, optional `timeout`
(seconds), and optional `auth_token_env` (the name of an environment variable
holding a bearer token). Do not put a token in the configuration or URL.

This is the project-specific **`robocasa-g1-v1`** JSON protocol. Existing Openpi
WebSocket or GROOT model servers need a bridge implementing it. Each endpoint
accepts a JSON `POST`; every request and response contains
`"protocol": "robocasa-g1-v1"`:

| Endpoint suffix | Request fields besides `protocol` | Response fields besides `protocol` |
| --- | --- | --- |
| `/reset` | `context` | `ok: true` |
| `/predict` | `observations`, `contexts` | `actions`, in matching input order |
| `/end_episode` | `context` | `ok: true` |

Contexts serialize to objects with the six `EpisodeContext` fields. Arrays use
four fields: `__ndarray__: true`, NumPy `dtype` string, `shape` list, and `data`
containing base64-encoded C-order bytes. For example, motion tokens use dtype
`"<f4"` and shape `[T, 64]`; RGB images use `"|u1"` and `[H, W, 3]`. Numeric
JSON lists are also accepted for actions. The server must return finite,
denormalized action values.

Use `encode_value` and `decode_value` from
[`robocasa_eval.adapters.http`](../../robocasa_eval/adapters/http.py) when
implementing a Python bridge. A shared server must isolate concurrent episode
histories and support multiple rank clients. Configure enough serving capacity
for the intended worker count; a 16-worker client alone does not distribute the
server's model across 16 GPUs.

## Results and scoring

Outputs are stored under `workspace/eval_results/run_name/`, including the
resolved `config.json`, individual episode records, worker logs/status,
`episodes.csv`, `tasks.csv`, and `summary.json`.

- **Success:** the first positive native task reward within the simulation-time
  limit. The evaluator does not make a second success query that could mutate
  task bookkeeping.
- **Timeout:** the simulation-time budget is exhausted without success.
- **Error:** simulation, inference, or transport failed. A wall-clock watchdog
  protects stuck operations; it is separate from the task time budget.
- **Pending:** no terminal record exists yet, including an interrupted episode.

The primary success rate uses all planned episodes as its denominator. Errors
and pending episodes are reported separately; a valid-only rate must not be
presented as the full benchmark rate. Reports include task and seen/unseen
breakdowns. Success can occur on the first physics step for some native task
initializations; preserve the result and inspect its recorded timing/metadata
when interpreting behavior. No source dataset is rewritten by evaluation.
