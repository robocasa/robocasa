# Unitree G1 Policy Evaluation

For the released task definitions and demonstration formats, see
[Tasks](tasks.md) and [Datasets](datasets.md). The verified historical simulator
and decoder revisions are recorded in [Release provenance](reproducibility.md).

Evaluate VLA models through a shared **Unitree G1 + SONIC** simulation protocol.
The model runs as a service on **one port, 8000 by default**; the evaluator
connects to that service and manages tasks, simulation, timeouts, and reports.
The model and simulator can run in separate Conda environments or on separate
machines, following the model-server → simulation-client workflow used by the
RoboCasa Openpi evaluation scripts.

The default benchmark is **50 tasks × 30 episodes = 1,500 episodes**:

| Task group | Tasks | Episodes | Simulation-time limit per episode |
| --- | ---: | ---: | ---: |
| Seen atomic | 25 | 750 | 40 seconds |
| Seen composite | 15 | 450 | 180 seconds |
| Unseen composite | 10 | 300 | 180 seconds |

The task split comes from
[`robocasa_eval/tasks50.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa_eval/tasks50.json). All models
use the same observations and produce SONIC motion tokens plus G1 hand joint
targets. Model adapters handle preprocessing, normalization, temporal history,
and conversion to this action representation.

## Prepare the two environments

Run the following commands from the repository root. The public entry point is
`python -m robocasa_eval`; `robocasa-eval`, `scripts/evaluate_g1.sh`, and
`robocasa/scripts/evaluate.py` expose the same commands. The former
`robocasa/utils/eval_utils.py` command also dispatches to this CLI. Its
`create_eval_env` and `run_random_rollouts` helpers remain available.

The **model environment** needs this checkout, NumPy, and the dependencies of
the selected model adapter. For GROOT, install the compatible GR00T SDK and
PyTorch here. The server loads the checkpoint and receives observations through
HTTP; it does not load the simulator or SONIC decoder.

For the GROOT N1.7 SDK, prepare the complete offline Cosmos backbone snapshot
requested by the checkpoint's `processor_config.json`, including its pretrained
weights, configuration, and tokenizer, in the model workspace's Hugging Face
cache. This SDK initializes the backbone before restoring the fine-tuned
checkpoint, so a tokenizer-only cache is insufficient. The adapter uses offline
mode and does not download missing files during service startup.

The adapter creates a small inference view under `workspace/cache/groot_inference/`:
copies of the model and processor configurations refer to that local snapshot,
while weights and normalization files are linked for reading. The original
checkpoint stays unchanged. Each view retains `inference_view.json` with the
source configuration hashes and resolved snapshot, avoiding tokenizer Hub
metadata requests during offline loading.

The **simulation environment** needs this checkout, RoboSuite, MuJoCo, ONNX
Runtime, SONIC, its SDK/reset dependencies, the G1 assets, and the chosen renderer.
Install the evaluation extra with `pip install -e '.[eval]'`. This environment
uses the HTTP client and does not need PyTorch or the GR00T SDK. Set `sim_python`
to its Python interpreter and `decoder_path` to the compatible SONIC
`model_decoder.onnx`.

With `sim_root` set, simulation workers use the pinned runtime checkouts and
source manifest under that directory. With `sim_root: null`, they use packages
installed in `sim_python`'s environment. The runtime provides `SonicG1` and the
benchmark's registered tasks. `renderer: "osmesa"` uses CPU rendering; `egl`
is also available with a compatible rendering runtime.

Planning needs only Python's standard library and creates no output files:

```sh
python -m robocasa_eval plan --config examples/evaluation/http.json
```

## Configure the model server and evaluator

| Example | Purpose |
| --- | --- |
| [`server_groot.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/server_groot.json) | Serve a GROOT N1.7 Flow checkpoint trained for G1 SONIC |
| [`http.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/http.json) | Connect the evaluator to the model server on port 8000 |
| [`server_custom.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/server_custom.json) | Run the custom adapter interface template as a service |
| [`groot.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/groot.json) | Run GROOT directly inside evaluation workers |
| [`custom.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/custom.json) | Run a custom Python adapter directly |

Create separate working configurations for the model service and evaluator:

```sh
mkdir -p evaluation_runs/configs
cp examples/evaluation/server_groot.json evaluation_runs/configs/server.json
cp examples/evaluation/http.json evaluation_runs/configs/evaluation.json
```

In `server.json`, set the checkpoint and GR00T repository paths and a descriptive,
immutable `policy.id`. In `evaluation.json`, set the simulation interpreter and
SONIC decoder paths, and use the **same `policy.id`** as the server. Its
`policy.kwargs.endpoint` points to `http://127.0.0.1:8000` for a local service.
On separate machines, start the server with `--host 0.0.0.0` and set the
endpoint to `http://MODEL_SERVER_IP:8000`.

Paths in these configurations are resolved from the invoking working directory.
Custom factory kwargs pass through unchanged; use absolute paths for custom
model files, since evaluation workers run inside their output directory. List
extra model files or directories in `policy.artifacts` to fingerprint their
contents.

GROOT checkpoint files, resolved processor/tokenizer configuration, configured
SDK source, and key inference dependency versions are fingerprinted by the
model server. The evaluator fetches its metadata, verifies `policy.id`, and
binds the run to the reported fingerprint. A changed serving model requires a
new smoke test and run identity.

Simulator fingerprints cover imported source/configuration files, lightweight
robot and scene descriptions, and recorded package versions. Large mesh and
texture versions remain part of the prepared simulation runtime.

### Model service settings

| Field | Meaning |
| --- | --- |
| `workspace` | Writable output/cache directory; default `./evaluation_runs` |
| `policy.factory` | `groot` or an importable `module:function` |
| `policy.id`, `policy.kwargs`, `policy.artifacts` | Model revision, constructor options, and additional files to fingerprint |
| `host`, `port` | Listening address; defaults `127.0.0.1`, `8000` |
| `devices` | Model replica devices, for example `["cuda:0", "cuda:1"]` |
| `max_batch_size`, `batch_wait_ms` | Per-replica dynamic batching limit and collection window; defaults 32 and 5 ms |
| `request_timeout_seconds` | Model operation timeout; default 120 seconds |
| `max_active_episodes`, `idle_session_seconds` | Session capacity and idle expiration timeout; defaults 4096 and 600 seconds |
| `seed` | Initial model RNG seed |

The service keeps one model replica per configured device. Each evaluation
worker/rank is assigned to a replica, and its episodes keep that replica for
their full lifetime so history and recurrent state stay together. Each replica
batches ready observations from its concurrent environments. All replicas share
the **same public port**.

### Evaluator settings

| Field | Meaning |
| --- | --- |
| `workspace`, `run_name` | Output/cache directory and name for this evaluation |
| `policy.factory` | `http` for the service-based workflow |
| `policy.id` | Model ID expected from the service |
| `policy.kwargs.endpoint`, `policy.kwargs.timeout` | Service URL and request timeout |
| `num_workers` | HTTP evaluation worker count; model GPU devices are configured on the server |
| `envs_per_worker` | Concurrent simulations per evaluation worker |
| `device_prefix` | `cpu` for HTTP clients, which do not allocate model GPUs |
| `episodes_per_task` | Default 30 |
| `atomic_seconds`, `composite_seconds` | Default 40 and 180 simulation seconds |
| `execution_horizon` | Maximum action steps executed before requesting new inference |
| `sim_python`, `sim_root`, `decoder_path` | Simulation interpreter, optional pinned runtime, and SONIC decoder |
| `renderer`, `camera_names` | Rendering backend and observation cameras |
| `task_manifest` | Optional task manifest; `null` uses the bundled benchmark |

Logs, reports, worker directories, temporary files, and caches use `workspace`.
`disk_limit_bytes` controls evaluation workspace capacity and defaults to
2,000,000,000,000 bytes. Preflight and periodic checks enforce this capacity;
existing data is preserved. Custom adapters keep generated files and caches
inside the workspace supplied to their factory.

## Start the service and evaluate

**Terminal 1, model environment:** start one model replica on one GPU.

```sh
python -m robocasa_eval serve --config evaluation_runs/configs/server.json \
  --host 127.0.0.1 --port 8000 --devices cuda:0
```

Keep the service running while the simulation client performs its checks and
evaluation. Model-service stdout and stderr are mirrored to the terminal and
`workspace/model_service/<fingerprint>/server.log`; the same directory contains
the resolved `config.json` and `metadata.json`. The service exposes its model
identity and replica information at `http://127.0.0.1:8000/metadata`.
`GET /health` returns HTTP 200 with `ready: true` while the model pool is ready,
or HTTP 503 with `ready: false` when it is unavailable.

**Terminal 2, simulation environment:** plan, check, and exercise the model
connection before running the benchmark.

```sh
# Resolve the schedule without loading a model or simulator.
python -m robocasa_eval plan --config evaluation_runs/configs/evaluation.json

# Check the running service identity, simulation imports, decoder, and rendering.
python -m robocasa_eval check --config evaluation_runs/configs/evaluation.json

# Short integration probe.
python -m robocasa_eval smoke --config evaluation_runs/configs/evaluation.json \
  --envs 4 --steps 100

# Full-horizon integration test, with the one-GPU service still running.
python -m robocasa_eval smoke --config evaluation_runs/configs/evaluation.json \
  --envs 4 --full-episodes \
  --tasks CloseDishwasher OpenDrawer CoolBakedCake CupcakeCleanup

# Launch evaluation workers after the full-horizon test passes.
python -m robocasa_eval run --config evaluation_runs/configs/evaluation.json

# Regenerate reports from saved episode results.
python -m robocasa_eval summarize --config evaluation_runs/configs/evaluation.json
```

The full-horizon smoke covers atomic and composite tasks using complete 40- and
180-second simulation budgets with multiple simultaneous environments. Success
and task timeout are both completed smoke outcomes; integration errors must be
resolved before a formal run. A short probe does not create the full-horizon
smoke receipt.

Changing the model, model device family (such as CPU to CUDA), adapter,
simulator runtime, decoder, cameras, task manifest, or scoring protocol requires
a new full-horizon smoke. Service replica count
and evaluation worker count can change after the single-GPU test while the
model and evaluation protocol stay the same.

### Serve 16 GPUs through one port

After the single-GPU test, restart the service with all 16 devices. Bash expands
`cuda:{0..15}` into the individual device arguments:

```sh
# Model environment: 16 model replicas, one listening port.
python -m robocasa_eval serve --config evaluation_runs/configs/server.json \
  --host 127.0.0.1 --port 8000 --devices cuda:{0..15}
```

In the simulation environment, the example's `num_workers: 16` and
`envs_per_worker: 4` create up to 64 simultaneous simulations. Every worker connects
to the same endpoint, with the 16 worker ranks assigned across the 16 model
replicas.

```sh
bash scripts/evaluate_g1.sh run --config evaluation_runs/configs/evaluation.json \
  --num-workers 16 --envs-per-worker 4
```

For a smaller evaluation client, add `--num-workers 1` and choose a separate
`--run-name`. This changes client concurrency; it does not change server devices. Use at least
as many client workers as model replicas to exercise every serving GPU.
For simulation workers across multiple nodes, use a shared evaluation workspace
and the compatible `NNODES`, `GPUS_PER_NODE`, and `NODE_RANK` environment settings.
All clients use the same model-service address. The server's `devices` refer to
GPUs on its own host.

To resume an interrupted run, keep its configuration and `run_name`. Completed
records with matching identity are reused. Increasing concurrency needs enough
CPU, memory, network bandwidth, and model serving capacity.

### Direct model adapters

For a model and evaluator in the same environment, `groot.json` and `custom.json`
retain the direct-adapter workflow. Each evaluation worker loads its own model
on its selected GPU. The existing `total_gpus` and `envs_per_gpu` settings remain
available for direct mode. Apply the same `check`, full-horizon `smoke`, `run`, and
`summarize` commands to that configuration. The shared task and action protocol
is identical to the HTTP workflow.

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
[example adapter](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa_eval/examples/custom_policy.py) illustrates
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

### HTTP communication protocol

The built-in `serve` command implements **`robocasa-g1-v1`**. A different model
framework can reuse this server by providing a Python factory, or implement the
same HTTP endpoints. This follows the separate model-server / simulator-client
workflow; its wire format is JSON HTTP rather than Openpi's native WebSocket
format.

Set evaluator `policy.factory` to `"http"`. Its kwargs are `endpoint`, optional
`timeout` (seconds), and optional `auth_token_env`, which names an environment
variable containing a bearer token for an authenticated service. Credentials
remain outside configuration files.

Every POST body and JSON response uses `"protocol": "robocasa-g1-v1"`. The service
provides these routes on the configured single port:

| Method and endpoint | Request fields besides `protocol` | Response |
| --- | --- | --- |
| `GET /metadata` | None | `model_id`, `fingerprint`, `action_dims`, and `replicas` |
| `GET /health` | None | Metadata plus `ready`; HTTP 200 when ready, HTTP 503 otherwise |
| `POST /reset` | `context` | `ok: true` |
| `POST /predict` | `observations`, `contexts` | `actions`, in matching input order |
| `POST /end_episode` | `context` | `ok: true` |

Every JSON response also contains the serving model's `fingerprint`. The
evaluator checks `/metadata` during preflight and sends
`X-Robocasa-Model-Fingerprint` with each inference/lifecycle request; the service
rejects a different fingerprint, and the client verifies the response identity.
Contexts contain the six `EpisodeContext` fields. Arrays use `__ndarray__: true`, a NumPy `dtype` string, a `shape` list,
and base64-encoded C-order bytes in `data`. For example, motion tokens use
`"<f4"` and `[T, 64]`; RGB images use `"|u1"` and `[H, W, 3]`. Numeric JSON lists
are also accepted for action arrays. Actions contain finite, denormalized values.

Use `encode_value` and `decode_value` from
[`robocasa_eval.adapters.http`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa_eval/adapters/http.py) for Python
protocol implementations. The built-in service maintains per-episode history,
routes each episode to its assigned replica, and dynamically batches inference
across concurrent clients. `server_custom.json` demonstrates this interface
using the example adapter's placeholder outputs; replace it with a trained
model adapter for benchmark results.

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
