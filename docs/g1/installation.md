# G1 installation

Prepare a **Linux x86-64 simulation environment** for Unitree G1 evaluation.
The commands below use Bash, Conda, Python 3.12, and the recorded simulation
dependency versions. They cover the evaluator, SonicG1 robot, kitchen assets,
SONIC decoder, and a connection to an existing inference service.

The default renderer is CPU OSMesa and the decoder uses CPU ONNX Runtime.
The inference service runs in its own environment and exposes the
[shared HTTP protocol](evaluation.md#http-communication-protocol). Its model
architecture, training packages, and accelerator setup are independent of this
installation. For interactive teleoperation and exporter setup, continue with
[Data collection](data_collection.md) after preparing the simulator.

## 1. Create the workspace and environment

On Ubuntu or Debian, install the system compiler and rendering libraries:

```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential libosmesa6 libegl1 libgl1 libglib2.0-0
```

Choose a new writable workspace. Keep the exported variables in the same shell
throughout this guide. Source checkouts, the environment, downloads, and outputs
will use this directory.

```bash
mkdir -p ./robocasa-g1-workspace
export G1_WORKSPACE="$(cd ./robocasa-g1-workspace && pwd)"
export G1_RUNTIME="$G1_WORKSPACE/runtime"
export ROBOCASA_ROOT="$G1_WORKSPACE/robocasa"
mkdir -p "$G1_RUNTIME" "$G1_WORKSPACE/configs" "$G1_WORKSPACE/logs" \
  "$G1_WORKSPACE/tmp" "$G1_WORKSPACE/cache"
export HF_HOME="$G1_WORKSPACE/cache/huggingface"
export PIP_CACHE_DIR="$G1_WORKSPACE/cache/pip"
export XDG_CACHE_HOME="$G1_WORKSPACE/cache/xdg"
export TMPDIR="$G1_WORKSPACE/tmp"
export PYTHONDONTWRITEBYTECODE=1

conda create -y --prefix "$G1_WORKSPACE/envs/sim" python=3.12 pip
conda activate "$G1_WORKSPACE/envs/sim"
```

## 2. Check out the compatible sources

Clone this repository's G1 release branch. Use its current evaluator alongside
the fixed companion dependencies below. Record the resolved RoboCasa commit
with each installation.

```bash
git clone --branch agent/sonic-g1-followups \
  https://github.com/Xucxname/robocasa.git "$ROBOCASA_ROOT"
git -C "$ROBOCASA_ROOT" rev-parse HEAD > "$G1_WORKSPACE/logs/robocasa-commit.txt"

git clone --filter=blob:none --no-checkout \
  https://github.com/Xucxname/robosuite.git "$G1_RUNTIME/robosuite"
git -C "$G1_RUNTIME/robosuite" checkout --detach 67c18a4c66fa7bca02258a8a6aee2abf29e5427f

git clone --filter=blob:none --no-checkout \
  https://github.com/ARISE-Initiative/robosuite_models.git "$G1_RUNTIME/robosuite_models"
git -C "$G1_RUNTIME/robosuite_models" checkout --detach 2d63a6b4319bda772420daf10594ce0de5bd793d

git clone --filter=blob:none --no-checkout \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git "$G1_RUNTIME/sonic"
git -C "$G1_RUNTIME/sonic" checkout --detach a9d20b2ac0949244d94461a1a3263f38c5027c4a
```

The robosuite fork contains the SonicG1/Dex3 definitions, controller, and robot
meshes. The SONIC checkout supplies simulation reset helpers and its vendored
Unitree SDK. The companion versions are recorded in
[Release provenance](reproducibility.md) and
[`runtime_versions.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/runtime_versions.json).

## 3. Install the SDK runtime and Python dependencies

Build CycloneDDS 0.10.2, required by the vendored Unitree SDK. Its installation
prefix stays inside the workspace. Keep `CYCLONEDDS_HOME` exported when using
this environment, including in a new terminal.

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/eclipse-cyclonedds/cyclonedds.git "$G1_RUNTIME/cyclonedds/source"
git -C "$G1_RUNTIME/cyclonedds/source" checkout --detach 9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
export CYCLONEDDS_HOME="$G1_RUNTIME/cyclonedds/install"

cmake -S "$G1_RUNTIME/cyclonedds/source" -B "$G1_RUNTIME/cyclonedds/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
  -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_DDSPERF=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF -DENABLE_SSL=NO -DENABLE_SECURITY=NO \
  -DBUILD_SHARED_LIBS=ON -DBUILD_IDLC=ON -DENABLE_LTO=OFF
cmake --build "$G1_RUNTIME/cyclonedds/build" --parallel 4
cmake --install "$G1_RUNTIME/cyclonedds/build"

python -m pip install setuptools==80.9.0 wheel==0.45.1
python -m pip install -r "$ROBOCASA_ROOT/examples/evaluation/requirements-sim.txt"
python -m pip install --no-deps --no-build-isolation \
  -e "$G1_RUNTIME/robosuite" \
  -e "$G1_RUNTIME/robosuite_models" \
  -e "$ROBOCASA_ROOT" \
  -e "$G1_RUNTIME/sonic/external_dependencies/unitree_sdk2_python"

python - <<'PY'
import os
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])
sonic = Path(os.environ["G1_RUNTIME"]) / "sonic"
(site / "sonic-eval-source.pth").write_text(str(sonic) + "\n")
PY
```

The [simulation requirements](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/examples/evaluation/requirements-sim.txt)
install the runtime packages explicitly; editable installs then register the
source checkouts. This separates simulation dependencies from the repositories'
broader training and collection requirements. The complete SONIC package pins
NumPy 1.26.4, whereas this simulation runtime uses 2.2.5. The simulation uses
SONIC's source helpers through the environment's `.pth` file. Install collection
and model packages in their respective environments.

Repository package metadata still lists those broader dependencies, so
`pip check` may report intentionally absent training packages or the
`opencv-python` distribution when this profile uses `opencv-python-headless`.
Use the simulator and integration checks below to validate this profile.

## 4. Prepare kitchen assets

Robot meshes come from the companion checkouts. Kitchen textures, fixtures, and
objects are separate downloads. The following fixed-revision downloads contain
approximately **11 GB of ZIP archives**; allow space for both cached archives
and their extracted contents.

Use the official [RoboCasa asset release](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166)
and [NVIDIA kitchen asset release](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/tree/420a04af939c34873e6839a586b70844baf28aab).
The script extracts each archive into the package's expected asset location.
All six groups are needed; scene styles can reference `generative_textures`
even when runtime texture randomization is disabled.

```bash
python - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import HfApi, hf_hub_download

assets = Path(os.environ["ROBOCASA_ROOT"]) / "robocasa/models/assets"
rc_repo = "robocasa/robocasa-assets"
rc_rev = "1b92c3d02ca4354984fec961357db0bff7b32166"
nv_repo = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
nv_rev = "420a04af939c34873e6839a586b70844baf28aab"
jobs = [
    (rc_repo, rc_rev, "textures.zip", assets),
    (rc_repo, rc_rev, "generative_textures.zip", assets),
    (rc_repo, rc_rev, "objaverse.zip", assets / "objects"),
    (rc_repo, rc_rev, "aigen_objs.zip", assets / "objects"),
]
files = HfApi().list_repo_files(nv_repo, repo_type="dataset", revision=nv_rev)
for prefix, target, expected_count in [
    ("fixtures_lightwheel/", assets / "fixtures", 17),
    ("objects_lightwheel/", assets / "objects/lightwheel", 69),
]:
    archives = sorted(f for f in files if f.startswith(prefix) and f.endswith(".zip"))
    if len(archives) != expected_count:
        raise RuntimeError(f"Unexpected archive set: {prefix}: {len(archives)}")
    jobs.extend((nv_repo, nv_rev, f, target) for f in archives)

for repo, revision, filename, target in jobs:
    archive = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        for member in zf.infolist():
            if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Invalid archive path: {member.filename}")
        zf.extractall(target)
    print(f"Extracted {filename} -> {target}", flush=True)
PY
```

The resulting package tree includes `assets/textures`, `assets/generative_textures`,
`assets/fixtures`, and `assets/objects/{objaverse,aigen_objs,lightwheel}`. Keep the
Git-tracked `assets/scenes` and `assets/fixtures/fixture_registry` content in place.
The source asset downloader also supports Box downloads; this guide uses the
fixed Hugging Face revisions for repeatable archive selection.

## 5. Download the SONIC decoder

Download only `model_decoder.onnx` from the fixed
[SONIC release](https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f).
The evaluator loads this model through CPU ONNX Runtime; it requires no
TensorRT engine or SONIC deployment process.

```bash
python - <<'PY'
import hashlib
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

decoder = Path(hf_hub_download(
    repo_id="nvidia/GEAR-SONIC",
    filename="model_decoder.onnx",
    revision="6733128a3d8a523b1418b06bca3cdf61c8b0987f",
    local_dir=Path(os.environ["G1_WORKSPACE"]) / "models/sonic",
))
with decoder.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if decoder.stat().st_size != 40900688 or digest != (
    "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
):
    raise ValueError("SONIC decoder does not match the recorded release")
print(f"Verified {decoder}")
PY
```

## 6. Configure and check the simulator

Set the renderer before importing MuJoCo or RoboSuite. OSMesa uses the system
libraries installed in step 1 and works without a display.

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
cd "$ROBOCASA_ROOT"
python -m robocasa.scripts.setup_macros
```

`setup_macros` creates the local macro configuration; assets remain under the
installed RoboCasa source tree. The evaluator's OSMesa subprocesses default to
`NUMBA_OPT=0`, keeping JIT compilation enabled while avoiding an LLVM optimizer
crash reproduced during scene placement in a clean Ubuntu 24.04 installation.
An explicit `NUMBA_OPT` setting takes precedence. The diagnostic records the
effective compiler settings and package versions in the runtime identity;
changing them requires a new smoke test. Compiler caches are separated by the
worker's Numba settings so diagnostics do not reuse code built with different
optimization settings.

Configure a single evaluator worker with two
concurrent environments for the first integration test:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["G1_WORKSPACE"])
source = Path(os.environ["ROBOCASA_ROOT"])
config = json.loads((source / "examples/evaluation/http.json").read_text())
config.update(
    workspace=str(root / "outputs"),
    run_name="g1-installation-check",
    num_workers=1,
    envs_per_worker=2,
    sim_python=sys.executable,
    sim_root=None,
    decoder_path=str(root / "models/sonic/model_decoder.onnx"),
)
(root / "configs/evaluation.json").write_text(json.dumps(config, indent=2) + "\n")
PY

python -m robocasa_eval plan --config "$G1_WORKSPACE/configs/evaluation.json" \
  > "$G1_WORKSPACE/logs/plan.json"
```

The plan should contain 50 tasks and 1,500 episodes. To check the simulator
without a running inference service, pass the resolved plan configuration to
its diagnostic entry point:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
from robocasa_eval.evaluation import prepare_workspace, sim_environment

root = Path(os.environ["G1_WORKSPACE"])
config = json.loads((root / "logs/plan.json").read_text())["config"]
prepare_workspace(config)
with (root / "logs/simulator-check.log").open("w") as log:
    subprocess.run(
        [sys.executable, "-B", "-m", "robocasa_eval.sim_worker", "--check"],
        input=json.dumps(config), text=True, check=True,
        cwd=root / "tmp", env=sim_environment(config),
        stdout=log, stderr=subprocess.STDOUT,
    )
print((root / "logs/simulator-check.log").read_text())
PY
```

The final JSON should report `tasks_registered: 50`, `render_probe: "passed"`,
`geometry_probe: "passed"`, and `sonic_reset_dependency: "passed"`. This checks
decoder loading, task registration, SDK imports, a small render, and the
matrix-to-quaternion computation used during scene placement. The next step exercises real
kitchen scenes and the connected policy.

## 7. Connect inference and run the first episodes

Keep a compatible inference service running. Set its URL in the configuration
so the same endpoint is used for checks, smoke tests, and the eventual run:

```bash
export G1_INFERENCE_ENDPOINT="http://127.0.0.1:8000"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["G1_WORKSPACE"]) / "configs/evaluation.json"
config = json.loads(path.read_text())
config["policy"]["kwargs"]["endpoint"] = os.environ["G1_INFERENCE_ENDPOINT"]
path.write_text(json.dumps(config, indent=2) + "\n")
PY

python -m robocasa_eval check --config "$G1_WORKSPACE/configs/evaluation.json"
python -m robocasa_eval smoke --config "$G1_WORKSPACE/configs/evaluation.json" \
  --envs 2 --steps 100
python -m robocasa_eval smoke --config "$G1_WORKSPACE/configs/evaluation.json" \
  --envs 2 --full-episodes \
  --tasks CloseDishwasher OpenDrawer CoolBakedCake CupcakeCleanup
python -m pip freeze > "$G1_WORKSPACE/logs/simulation-packages.txt"
```

The full-horizon smoke covers atomic and composite tasks with 40- and
180-second simulation limits. Success or task timeout are completed episodes;
resolve integration errors before launching the benchmark. A passed test
creates the receipt required by `run`. Reports and logs use the configured
workspace under `outputs/`. Follow [Evaluation](evaluation.md) for full runs,
configurable parallel environments, model serving, and resume settings.

## Rendering and common setup errors

| Symptom | Check |
| --- | --- |
| `SonicG1` or `SONIC_WBC` is unavailable | Check the robosuite checkout and installed editable path against step 2. |
| `gear_sonic` or `unitree_sdk2py` cannot import | Check the SONIC `.pth`, vendored SDK editable install, and active Python environment. |
| CycloneDDS cannot be located | Export the absolute `CYCLONEDDS_HOME` from step 3 before installing the Python binding or running evaluation. |
| OSMesa or OpenGL cannot create a context | Install the system `libosmesa6` library; set renderer variables before Python starts. The evaluator clears inherited accelerator library overrides for its CPU simulator. |
| Simulator exits during scene placement with an LLVM optimization fault | Check the worker's `simulator.log`. Remove an inherited `NUMBA_OPT` override to use the OSMesa default of `0`, then rerun the diagnostic in a new process. |
| A kitchen mesh or texture is missing | Complete all six asset groups and check the extraction roots in step 4. |
| Decoder input/output shapes do not match | Use the exact decoder revision and SHA-256 from step 5. |
| HTTP connection fails or metadata is rejected | Check the service URL and the shared protocol in the evaluation guide. |

For GPU rendering, install a working NVIDIA driver and EGL runtime, change
`renderer` in the evaluator configuration to `"egl"`, and start a new Python process
with `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `LIBGL_ALWAYS_SOFTWARE` unset.
Keep the renderer GPU visible and rerun `check` and full-horizon `smoke`.
Model inference devices remain configured on the inference server.
