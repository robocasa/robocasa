import subprocess
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("models_folder")
parser.add_argument("--category_name", type=str)
parser.add_argument("--base_asset_path", type=str)
parser.add_argument("--no_cached_coll", action="store_true")
parser.add_argument("--max_processes", type=int, default=10)
parser.add_argument(
    "--import_script_path",
    type=str,
    default="/Users/lancezhang/projects/kitchen/robosuite-model-zoo"
    "-dev/robosuite_model_zoo/scripts/objaverse/import_objaverse.py",
)
parser.add_argument("--default_scale", type=float, default=0.2)
args = parser.parse_args()

if args.category_name is None:
    args.category_name = args.models_folder.split("/")[-2]
    print("Setting category name as:", args.category_name)
args.models_folder = os.path.abspath(args.models_folder)


processes = set()
commands = list()

# generate commands
for i, model in enumerate(os.listdir(args.models_folder)):
    if model.startswith("."):
        continue
    command = (
        "python {script_path} --model_folder {model_folder} --model_name {model_name} "
        + "--scale {scale} --path {path} --base_asset_path {base_asset_path}"
    )
    command = command.format(
        script_path=args.import_script_path,
        model_folder=args.category_name,
        model_name=args.category_name + "_" + str(i),
        scale=args.default_scale,
        path=os.path.join(args.models_folder, model),
        base_asset_path=args.base_asset_path,
    )
    if args.no_cached_coll:
        command += " --no_cached_coll"
    commands.append(command)

# call commands
for i, cmd in enumerate(commands):
    print("Adding command #" + str(i))
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= args.max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
