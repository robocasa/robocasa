"""
[Mutli-processing version] Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

    num_procs (int): number of parallel processes to use for extraction. Default is 1.

    gpu_ids (int or [int]): GPU IDs to use for processes. Processes will be distributed 
        across these GPUs in round-robin fashion.

    procs_per_gpu (int or [int]): Number of processes to allocate to each GPU. Must have 
        same length as gpu_ids and sum must equal num_procs.

Example usage:
    
    # extract low-dimensional observations with 4 processes
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2 --num_procs 4
    
    # extract 84x84 image observations
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # extract with 8 processes distributed across 4 GPUs
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --num_procs 8 --gpu_ids 0 1 2 3

    # extract with 4 processes, each using a specific GPU
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --num_procs 4 --gpu_ids 0 1 2 3

    # extract with custom GPU allocation: 3 processes on GPU 0, 2 on GPU 1, 2 on GPU 2, 1 on GPU 3
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --num_procs 8 --gpu_ids 0 1 2 3 --procs_per_gpu 3 2 2 1

    # extract with 6 processes: 4 on GPU 0, 2 on GPU 1
    python dataset_states_to_obs_mp.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --num_procs 6 --gpu_ids 0 1 --procs_per_gpu 4 2
"""
import time
import socket
import traceback
import os
import json
import h5py
import psutil
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from queue import Empty
import shutil  # For getting terminal size

import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils


try:
    import mimicgen
except ImportError:
    print("WARNING: could not import mimicgen envs")

MAX_RETRIES = 3


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        # actions_abs=[],
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(traj_len):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        # traj["actions_abs"].append(action_abs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def process_demo_batch(
    process_id, args, env_meta, work_queue, result_queue, progress_queue, gpu_id=None
):
    """
    Process demonstrations from a work queue until the queue is empty.

    Args:
        process_id (int): ID of this worker process
        args: Script arguments
        env_meta: Environment metadata
        work_queue (mp.Queue): Queue containing work items (demos to process)
        result_queue (mp.Queue): Queue to store results
        progress_queue (mp.Queue): Queue to report progress
        gpu_id (int, optional): GPU ID to use for this process
    """
    # Set GPU environment variables if gpu_id is specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
        print(f"Process {process_id} using GPU {gpu_id}")

    # robocasa-specific features
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True

    # Create environment for this process
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    # print("==== Using environment with the following metadata ====")
    # print(json.dumps(env.serialize(), indent=4))
    # print("")

    # Open input file in read mode
    f = h5py.File(args.dataset, "r")

    # Create temporary output file for this process
    temp_output = f"{args.dataset}_temp_{process_id}.hdf5"
    f_out = h5py.File(temp_output, "w")
    data_grp = f_out.create_group("data")

    total_samples = 0
    processed_demos = []
    success = True
    retries = 0

    # Process demos until the queue is empty
    while True:
        try:
            # Get next demo from queue with timeout
            if success or retries > MAX_RETRIES:
                ep = work_queue.get(timeout=1)
                success = False
                retries = 0
            else:
                retries += 1

            if ep is None:  # Poison pill
                break

            # prepare states to reload from

            states = f["data/{}/states".format(ep)][()]

            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]
            # actions_abs = f["data/{}/actions_abs".format(ep)][()]
            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                # actions_abs=actions_abs,
                done_mode=args.done_mode,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            # store transitions
            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            # ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.no_compress:
                    ep_data_grp.create_dataset(
                        "obs/{}".format(k), data=np.array(traj["obs"][k])
                    )

                else:
                    ep_data_grp.create_dataset(
                        "obs/{}".format(k),
                        data=np.array(traj["obs"][k]),
                        compression="gzip",
                    )
                if args.include_next_obs:
                    if args.no_compress:
                        ep_data_grp.create_dataset(
                            "next_obs/{}".format(k), data=np.array(traj["next_obs"][k])
                        )
                    else:
                        ep_data_grp.create_dataset(
                            "next_obs/{}".format(k),
                            data=np.array(traj["next_obs"][k]),
                            compression="gzip",
                        )

            # episode metadata
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]

            total_samples += traj["actions"].shape[0]
            processed_demos.append(ep)
            # Report progress
            progress_queue.put((1, traj["actions"].shape[0]))
            success = True

        except Exception as e:
            print("_" * 50)
            print("Error processing demo index {}: {}".format(ep, e))
            print(traceback.format_exc())
            print("_" * 50)
            # Queue is empty, check if we should continue waiting
            if work_queue.empty():
                break
            del env
            env = EnvUtils.create_env_for_data_processing(  # when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )

    # Store chunk metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)

    f.close()
    f_out.close()

    # Put result in result queue
    result_queue.put((temp_output, total_samples, processed_demos))


def get_gpu_allocation(num_procs, gpu_ids, procs_per_gpu=None):
    """
    Determine GPU allocation for processes.

    Args:
        num_procs (int): Total number of processes
        gpu_ids (list): List of GPU IDs to use
        procs_per_gpu (list, optional): Number of processes to allocate to each GPU

    Returns:
        list: GPU IDs assigned to each process, or None if no GPU allocation
    """
    if not gpu_ids:
        return None

    if procs_per_gpu is not None:
        if len(procs_per_gpu) != len(gpu_ids):
            raise ValueError(
                f"--procs_per_gpu must have same length as --gpu_ids. "
                f"Got {len(procs_per_gpu)} values for {len(gpu_ids)} GPUs."
            )

        total_allocated_procs = sum(procs_per_gpu)
        if total_allocated_procs != num_procs:
            # Adjust procs_per_gpu to match num_procs
            if total_allocated_procs > num_procs:
                # Need to reduce processes
                excess = total_allocated_procs - num_procs
                # Sort GPUs by number of processes (descending) to reduce from most loaded GPUs first
                gpu_loads = sorted(
                    enumerate(procs_per_gpu), key=lambda x: x[1], reverse=True
                )

                for i in range(excess):
                    # Reduce processes from most loaded GPU
                    gpu_idx = gpu_loads[i % len(gpu_loads)][0]
                    procs_per_gpu[gpu_idx] -= 1

                print(f"Adjusted GPU allocation to match {num_procs} processes:")
                for gpu_id, num_procs_for_gpu in zip(gpu_ids, procs_per_gpu):
                    print(f"  GPU {gpu_id}: {num_procs_for_gpu} processes")
            else:
                raise ValueError(
                    f"Sum of --procs_per_gpu ({total_allocated_procs}) must equal --num_procs ({num_procs})"
                )

        # Create allocation list based on procs_per_gpu
        gpu_allocation = []
        for gpu_id, num_procs_for_gpu in zip(gpu_ids, procs_per_gpu):
            gpu_allocation.extend([gpu_id] * num_procs_for_gpu)
    else:
        # Use round-robin allocation
        gpu_allocation = [gpu_ids[i % len(gpu_ids)] for i in range(num_procs)]
        print(f"Using round-robin GPU allocation across {len(gpu_ids)} GPUs: {gpu_ids}")

    return gpu_allocation


def dataset_states_to_obs_mp(args):

    # Get environment metadata
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)

    # Read demonstrations and sort them
    with h5py.File(args.dataset, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

    # Store original demo names for merging
    original_demos = demos.copy()
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    # Maybe reduce number of demonstrations
    if args.n is not None:
        demos = demos[: args.n]
        original_demos = original_demos[: args.n]

    if len(demos) == 0:
        raise ValueError("No demonstrations to process after applying start/n filters")

    # Cap number of processes to number of demos
    num_processes = min(args.num_procs, len(demos))
    if num_processes < args.num_procs:
        print(
            f"\nWarning: Reducing number of processes from {args.num_procs} to {num_processes} "
            f"to match number of demos"
        )

    # Initialize multiprocessing queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Fill work queue with demos
    for demo in demos:
        work_queue.put(demo)

    # Add poison pills to signal processes to terminate
    for _ in range(num_processes):
        work_queue.put(None)

    # Handle GPU allocation
    gpu_allocation = None
    if args.gpu_ids is not None:
        if len(args.gpu_ids) == 0:
            print(
                "Warning: --gpu_ids specified but no GPU IDs provided. Running without GPU allocation."
            )
        else:
            # Get GPU allocation for the actual number of processes
            gpu_allocation = get_gpu_allocation(
                num_processes, args.gpu_ids, args.procs_per_gpu
            )

    print(
        f"\nProcessing {len(demos)} demonstrations using {num_processes} processes..."
    )
    if args.n is not None:
        print(f"Processing {args.n} demos")

    if gpu_allocation is not None:
        if args.procs_per_gpu is not None:
            print(f"Custom GPU allocation specified")
        else:
            print(f"Round-robin GPU allocation: {args.gpu_ids}")
    else:
        print(f"Running without explicit GPU allocation")

    # Initialize multiprocessing with progress bars
    mp.freeze_support()  # For Windows support
    processes = []
    try:
        # Start worker processes
        for i in range(num_processes):
            # Determine GPU ID for this process if GPU allocation is specified
            gpu_id = None
            if gpu_allocation:
                gpu_id = gpu_allocation[i]

            p = mp.Process(
                target=process_demo_batch,
                args=(
                    i,
                    args,
                    env_meta,
                    work_queue,
                    result_queue,
                    progress_queue,
                    gpu_id,
                ),
            )
            p.start()
            processes.append(p)

        # Monitor progress with a single progress bar
        pbar = tqdm(total=len(demos), desc="Processing demos")
        total_samples = 0
        completed_demos = 0

        while completed_demos < len(demos):
            try:
                # Get progress update with timeout
                demos_done, samples = progress_queue.get(timeout=0.1)
                completed_demos += demos_done
                total_samples += samples
                pbar.update(demos_done)
                pbar.set_postfix({"total_samples": total_samples})
            except Empty:
                # Check if any process has terminated unexpectedly
                if not any(p.is_alive() for p in processes):
                    break
        pbar.close()

        # Wait for all processes to complete
        for p in processes:
            if p.is_alive():
                p.join()

        # Collect results
        results = []
        demo_locations = {}  # Maps demo name to temp file location
        while not result_queue.empty():
            temp_output, total_samples, processed_demos = result_queue.get()
            results.append((temp_output, total_samples))
            # Record which demos are in which temp files
            for demo in processed_demos:
                demo_locations[demo] = temp_output

    finally:
        # Ensure processes are terminated
        for p in processes:
            if p.is_alive():
                p.terminate()

    # Merge results in the original demo order
    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("\nMerging temporary files...")
    with h5py.File(output_path, "w") as f_out:
        data_grp = f_out.create_group("data")
        total_samples = 0

        # Copy data maintaining original demo order by following the original demos list
        pbar = tqdm(demos, desc="Merging")
        try:
            for demo in pbar:
                # Look up which temp file contains this demo
                if demo not in demo_locations:
                    print(
                        f"Warning: Demo {demo} not found in any temporary files. Skipping."
                    )
                    continue
                temp_file = demo_locations[demo]
                with h5py.File(temp_file, "r") as f_temp:
                    # Copy this demo's data to the output file
                    f_temp.copy(f"data/{demo}", data_grp)
                    total_samples += data_grp[demo].attrs["num_samples"]
                    # Copy environment args from the first temp file if not done yet
                    if "env_args" not in data_grp.attrs:
                        data_grp.attrs["env_args"] = f_temp["data"].attrs["env_args"]
                pbar.set_postfix({"total_samples": total_samples})
        finally:
            pbar.close()

        # Copy filter masks if they exist in the original file
        with h5py.File(args.dataset, "r") as f:
            if "mask" in f:
                f.copy("mask", f_out)

        data_grp.attrs["total"] = total_samples

    DatasetUtils.extract_action_dict(dataset=output_path)
    DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(
            output_path,
            num_demos=num_demos,
        )

    print("\nCleaning up temporary files...")
    for (
        temp_file,
        _,
    ) in results:
        try:
            os.remove(temp_file)
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")

    # Get memory usage
    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000)

    important_stats = dict(
        name=output_path,
        num_demos=len(demos),
        start_idx=0,
        mem_usage=f"{mem_usage} MB",
        num_processes=num_processes,
        gpu_allocation=gpu_allocation if gpu_allocation else "no GPU allocation",
    )
    return important_stats


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
    )

    # flag for using generative textures
    parser.add_argument(
        "--generative-textures",
        action="store_true",
        help="(optional) use generative textures for robosuite environments",
    )

    # flag for randomizing cameras
    parser.add_argument(
        "--randomize-cameras",
        action="store_true",
        help="(optional) randomize camera poses for robosuite environments",
    )

    # Add multiprocessing argument
    parser.add_argument(
        "--num_procs",
        type=int,
        default=1,
        help="number of parallel processes to use for extraction",
    )

    # Add GPU allocation argument
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="*",
        default=None,
        help="(optional) GPU IDs to use for processes. Processes will be distributed across these GPUs in round-robin fashion. Example: --gpu_ids 0 1 2 3",
    )

    # Add processes per GPU argument
    parser.add_argument(
        "--procs_per_gpu",
        type=int,
        nargs="*",
        default=None,
        help="(optional) Number of processes to allocate to each GPU. Must have same length as --gpu_ids. Example: --procs_per_gpu 3 2 2 1",
    )

    args = parser.parse_args()
    res_str = "finished run successfully!"
    important_stats = None
    try:
        t = time.time()
        important_stats = dataset_states_to_obs_mp(args)
        time_taken_hrs = (time.time() - t) / 3600.0
        important_stats["time_taken (hrs)"] = time_taken_hrs
        important_stats = json.dumps(important_stats, indent=4)
        print("\nExtraction Stats")
        print(important_stats)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)
