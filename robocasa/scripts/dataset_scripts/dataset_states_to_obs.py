"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback
import torch

import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

# from robomimic.utils.log_utils import log_warning


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


""" The process that writes over the generated files to memory """


def write_traj_to_file(
    args, output_path, total_samples, total_run, processes, mul_queue
):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0

    try:
        while (total_run.value < (processes)) or not mul_queue.empty():
            if not mul_queue.empty():
                num_processed = num_processed + 1
                item = mul_queue.get()
                ep = item[0]
                traj = item[1]
                process_num = item[2]
                try:
                    ep_data_grp = data_grp.create_group(ep)
                    ep_data_grp.create_dataset(
                        "actions", data=np.array(traj["actions"])
                    )
                    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                    ep_data_grp.create_dataset(
                        "rewards", data=np.array(traj["rewards"])
                    )
                    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                    # ep_data_grp.create_dataset(
                    #     "actions_abs", data=np.array(traj["actions_abs"])
                    # )
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
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                )
                            else:
                                ep_data_grp.create_dataset(
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                    compression="gzip",
                                )

                    if "datagen_info" in traj:
                        for k in traj["datagen_info"]:
                            ep_data_grp.create_dataset(
                                "datagen_info/{}".format(k),
                                data=np.array(traj["datagen_info"][k]),
                            )

                    # copy action dict (if applicable)
                    if "data/{}/action_dict".format(ep) in f:
                        action_dict = f["data/{}/action_dict".format(ep)]
                        for k in action_dict:
                            ep_data_grp.create_dataset(
                                "action_dict/{}".format(k),
                                data=np.array(action_dict[k][()]),
                            )

                    # episode metadata
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                        "model"
                    ]  # model xml for this episode
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                        "ep_meta"
                    ]  # ep meta data for this episode
                    # if "ep_meta" in f["data/{}".format(ep)].attrs:
                    #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
                    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                        0
                    ]  # number of transitions in this episode

                    total_samples.value += traj["actions"].shape[0]
                except Exception as e:
                    print("++" * 50)
                    print(
                        f"Error at Process {process_num} on episode {ep} with \n\n {e}"
                    )
                    print("++" * 50)
                    raise Exception("Write out to file has failed")
                print(
                    "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                        num_processed,
                        ep_data_grp.attrs["num_samples"],
                        ep,
                        process_num,
                        total_run.value,
                        (time.time() - start_time) / num_processed,
                    )
                )
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )
    print("total processes end {}".format(total_run.value))
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4
    )  # environment info
    print("Wrote {} total samples to {}".format(total_samples.value, output_path))

    f_out.close()
    f.close()

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

    print("Writing has finished")

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return


# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(
    process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue
):
    try:
        extract_multiple_trajectories_with_error(
            process_num, current_work_array, work_queue, lock, args2, mul_queue
        )
    except Exception as e:
        work_queue.put(current_work_array[process_num])
        print("*>*" * 50)
        print("Error process num {}:".format(process_num))
        print(e)
        print(traceback.format_exc())
        print("*>*" * 50)
        print()

    num_finished.value = num_finished.value + 1


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1


def extract_multiple_trajectories_with_error(
    process_num, current_work_array, work_queue, lock, args, mul_queue
):
    # create environment to use for data processing

    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    start_time = time.time()

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while (not work_queue.empty()) and (ind != -1):
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                add_datagen_info=args.add_datagen_info,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_grp = f["data/{}".format(ep)]

            states = ep_grp["states"][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = ep_grp.attrs["model_file"]
            initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
            mul_queue.put([ep, traj, process_num])

            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            print("_" * 50)
            print("Process {}:".format(process_num))
            print("Error processing demo index {}: {}".format(ind, e))
            print(traceback.format_exc())
            print("_" * 50)
            del env
            env = EnvUtils.create_env_for_data_processing(  # when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )

    f.close()
    print("Process {} finished".format(process_num))


def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
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

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    num_processes = args.num_procs

    index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value("i", 0)
    num_finished = multiprocessing.Value("i", 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array("i", num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                current_work_array,
                work_queue,
                lock,
                args,
                num_finished,
                mul_queue,
            ),
        )
        processes.append(process)

    process1 = multiprocessing.Process(
        target=write_traj_to_file,
        args=(
            args,
            output_path,
            total_samples_shared,
            num_finished,
            num_processes,
            mul_queue,
        ),
    )
    processes.append(process1)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return


if __name__ == "__main__":
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

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
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

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )

    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    args = parser.parse_args()
    dataset_states_to_obs_multiprocessing(args)
