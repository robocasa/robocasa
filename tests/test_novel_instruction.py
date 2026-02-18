import argparse
from termcolor import colored
import traceback

from robocasa.utils.dataset_registry import TARGET_TASKS
from robocasa.utils.env_utils import create_env, run_random_rollouts


def test_tasks_validity(
    env_names=None, env_seed=0, num_rollouts=10, num_steps=20, split="all"
):
    """
    Tests that all kitchen environment tasks run error free. Iterates through
    all tasks, creates the environment, then runs NUM_ROLLOUTS test episodes per scene.
    At the end, prints out all tests that were successful, and then any that were not
    completed successfully along with their errors.
    """

    successful = []
    unsucessful = []
    error_dict = {}

    if env_names is None or len(env_names) == 0:
        env_names = list(TARGET_TASKS["composite"])

    for i, env_name in enumerate(env_names):
        print(
            colored(
                f"[{i+1}/{len(env_names)}] Testing {env_name}",
                "green",
            )
        )

        completed = True
        try:
            env = create_env(
                env_name, seed=env_seed, split=split, use_novel_instructions=True
            )
            run_random_rollouts(
                env,
                num_rollouts=num_rollouts,
                num_steps=num_steps,
                video_path=f"../novel_lang_test/{env_name}.mp4",
            )
            ep_meta = env.get_ep_meta()
            with open(f"../novel_lang_test/{env_name}_lang.txt", "a") as f:
                f.write(ep_meta["lang"] + "\n")
            env.close()
        except KeyboardInterrupt:
            print(colored(f"Exiting Test Early.", "yellow"))
            break
        except Exception:
            print(colored(f"Test {env_name} Failed:\n{traceback.format_exc()}", "red"))
            if env_name not in error_dict.keys():
                error_dict[env_name] = []
            error_dict[env_name].append(traceback.format_exc())
            completed = False

        if completed:
            successful.append(env_name)
        else:
            unsucessful.append(env_name)

        print()

    print(colored(f"The following tests ran successfully:\n{successful}\n", "green"))
    print(colored(f"The following tests ran unsuccessfully:\n{unsucessful}", "red"))
    if error_dict:
        print(colored(f"With the following errors:", "red"))
        for key in error_dict.keys():
            print(colored(f"{key}:", "red"))
            for i, error in enumerate(error_dict[key]):
                print(colored(f"Error #{i+1}:", "red"))
                print(colored(f"{error}\n", "red"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="(optional) list of environments to test for. if not specified, tests all of them",
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=0,
        help="Environment seed",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=100,
        help="Number of rollouts per task",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of steps to run each rollout",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="split. choose amongst all, train, test",
    )
    args = parser.parse_args()

    test_tasks_validity(
        env_names=args.envs,
        num_rollouts=args.num_rollouts,
        env_seed=args.env_seed,
        num_steps=args.num_steps,
        split=args.split,
    )
