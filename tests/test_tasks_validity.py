import argparse
from termcolor import colored
import traceback

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts


def test_tasks_validity(env_seed=0, num_rollouts=10, num_steps=20):
    """
    Tests that all kitchen environment tasks run error free. Iterates through
    all tasks, creates the environment, then runs NUM_ROLLOUTS test episodes per scene.
    At the end, prints out all tests that were successful, and then any that were not
    completed successfully along with their errors.
    """

    successful = []
    unsucessful = []
    error_dict = {}

    for i, env_name in enumerate(list(ALL_KITCHEN_ENVIRONMENTS)):
        print(
            colored(
                f"Testing {env_name} environment [{i}/{len(list(ALL_KITCHEN_ENVIRONMENTS))}]...",
                "green",
            )
        )

        completed = True
        try:
            env = create_env(env_name, seed=env_seed)
            run_random_rollouts(
                env,
                num_rollouts=num_rollouts,
                num_steps=num_steps,
                video_path=f"/tmp/{env_name}.mp4",
            )
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
        "--env_seed",
        type=int,
        default=0,
        help="Environment seed",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollouts per task",
    )
    args = parser.parse_args()

    test_tasks_validity(
        num_rollouts=args.num_rollouts,
        env_seed=args.env_seed,
        num_steps=args.num_rollouts,
    )
