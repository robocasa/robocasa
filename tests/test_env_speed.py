from robocasa.scripts.bench_speed import run_bench, get_args

import argparse
import numpy as np
from termcolor import colored

TASKS_COMPOSITE30 = [
    "PrewashFoodAssembly",
    "StackBowlsInSink",
    "DryDrinkware",
    "DryDishes",
    "SnackSorting",
    "SteamInMicrowave",
    "VeggieDipPrep",
    "ArrangeBreadBasket",
    "SetBowlsForSoup",
    "SizeSorting",
    "ServeSteak",
    "PanTransfer",
    "PrepForSanitizing",
    "StockingBreakfastFoods",
    "BeverageSorting",
    "SimmeringSauce",
    "WaffleReheat",
    "SpicyMarinade",
    "PrepMarinatingMeat",
    "PrepareToast",
    "CheesyBread",
    "FryingPanAdjustment",
    "MealPrepStaging",
    "SearingMeat",
    "DefrostByCategory",
    "FoodCleanup",
    "BowlAndCup",
    "ArrangeTea",
    "HeatMultipleWater",
    "CupcakeCleanup",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--layouts",
        type=int,
        nargs="+",
    )
    args = parser.parse_args()

    all_infos = {}

    task_list = args.tasks
    if task_list is None:
        task_list = ["PnPCounterToStove"]

    layout_list = args.layouts
    if layout_list is None:
        layout_list = list(range(0, 10)) + list(range(101, 121))

    for task in task_list:
        all_infos[task] = {}
        for layout in layout_list:
            print(colored(f"Task: {task}; Layout: {layout}", "yellow"))
            info = run_bench(
                env_name=task,
                robots="PandaOmron",
                onscreen=True,
                camera="robot0_frontview",
                seed=0,
                n_trials=30,
                layout=layout,
                style=-1,
            )
            all_infos[task][layout] = info
            print()

    print()
    print(colored("***** Summary of stats *****", "yellow"))
    for task in all_infos.keys():
        for layout in all_infos[task].keys():
            reset_time_list = all_infos[task][layout]["reset_time_list"]
            steps_per_sec_list = all_infos[task][layout]["steps_per_sec_list"]

            print(colored(f"Task: {task}; Layout: {layout}", "yellow"))
            print(
                colored(
                    "reset time: {:.2f}s".format(np.mean(reset_time_list)), "yellow"
                )
            )
            print(colored("fps: {:.2f}".format(np.mean(steps_per_sec_list)), "yellow"))
            print()
