from robocasa.scripts.bench_speed import run_bench, get_args

import argparse
import numpy as np
from termcolor import colored

TASKS = [
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

task_infos = {}

for task in TASKS:
    args = get_args()
    args.env = task
    args.onscreen = True
    args.camera = "robot0_frontview"
    args.seed = 0
    args.n_trials = 30
    args.layout_id = -1
    args.style_id = -1

    info = run_bench(args)
    task_infos[task] = info

print()
print(colored("***** Summary of stats *****", "yellow"))
for task, info in task_infos.items():
    reset_time_list = info["reset_time_list"]
    steps_per_sec_list = info["steps_per_sec_list"]

    print(colored(f"Task: {task}", "yellow"))
    print(colored("reset time: {:.2f}s".format(np.mean(reset_time_list)), "yellow"))
    print(colored("fps: {:.2f}".format(np.mean(steps_per_sec_list)), "yellow"))
    print()
