import math
import os
import pathlib
import random
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from robosuite.utils.mjcf_utils import find_elements, string_to_array

import robocasa

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")


OBJ_CATEGORIES = dict(
    liquor=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            model_folders=["aigen_objs/alcohol"],
            scale=1.50,
        ),
        objaverse=dict(
            model_folders=["objaverse/alcohol"],
            scale=1.35,
        ),
    ),
    apple=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    avocado=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    bagel=dict(
        types=("bread_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.2,
        ),
        objaverse=dict(
            exclude=[
                "bagel_8",
            ],
        ),
    ),
    bagged_food=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            exclude=[
                "bagged_food_12",
            ],
        ),
    ),
    baguette=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "baguette_3",  # small holes on ends
            ],
        ),
    ),
    banana=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    bar=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.25, 1.25, 1.75],
        ),
        objaverse=dict(
            scale=[0.75, 0.75, 1.2],
            exclude=[
                "bar_1",  # small holes scattered
            ],
        ),
    ),
    bar_soap=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=[1.25, 1.25, 1.40],
        ),
        objaverse=dict(
            scale=[0.95, 0.95, 1.05],
            exclude=["bar_soap_2"],
        ),
    ),
    beer=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=1.15),
    ),
    bell_pepper=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.75,
        ),
    ),
    bottled_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    bottled_water=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
            exclude=[
                "bottled_water_0",  # minor hole at top
                "bottled_water_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
        ),
    ),
    bowl=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=2.0,
            exclude=[
                "bowl_21",  # can see through from bottom of bowl
            ],
        ),
    ),
    boxed_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            scale=0.80,
            exclude=[
                "boxed_drink_9",  # hole on bottom
                "boxed_drink_6",  # hole on bottom
                "boxed_drink_8",  # hole on bottom
            ],
        ),
    ),
    boxed_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.1,
            exclude=[
                "boxed_food_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
            # exclude=[
            #     "boxed_food_5",
            #     "boxed_food_3", "boxed_food_1", "boxed_food_6", "boxed_food_11", "boxed_food_10", "boxed_food_8", "boxed_food_9", "boxed_food_7", "boxed_food_2", # self turning due to single collision geom
            # ],
        ),
    ),
    bread=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[0.80, 0.80, 1.0],
        ),
        objaverse=dict(scale=[0.70, 0.70, 1.0], exclude=["bread_22"]),  # hole on bottom
    ),
    broccoli=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "broccoli_2",  # holes on one part
            ],
        ),
    ),
    cake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.8,
        ),
        objaverse=dict(
            scale=0.8,
        ),
    ),
    can=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "can_10",  # hole on bottom
                "can_5",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
    ),
    candle=dict(
        types=("decoration"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "candle_11",  # hole at bottom
                # "candle_2", # can't see from bottom view angle
                # "candle_15", # can't see from bottom view angle
            ],
        ),
    ),
    canned_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.90,
            exclude=[
                "canned_food_7",  # holes at top and bottom
            ],
        ),
    ),
    carrot=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    cereal=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            # exclude=[
            #     "cereal_2", "cereal_5", "cereal_13", "cereal_3", "cereal_9", "cereal_0", "cereal_7", "cereal_4", "cereal_8", "cereal_12", "cereal_11", "cereal_1", "cereal_6", "cereal_10", # self turning due to single collision geom
            # ]
        ),
    ),
    cheese=dict(
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
        ),
    ),
    chips=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "chips_12",  # minor hole at bottom of bag
                # "chips_2", # a weird texture at top/bottom but keeping this
            ]
        ),
    ),
    chocolate=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 1.35],
        ),
        objaverse=dict(
            scale=[0.80, 0.80, 1.20],
            exclude=[
                # "chocolate_2", # self turning due to single collision geom
            ],
        ),
    ),
    coffee_cup=dict(
        types=("drink"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "coffee_cup_18",  # can see thru top
                "coffee_cup_5",  # can see thru from bottom side
                "coffee_cup_19",  # can see thru from bottom side
            ],
        ),
    ),
    condiment_bottle=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
            model_folders=["aigen_objs/condiment"],
        ),
        objaverse=dict(
            scale=1.05,
            model_folders=["objaverse/condiment"],
        ),
    ),
    corn=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(scale=1.05),
    ),
    croissant=dict(
        types=("pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    cucumber=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    cup=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(),
    ),
    cupcake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            exclude=[
                "cupcake_0",  # can see thru bottom
                "cupcake_10",  # can see thru bottom,
                "cupcake_1",  # very small hole at bottom
            ]
        ),
    ),
    cutting_board=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.0,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "cutting_board_14",
                "cutting_board_3",
                "cutting_board_10",
                "cutting_board_6",  # these models still modeled with meshes which should work most of the time, but excluding them for safety
            ],
        ),
    ),
    donut=dict(
        types=("sweets", "pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.15,
        ),
    ),
    egg=dict(
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.85,
        ),
    ),
    eggplant=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=0.95),
    ),
    fish=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.35, 1.35, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 1.5],
        ),
    ),
    fork=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(),
    ),
    garlic=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(scale=1.10, exclude=["garlic_3"]),  # has hole on side
    ),
    hot_dog=dict(
        types=("cooked_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.4,
        ),
        objaverse=dict(),
    ),
    jam=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    jug=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.5,
        ),
    ),
    ketchup=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "ketchup_5"  # causing error: faces of mesh have inconsistent orientation.
            ]
        ),
    ),
    kettle_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i not in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i not in [0, 2, 6, 9, 10, 11]],
        ),
    ),
    kettle_non_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i in [0, 2, 6, 9, 10, 11]],
        ),
    ),
    kiwi=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    knife=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
        ),
    ),
    ladle=dict(
        types=("utensil"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.10,
        ),
    ),
    lemon=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    lime=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        objaverse=dict(
            scale=1.0,
        ),
        aigen=dict(
            scale=0.90,
        ),
    ),
    mango=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
            exclude=[
                "mango_3",  # one half is pitch dark
            ],
        ),
    ),
    milk=dict(
        types=("dairy", "drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "milk_6"  # causing error: eigenvalues of mesh inertia violate A + B >= C
            ]
        ),
    ),
    mug=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(),
    ),
    mushroom=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
            exclude=[
                # "mushroom_16", # very very small holes. keeping anyway
            ],
        ),
    ),
    onion=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    orange=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            exclude=[
                # "orange_11", # bottom half is dark. keeping anyway
            ]
        ),
    ),
    pan=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            scale=1.70,
            exclude=[
                "pan_16",  # causing error. faces of mesh have inconsistent orientation,
                "pan_0",
                "pan_12",
                "pan_17",
                "pan_22",  # these are technically what we consider "pots"
            ],
        ),
    ),
    pot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            model_folders=["objaverse/pan"],
            scale=1.70,
            exclude=list(
                set([f"pan_{i}" for i in range(25)])
                - set(["pan_0", "pan_12", "pan_17", "pan_22"])
            ),
        ),
    ),
    peach=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(),
    ),
    pear=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "pear_4",  # has big hole. excluding
            ]
        ),
    ),
    plate=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.65,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "plate_6",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
    ),
    potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(),
    ),
    rolling_pin=dict(
        types=("tool"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                # "rolling_pin_5", # can see thru side handle edges, keeping anyway
                # "rolling_pin_1", # can see thru side handle edges, keeping anyway
            ],
        ),
    ),
    scissors=dict(
        types=("tool"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.15,
        ),
    ),
    shaker=dict(
        types=("condiment"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    soap_dispenser=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.7,
        ),
        objaverse=dict(
            exclude=[
                # "soap_dispenser_4", # can see thru body but that's fine if this is glass
            ]
        ),
    ),
    spatula=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
        ),
    ),
    sponge=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.20,
        ),
        objaverse=dict(
            scale=0.90,
            # exclude=[
            #     "sponge_7", "sponge_1", # self turning due to single collision geom
            # ]
        ),
    ),
    spoon=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(),
    ),
    spray=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=1.75,
        ),
    ),
    squash=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            exclude=[
                "squash_10",  # hole at bottom
            ],
        ),
    ),
    steak=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 2.0],
            exclude=[
                "steak_13",  # bottom texture completely messed up
                "steak_1",  # bottom texture completely messed up
                # "steak_9", # bottom with some minor issues, keeping anyway
            ],
        ),
    ),
    sweet_potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(),
        objaverse=dict(),
    ),
    tangerine=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(),
    ),
    teapot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "teapot_9",  # hole on bottom
            ],
        ),
    ),
    tomato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    tray=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(scale=2.0),
        objaverse=dict(
            scale=1.80,
        ),
    ),
    waffle=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            exclude=[
                "waffle_2",  # bottom completely messed up
            ]
        ),
    ),
    water_bottle=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.5,
            exclude=[
                "water_bottle_11",  # sides and bottom see thru, but ok if glass. keeping anyway
            ],
        ),
    ),
    wine=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.9,
        ),
        objaverse=dict(
            scale=1.6,
            exclude=[
                "wine_7",  # causing error. faces of mesh have inconsistent orientation
            ],
        ),
    ),
    yogurt=dict(
        types=("dairy", "packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    dates=dict(
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
        aigen=dict(),
    ),
    lemonade=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("drink"),
    ),
    walnut=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=(),
    ),
    cheese_grater=dict(
        aigen=dict(
            scale=2.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    syrup_bottle=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("condiment"),
    ),
    scallops=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    candy=dict(
        aigen=dict(),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
    ),
    whisk=dict(
        aigen=dict(
            scale=1.8,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("utensil"),
    ),
    pitcher=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("receptacle"),
    ),
    ice_cream=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("sweets"),
    ),
    cherry=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    peanut_butter=dict(
        aigen=dict(
            scale=1.25,
            model_folders=["aigen_objs/peanut_butter_jar"],
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    thermos=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("drink"),
    ),
    ham=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    dumpling=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat", "cooked_food"),
    ),
    cabbage=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lettuce=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    tongs=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    ginger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    ice_cube_tray=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("receptacle"),
    ),
    shrimp=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    cantaloupe=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    honey_bottle=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    grapes=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    spaghetti_box=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    chili_pepper=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    celery=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    burrito=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("cooked_food"),
    ),
    olive_oil_bottle=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    kebabs=dict(
        aigen=dict(
            scale=1.65,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("cooked_food"),
    ),
    bottle_opener=dict(
        aigen=dict(),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("tool"),
    ),
    chicken_breast=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    jello_cup=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    lobster=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    brussel_sprout=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    sushi=dict(
        aigen=dict(
            scale=0.90,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("meat"),
    ),
    baking_sheet=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("receptacle"),
    ),
    wine_glass=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("receptacle"),
    ),
    asparagus=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lamb_chop=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pickle=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    bacon=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat"),
    ),
    canola_oil=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    strawberry=dict(
        aigen=dict(
            scale=0.9,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    watermelon=dict(
        aigen=dict(
            scale=2.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    pizza_cutter=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    pomegranate=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    apricot=dict(
        aigen=dict(
            scale=0.7,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    beet=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    radish=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("vegetable"),
    ),
    salsa=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    artichoke=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    scone=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("pastry", "bread_food"),
    ),
    hamburger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
    ),
    raspberry=dict(
        aigen=dict(
            scale=0.85,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    tacos=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
    ),
    vinegar=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food", "condiment"),
    ),
    zucchini=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    pork_loin=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pork_chop=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    sausage=dict(
        aigen=dict(
            scale=1.45,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    coconut=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    cauliflower=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lollipop=dict(
        aigen=dict(),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
    ),
    salami=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    butter_stick=dict(
        aigen=dict(
            scale=1.3,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("dairy"),
    ),
    can_opener=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    tofu=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=(),
    ),
    pineapple=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    skewers=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat", "cooked_food"),
    ),
)


class ObjCat:
    def __init__(
        self,
        name,
        types,
        model_folders=None,
        exclude=None,
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 2),
        density=100,
        friction=(0.95, 0.3, 0.1),
        priority=None,
        aigen_cat=False,
    ):
        self.name = name
        if not isinstance(types, tuple):
            types = (types,)
        self.types = types

        self.aigen_cat = aigen_cat

        self.graspable = graspable
        self.washable = washable
        self.microwavable = microwavable
        self.cookable = cookable
        self.freezable = freezable

        self.scale = scale
        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.priority = priority
        self.exclude = exclude or []

        if model_folders is None:
            subf = "aigen_objs" if self.aigen_cat else "objaverse"
            model_folders = ["{}/{}".format(subf, name)]
        cat_mjcf_paths = []
        for folder in model_folders:
            cat_path = os.path.join(BASE_ASSET_ZOO_PATH, folder)
            for root, _, files in os.walk(cat_path):
                if "model.xml" in files:
                    model_name = os.path.basename(root)
                    if model_name in self.exclude:
                        continue
                    cat_mjcf_paths.append(os.path.join(root, "model.xml"))
        self.mjcf_paths = sorted(cat_mjcf_paths)

    def get_mjcf_kwargs(self):
        return deepcopy(
            dict(
                scale=self.scale,
                solimp=self.solimp,
                solref=self.solref,
                density=self.density,
                friction=self.friction,
                priority=self.priority,
            )
        )


for (name, kwargs) in OBJ_CATEGORIES.items():
    common_properties = deepcopy(kwargs)
    for k in common_properties.keys():
        assert k in [
            "graspable",
            "washable",
            "microwavable",
            "cookable",
            "freezable",
            "types",
            "aigen",
            "objaverse",
        ]
    objaverse_kwargs = common_properties.pop("objaverse", None)
    aigen_kwargs = common_properties.pop("aigen", None)
    assert "scale" not in kwargs
    OBJ_CATEGORIES[name] = {}

    if objaverse_kwargs is not None:
        objaverse_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["objaverse"] = ObjCat(name=name, **objaverse_kwargs)
    if aigen_kwargs is not None:
        aigen_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["aigen"] = ObjCat(
            name=name, aigen_cat=True, **aigen_kwargs
        )


OBJ_GROUPS = dict(
    all=list(OBJ_CATEGORIES.keys()),
)

for k in OBJ_CATEGORIES:
    OBJ_GROUPS[k] = [k]

# OBJ_GROUPS["washable"] = [cat for (cat, cat_meta) in OBJ_CATEGORIES.items() if cat_meta.washable is True]
# OBJ_GROUPS["graspable"] = [cat for (cat, cat_meta) in OBJ_CATEGORIES.items() if cat_meta.graspable is True]
# OBJ_GROUPS["heatable"] = [cat for (cat, cat_meta) in OBJ_CATEGORIES.items() if cat_meta.heatable is True]

all_types = set()
for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
    # types are common to both so we only need to examine one
    k = "objaverse" if "objaverse" in cat_meta_dict else "aigen"
    all_types = all_types.union(cat_meta_dict[k].types)

for t in all_types:
    cats = []
    for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
        k = "objaverse" if "objaverse" in cat_meta_dict else "aigen"
        if t in cat_meta_dict[k].types:
            cats.append(cat)
    OBJ_GROUPS[t] = cats

food_groups = []
container_groups = []
for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
    k = "objaverse" if "objaverse" in cat_meta_dict else "aigen"
    cat_meta = cat_meta_dict[k]
    if any(
        [
            t in cat_meta.types
            for t in [
                "vegetable",
                "fruit",
                "sweets",
                "dairy",
                "meat",
                "bread_food",
                "pastry",
                "cooked_food",
            ]
        ]
    ):
        food_groups.append(cat)
    if any(
        [
            t in cat_meta.types
            for t in [
                "vegetable",
                "fruit",
                "sweets",
                "dairy",
                "meat",
                "bread_food",
                "pastry",
                "cooked_food",
            ]
        ]
    ):
        container_groups.append(cat)


OBJ_GROUPS["food"] = food_groups
# obj categories that can be placed in open receptacles
OBJ_GROUPS["in_container"] = container_groups

# custom groups
OBJ_GROUPS["container"] = ["plate"]  # , "bowl"]
OBJ_GROUPS["kettle"] = ["kettle_electric", "kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["plate", "bowl"]


def sample_kitchen_object(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse",),
    split=None,
    max_size=(None, None, None),
    object_scale=None,
):
    valid_object_sampled = False
    while valid_object_sampled is False:
        mjcf_kwargs, info = sample_kitchen_object_helper(
            groups=groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=rng,
            obj_registries=obj_registries,
            split=split,
            object_scale=object_scale,
        )

        # check if object size is within bounds
        mjcf_path = info["mjcf_path"]
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        bottom = string_to_array(
            find_elements(root=root, tags="site", attribs={"name": "bottom_site"}).get(
                "pos"
            )
        )
        top = string_to_array(
            find_elements(root=root, tags="site", attribs={"name": "top_site"}).get(
                "pos"
            )
        )
        horizontal_radius = string_to_array(
            find_elements(
                root=root, tags="site", attribs={"name": "horizontal_radius_site"}
            ).get("pos")
        )
        scale = mjcf_kwargs["scale"]
        obj_size = (
            np.array(
                [horizontal_radius[0] * 2, horizontal_radius[1] * 2, top[2] - bottom[2]]
            )
            * scale
        )
        valid_object_sampled = True
        for i in range(3):
            if max_size[i] is not None and obj_size[i] > max_size[i]:
                valid_object_sampled = False

    return mjcf_kwargs, info


def sample_kitchen_object_helper(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse",),
    split=None,
    object_scale=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # option to spawn specific object instead of sampling from a group
    if isinstance(groups, str) and groups.endswith(".xml"):
        mjcf_path = groups
        # reverse look up mjcf_path to category
        mjcf_kwargs = dict()
        cat = None
        obj_found = False
        for cand_cat in OBJ_CATEGORIES:
            for reg in obj_registries:
                if (
                    reg in OBJ_CATEGORIES[cand_cat]
                    and mjcf_path in OBJ_CATEGORIES[reg][cand_cat].mjcf_paths
                ):
                    mjcf_kwargs = OBJ_CATEGORIES[reg][cand_cat].get_mjcf_kwargs()
                    cat = cand_cat
                    obj_found = True
                    break
            if obj_found:
                break
        if obj_found is False:
            raise ValueError
        mjcf_kwargs["mjcf_path"] = mjcf_path
    else:
        if not isinstance(groups, tuple) and not isinstance(groups, list):
            groups = [groups]

        if exclude_groups is None:
            exclude_groups = []
        if not isinstance(exclude_groups, tuple) and not isinstance(
            exclude_groups, list
        ):
            exclude_groups = [exclude_groups]

        invalid_categories = []
        for g in exclude_groups:
            for cat in OBJ_GROUPS[g]:
                invalid_categories.append(cat)

        valid_categories = []
        for g in groups:
            for cat in OBJ_GROUPS[g]:
                # don't repeat if already added
                if cat in valid_categories:
                    continue
                if cat in invalid_categories:
                    continue

                # don't include if category not represented in any registry
                cat_in_any_reg = np.any(
                    [reg in OBJ_CATEGORIES[cat] for reg in obj_registries]
                )
                if not cat_in_any_reg:
                    continue

                invalid = False
                for reg in obj_registries:
                    if reg not in OBJ_CATEGORIES[cat]:
                        continue
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if graspable is True and cat_meta.graspable is not True:
                        invalid = True
                    if washable is True and cat_meta.washable is not True:
                        invalid = True
                    if microwavable is True and cat_meta.microwavable is not True:
                        invalid = True
                    if cookable is True and cat_meta.cookable is not True:
                        invalid = True
                    if freezable is True and cat_meta.freezable is not True:
                        invalid = True

                if invalid:
                    continue

                valid_categories.append(cat)

        cat = rng.choice(valid_categories)

        choices = {reg: [] for reg in obj_registries}

        for reg in obj_registries:
            if reg not in OBJ_CATEGORIES[cat]:
                choices[reg] = []
                continue
            reg_choices = deepcopy(OBJ_CATEGORIES[cat][reg].mjcf_paths)
            if split is not None:
                split_th = max(len(choices) - 3, int(math.ceil(len(reg_choices) / 2)))
                if split == "A":
                    reg_choices = reg_choices[:split_th]
                elif split == "B":
                    reg_choices = reg_choices[split_th:]
                else:
                    raise ValueError
            choices[reg] = reg_choices

        chosen_reg = random.choices(
            population=obj_registries,
            weights=[len(choices[reg]) for reg in obj_registries],
        )[0]

        mjcf_path = rng.choice(choices[chosen_reg])
        mjcf_kwargs = OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs["mjcf_path"] = mjcf_path

    if object_scale is not None:
        mjcf_kwargs["scale"] *= object_scale

    groups_containing_sampled_obj = []
    for group, group_cats in OBJ_GROUPS.items():
        if cat in group_cats:
            groups_containing_sampled_obj.append(group)

    info = {
        "groups_containing_sampled_obj": groups_containing_sampled_obj,
        "groups": groups,
        "cat": cat,
        "split": split,
        "mjcf_path": mjcf_path,
    }

    # print(mjcf_path)

    return mjcf_kwargs, info
