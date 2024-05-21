import random
import numpy as np

RNG = np.random.default_rng(12)

ALL_TASK_VARIANTS = [
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnOnStove",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    "OpenDoorSingleHinge",
    "CloseDoorSingleHinge",
    "OpenDoorDoubleHinge",
    "CloseDoorDoubleHinge",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
]

### add pick-place tasks ###
PnP_TRAINING_OBJ_CATS = dict(
    PnPCounterToCab=['liquor', 'apple', 'banana', 'bar', 'bar_soap', 'beer', 'bell_pepper', 'bottled_drink', 'bottled_water', 'bowl', 'boxed_drink', 'boxed_food', 'broccoli', 'cake', 'candle', 'canned_food', 'carrot', 'cereal', 'cheese', 'coffee_cup', 'corn', 'croissant', 'cucumber', 'cup', 'cupcake', 'donut', 'egg', 'eggplant', 'fish', 'garlic', 'hot_dog', 'jam', 'jug', 'ketchup', 'kettle_non_electric', 'kiwi', 'ladle', 'lemon', 'lime', 'mango', 'milk', 'mug', 'mushroom', 'onion', 'orange', 'pan', 'pot', 'peach', 'pear', 'potato', 'rolling_pin', 'shaker', 'soap_dispenser', 'sponge', 'spray', 'squash', 'steak', 'sweet_potato', 'tangerine', 'teapot', 'tomato', 'water_bottle', 'wine', 'yogurt'],
    PnPCabToCounter=['liquor', 'apple', 'avocado', 'baguette', 'banana', 'bar', 'bar_soap', 'bell_pepper', 'bottled_drink', 'bottled_water', 'bowl', 'boxed_drink', 'boxed_food', 'broccoli', 'cake', 'can', 'candle', 'carrot', 'cereal', 'cheese', 'condiment_bottle', 'corn', 'croissant', 'cucumber', 'cup', 'cupcake', 'donut', 'egg', 'eggplant', 'fish', 'garlic', 'hot_dog', 'jug', 'ketchup', 'kettle_electric', 'kettle_non_electric', 'kiwi', 'ladle', 'lemon', 'lime', 'mango', 'milk', 'mug', 'mushroom', 'onion', 'pan', 'pot', 'peach', 'pear', 'potato', 'rolling_pin', 'shaker', 'soap_dispenser', 'sponge', 'spray', 'squash', 'steak', 'sweet_potato', 'tangerine', 'teapot', 'tomato', 'water_bottle', 'wine', 'yogurt'],
    PnPCounterToSink=['liquor', 'avocado', 'beer', 'bell_pepper', 'bottled_drink', 'bottled_water', 'bowl', 'boxed_drink', 'broccoli', 'can', 'canned_food', 'carrot', 'cheese', 'condiment_bottle', 'corn', 'egg', 'eggplant', 'fish', 'garlic', 'jam', 'jug', 'ketchup', 'kettle_electric', 'kettle_non_electric', 'kiwi', 'ladle', 'lemon', 'lime', 'mango', 'milk', 'mug', 'mushroom', 'onion', 'orange', 'pan', 'pot', 'peach', 'pear', 'potato', 'rolling_pin', 'shaker', 'soap_dispenser', 'sponge', 'spray', 'squash', 'steak', 'sweet_potato', 'tangerine', 'teapot', 'tomato', 'water_bottle', 'wine', 'yogurt'],
    PnPSinkToCounter=['apple', 'avocado', 'banana', 'bell_pepper', 'broccoli', 'carrot', 'cheese', 'corn', 'cucumber', 'egg', 'eggplant', 'garlic', 'lemon', 'mango', 'milk', 'mushroom', 'onion', 'orange', 'pear', 'potato', 'squash', 'steak', 'sweet_potato', 'tangerine', 'tomato'],
    PnPCounterToMicrowave=['apple', 'avocado', 'baguette', 'banana', 'carrot', 'corn', 'croissant', 'cucumber', 'donut', 'egg', 'eggplant', 'fish', 'garlic', 'hot_dog', 'lemon', 'lime', 'mango', 'milk', 'mushroom', 'onion', 'orange', 'peach', 'pear', 'potato', 'steak', 'tangerine', 'tomato'],
    PnPMicrowaveToCounter=['apple', 'avocado', 'bagel', 'baguette', 'banana', 'bell_pepper', 'bread', 'broccoli', 'cake', 'cheese', 'chocolate', 'croissant', 'cucumber', 'cupcake', 'donut', 'eggplant', 'fish', 'garlic', 'kiwi', 'lemon', 'lime', 'mango', 'milk', 'mushroom', 'onion', 'orange', 'peach', 'pear', 'potato', 'squash', 'steak', 'sweet_potato', 'tangerine', 'waffle', 'yogurt'],
    PnPCounterToStove=['apple', 'avocado', 'baguette', 'banana', 'bell_pepper', 'broccoli', 'carrot', 'cheese', 'corn', 'croissant', 'cucumber', 'donut', 'egg', 'fish', 'hot_dog', 'lemon', 'lime', 'milk', 'mushroom', 'onion', 'orange', 'peach', 'pear', 'squash', 'sweet_potato', 'tangerine', 'tomato'],
)

for task in PnP_TRAINING_OBJ_CATS:
    for cat in PnP_TRAINING_OBJ_CATS[task]:
        ALL_TASK_VARIANTS.append("{}-{}".format(task, cat))

total_num_variants = len(ALL_TASK_VARIANTS)

# 5 percent of tasks: 16 task variants
tasks_5p = RNG.choice(ALL_TASK_VARIANTS, 16)
for variant in tasks_5p:
    print(variant)

"""
tasks_5p

PnPCounterToCab-tangerine
PnPCounterToCab-bell_pepper
PnPCounterToCab-mushroom
PnPCounterToCab-pan
PnPCounterToCab-pear
PnPCounterToCab-cucumber

PnPCabToCounter-liquor
PnPCabToCounter-garlic

PnPCounterToSink-tangerine
PnPCounterToSink-potato
PnPCounterToSink-boxed_drink

PnPSinkToCounter-lemon
PnPSinkToCounter-eggplant

PnPCounterToStove-egg
PnPCounterToStove-mushroom
PnPCounterToStove-cucumber
"""

