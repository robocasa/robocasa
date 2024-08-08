import random
from collections import OrderedDict

import numpy as np
from robosuite.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

SPEC = OrderedDict(
    PnPCounterToCab=dict(
        groups="all",
        exclude_groups=[
            "condiment_bottle",
            "baguette",
            "kettle_electric",
            "avocado",
            "can",
        ],
        graspable=True,
    ),
    PnPCabToCounter=dict(
        groups="all",
        exclude_groups=["beer", "orange", "jam", "canned_food", "coffee_cup"],
        graspable=True,
    ),
    PnPCounterToSink=dict(
        groups="all",
        exclude_groups=["apple", "banana", "bar_soap", "cup", "cucumber"],
        graspable=True,
        washable=True,
    ),
    PnPSinkToCounter=dict(
        groups="food",
        exclude_groups=["peach", "lime", "yogurt", "fish", "kiwi"],
        graspable=True,
        washable=True,
    ),
    PnPCounterToMicrowave=dict(
        groups="food",
        exclude_groups=["broccoli", "cheese", "bell_pepper", "squash", "sweet_potato"],
        graspable=True,
        heatable=True,
    ),
    PnPMicrowaveToCounter=dict(
        groups="food",
        exclude_groups=["corn", "tomato", "hot_dog", "egg", "carrot"],
    ),
    PnPCounterToStove=dict(
        groups="food",
        exclude_groups=["potato", "garlic", "steak", "eggplant", "mango"],
        graspable=True,
        heatable=True,
    ),
)


def count_obj_cats(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    heatable=None,
    freezable=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(groups, tuple) and not isinstance(groups, list):
        groups = [groups]

    if exclude_groups is None:
        exclude_groups = []
    if not isinstance(exclude_groups, tuple) and not isinstance(exclude_groups, list):
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
            cat_meta = OBJ_CATEGORIES[cat]
            if graspable is True and cat_meta.graspable is not True:
                continue
            if washable is True and cat_meta.washable is not True:
                continue
            if heatable is True and cat_meta.heatable is not True:
                continue
            if freezable is True and cat_meta.freezable is not True:
                continue

            valid_categories.append(cat)

    return valid_categories


for task in SPEC:
    valid_cats = count_obj_cats(**SPEC[task])
    SPEC[task]["valid_cats"] = valid_cats
    print(task)
    print("{} total cats:".format(len(valid_cats)), valid_cats)
    print()
