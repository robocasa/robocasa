import argparse

from robocasa.models.objects.kitchen_objects import *


def list_of_strings(arg):
    return arg.split(",")


def get_items_by_type(types=["vegetable", "fruit"]):

    valid_items = set(types)

    res = []
    for key, val in OBJ_CATEGORIES.items():
        if len(set(val.types).intersection(valid_items)) > 0:
            res.append(key)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--types",
        type=list_of_strings,
    )
    args = parser.parse_args()

    res = get_items_by_type(args.types)
