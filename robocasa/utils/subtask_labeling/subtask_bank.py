"""The subtask bank: RoboCasa's 65 atomic tasks, used as the controlled
vocabulary for composite-task subtask labels.

The bank is loaded from the docs registry
(docs/atomic_tasks/atomic_task_attributes.js), which maps each atomic task to
its canonical natural-language description. Composite-task labelers should name
each subtask with a bank entry so labels are consistent across all tasks.
"""

import json
import os
import re

import robocasa

# docs/ sits next to the robocasa package.
_BANK_PATH = os.path.join(
    os.path.dirname(robocasa.__path__[0]),
    "docs",
    "atomic_tasks",
    "atomic_task_attributes.js",
)


def _load_bank():
    """Parse the `window.X = {...};` JS file into {task_name: description}."""
    try:
        with open(_BANK_PATH) as f:
            txt = f.read()
    except OSError:
        return {}
    m = re.search(r"=\s*(\{.*\})\s*;?\s*$", txt.strip(), re.S)
    if not m:
        return {}
    raw = json.loads(m.group(1))
    return {name: attrs.get("description", "") for name, attrs in raw.items()}


# name -> canonical description string
BANK = _load_bank()

# Sentinel subtask name for the terminal "task complete" segment.
DONE = "done"


def is_known(name: str) -> bool:
    """True if `name` is a bank atomic task (or the DONE sentinel)."""
    return name == DONE or name in BANK


def is_pick_place(name: str) -> bool:
    """True if `name` is a pick-and-place skill (these split into pick/place
    stages; all other skills stay as a single stage)."""
    return name.startswith("PickPlace")


def description(name: str) -> str:
    """Canonical bank description for `name`, or '' if unknown."""
    return BANK.get(name, "")
