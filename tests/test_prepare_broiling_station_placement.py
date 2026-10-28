"""Task-specific fresh placement constraints, without constructing a simulator."""
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.kitchen.composite.broiling_fish.prepare_broiling_station import (
    PrepareBroilingStation,
)
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.utils.errors import PlacementError


def _placements(tongs=(0.0, 0.0, 0.9), plate=(0.1, 0.0, 0.9)):
    quaternion = (1.0, 0.0, 0.0, 0.0)
    return {
        "tongs": (tongs, quaternion, object()),
        "plate": (plate, quaternion, object()),
    }


def _task(sampler):
    task = object.__new__(PrepareBroilingStation)
    task.deterministic_reset = False
    task._ep_meta = {}
    task.fxtr_placements = {"fixture": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), object())}
    task.placement_initializer = SimpleNamespace(sample=sampler)
    return task


def test_rejects_near_pair_and_preserves_first_acceptable_sample():
    near = _placements(plate=(0.2, 0.0, 0.9))
    far = _placements(plate=(0.5, 0.0, 0.9))
    candidates = iter((near, far))
    calls = []

    def sample(*, placed_objects):
        calls.append(placed_objects)
        return next(candidates)

    task = _task(sample)
    result = task._sample_object_placements()

    assert result is far
    assert len(calls) == 2
    assert all(fixtures is task.fxtr_placements for fixtures in calls)
    assert near["plate"][0] == (0.2, 0.0, 0.9)


@pytest.mark.parametrize(
    "plate,accepted",
    [
        ((np.nextafter(0.40, 0.0), 0.0, 0.9), False),
        ((0.40, 0.0, 0.9), True),
        ((0.0, -0.40, 0.9), True),
        ((0.30, 0.30, 0.9), True),
    ],
)
def test_distance_boundary_uses_xy_norm(plate, accepted):
    candidate = _placements(plate=plate)
    fallback = _placements(plate=(0.60, 0.0, 0.9))
    candidates = iter((candidate, fallback))
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return next(candidates)

    result = _task(sample)._sample_object_placements()

    assert result is (candidate if accepted else fallback)
    assert len(calls) == (1 if accepted else 2)


def test_vertical_separation_does_not_make_a_near_xy_pair_acceptable():
    near_with_large_z_gap = _placements(plate=(0.1, 0.1, 100.0))
    far_with_same_z = _placements(plate=(0.5, 0.0, 0.9))
    candidates = iter((near_with_large_z_gap, far_with_same_z))
    task = _task(lambda **kwargs: next(candidates))

    assert task._sample_object_placements() is far_with_same_z


def test_acceptable_distance_does_not_require_different_counters():
    placement = _placements(plate=(0.5, 0.0, 0.9))
    task = _task(lambda **kwargs: placement)
    counter = object()
    task.counter_items = task.counter_oven = counter

    assert task._sample_object_placements() is placement
    assert task.counter_items is task.counter_oven is counter


def test_exhausted_distance_attempts_raise_instead_of_accepting_invalid_pair():
    placement = _placements(plate=(0.1, 0.0, 0.9))
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return placement

    task = _task(sample)

    with pytest.raises(PlacementError):
        task._sample_object_placements()

    assert len(calls) == task.MAX_PLACEMENT_ATTEMPTS == 100


def test_underlying_sampler_failure_reaches_kitchen_scene_retry():
    error = PlacementError("The fixture cannot contain the sampled object")
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        raise error

    task = _task(sample)
    with pytest.raises(PlacementError) as caught:
        task._sample_object_placements()

    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "deterministic_reset,metadata",
    [(True, {}), (False, {"object_cfgs": []})],
    ids=["deterministic_xml_reset", "recorded_metadata_before_xml_reset"],
)
def test_recorded_scene_paths_do_not_reject_or_resample_near_pairs(
    deterministic_reset, metadata,
):
    placement = _placements(plate=(0.1, 0.0, 0.9))
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return placement

    task = _task(sample)
    task.deterministic_reset = deterministic_reset
    task._ep_meta = metadata

    assert task._sample_object_placements() is placement
    assert len(calls) == 1
    assert calls[0] == {"placed_objects": task.fxtr_placements}
    assert task._ep_meta is metadata


def test_default_kitchen_path_preserves_unconstrained_sampler_behavior():
    placement = _placements(plate=(0.1, 0.0, 0.9))
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return placement

    kitchen = object.__new__(Kitchen)
    kitchen.fxtr_placements = {"counter": object()}
    kitchen.placement_initializer = SimpleNamespace(sample=sample)

    assert kitchen._sample_object_placements() is placement
    assert calls == [{"placed_objects": kitchen.fxtr_placements}]
