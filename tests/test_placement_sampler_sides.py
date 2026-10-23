import pytest

from robocasa.utils.placement_samplers import UniformRandomSampler


class MidpointRng:
    def __init__(self):
        self.bounds = None

    def uniform(self, *, high, low):
        self.bounds = (low, high)
        return (low + high) / 2


@pytest.mark.parametrize(
    ("side", "axis", "expected_bounds"),
    [
        ("all", "x", (0.1, 0.9)),
        ("all", "y", (0.1, 0.9)),
        ("left", "x", (0.1, 0.5)),
        ("right", "x", (0.5, 0.9)),
        ("front", "y", (0.1, 0.5)),
        ("back", "y", (0.5, 0.9)),
        ("front_left", "x", (0.1, 0.5)),
        ("front_left", "y", (0.1, 0.5)),
        ("back_right", "x", (0.5, 0.9)),
        ("back_right", "y", (0.5, 0.9)),
    ],
)
def test_uniform_sampler_restricts_valid_center_range_by_side(
    side, axis, expected_bounds
):
    rng = MidpointRng()
    sampler = UniformRandomSampler(
        name="test",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        side=side,
        rng=rng,
    )

    getattr(sampler, f"_sample_{axis}")((0.2, 0.2, 0.1))

    assert rng.bounds == expected_bounds


def test_uniform_sampler_rejects_unknown_side():
    with pytest.raises(ValueError, match="Invalid value for side"):
        UniformRandomSampler(name="test", side="near_wall")
