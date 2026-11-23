import unittest
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from robocasa.models.fixtures.counter import Counter
from robocasa.models.fixtures.fixture_stack import FixtureStack


class TestFixtureSeedDeterminism(unittest.TestCase):
    def test_counter_regions_keep_first_occurrence_order(self):
        offsets = [-1.5, -0.5, 0.5, 1.5]
        geoms = [
            ElementTree.Element("geom", pos=f"{x} 0 0", size="0.3 0.3 0.1")
            for x in offsets
        ]
        reference = SimpleNamespace(pos=np.array([3.0, 0.0, 0.0]), rot=0.0)
        env = SimpleNamespace(get_fixture=lambda _: reference)

        # Exercise selection with a reference fixture and more than one region.
        # Reuse the same XML nodes in both orders so their identity stays fixed.
        for ordered, expected in ((geoms, offsets), (geoms[::-1], offsets[::-1])):
            candidates = [ordered[0], ordered[1], ordered[0], *ordered[2:]]
            counter = SimpleNamespace(
                _get_counter_geoms=lambda: {
                    f"top_{i}": [geom] for i, geom in enumerate(candidates)
                },
                pos=np.zeros(3),
                rot=0.0,
                size=np.array([4.0, 1.0, 1.0]),
            )
            regions = Counter.get_reset_regions(counter, env, ref="nearby", loc="any")
            self.assertEqual(
                [region["offset"][0] for region in regions.values()], expected
            )

    def test_fixture_stack_passes_rng_to_base_and_level(self):
        rng = np.random.default_rng(42)
        fixtures = {}
        FixtureStack(
            config={
                "name": "test_stack",
                "size": [0.8, 0.6, 1.0],
                "pos": [0.0, 0.0, 0.5],
                "levels": ["box"],
                "percentages": [1.0],
            },
            scene_fixtures=fixtures,
            scene_configs={},
            scene_style={},
            rng=rng,
        )

        self.assertIs(fixtures["test_stack_base"].rng, rng)
        self.assertIs(fixtures["test_stack_1"].rng, rng)
