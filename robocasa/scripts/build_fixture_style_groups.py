"""
Build fixture viewer `data-ids` and `data-style-groups` strings from:

- `robocasa/demo_style_mapping.json` (demo <-> style conversion)
- kitchen style YAMLs (style -> fixture model, e.g. sink)
- fixture registry YAML (valid fixture models)

Primary use: generate a deduped, style-ordered slider list for the docs fixtures page.

Example:
  python3 robocasa/scripts/build_fixture_style_groups.py \\
    --fixtures-dir /Users/sepehrnasiriany/robocasa-dev/docs/_static/fixtures/sinks \\
    --fixture sink
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FixtureSpec:
    # Key in kitchen style YAMLs (e.g. "sink", "stove")
    style_yaml_key: str
    # Fixture registry YAML name (e.g. "sink.yaml", "stove.yaml")
    registry_yaml: str
    # Model key prefix in registry (e.g. "Sink", "Stove")
    model_prefix: str


_FIXTURE_SPECS: Dict[str, FixtureSpec] = {
    "blender": FixtureSpec(
        style_yaml_key="blender", registry_yaml="blender.yaml", model_prefix="Blender"
    ),
    "sink": FixtureSpec(
        style_yaml_key="sink", registry_yaml="sink.yaml", model_prefix="Sink"
    ),
    "stove": FixtureSpec(
        style_yaml_key="stove", registry_yaml="stove.yaml", model_prefix="Stove"
    ),
    "stove_wide": FixtureSpec(
        style_yaml_key="stove_wide",
        registry_yaml="stove_wide.yaml",
        model_prefix="Stove",
    ),
    "microwave": FixtureSpec(
        style_yaml_key="microwave",
        registry_yaml="microwave.yaml",
        model_prefix="Microwave",
    ),
    "dishwasher": FixtureSpec(
        style_yaml_key="dishwasher",
        registry_yaml="dishwasher.yaml",
        model_prefix="Dishwasher",
    ),
    "fridge_bottom_freezer": FixtureSpec(
        style_yaml_key="fridge_bottom_freezer",
        registry_yaml="fridge_bottom_freezer.yaml",
        model_prefix="Refrigerator",
    ),
    "fridge_side_by_side": FixtureSpec(
        style_yaml_key="fridge_side_by_side",
        registry_yaml="fridge_side_by_side.yaml",
        model_prefix="Refrigerator",
    ),
    "fridge_french_door": FixtureSpec(
        style_yaml_key="fridge_french_door",
        registry_yaml="fridge_french_door.yaml",
        model_prefix="Refrigerator",
    ),
    "coffee_machine": FixtureSpec(
        style_yaml_key="coffee_machine",
        registry_yaml="coffee_machine.yaml",
        model_prefix="CoffeeMachine",
    ),
    "oven": FixtureSpec(
        style_yaml_key="oven",
        registry_yaml="oven.yaml",
        model_prefix="Oven",
    ),
    "toaster": FixtureSpec(
        style_yaml_key="toaster",
        registry_yaml="toaster.yaml",
        model_prefix="Toaster",
    ),
    "toaster_oven": FixtureSpec(
        style_yaml_key="toaster_oven",
        registry_yaml="toaster_oven.yaml",
        model_prefix="ToasterOven",
    ),
    "stovetop": FixtureSpec(
        style_yaml_key="stovetop",
        registry_yaml="stovetop.yaml",
        model_prefix="Stovetop",
    ),
    "electric_kettle": FixtureSpec(
        style_yaml_key="electric_kettle",
        registry_yaml="electric_kettle.yaml",
        model_prefix="ElectricKettle",
    ),
    "stand_mixer": FixtureSpec(
        style_yaml_key="stand_mixer",
        registry_yaml="stand_mixer.yaml",
        model_prefix="StandMixer",
    ),
}


def _find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find robocasa/demo_style_mapping.json.
    """
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / "robocasa" / "demo_style_mapping.json").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find repo root from {start}. Expected robocasa/demo_style_mapping.json somewhere above."
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_inline_comment(value: str) -> str:
    # e.g. "Sink012 #Sink025 # Sink006" -> "Sink012"
    return value.split("#", 1)[0].strip()


def _parse_style_yaml_fixture(style_yaml_path: Path, fixture_key: str) -> str:
    """
    Extracts the fixture model token from a kitchen style yaml line like:
      sink: Sink012 # ...
    """
    for raw in _read_text(style_yaml_path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(f"{fixture_key}:"):
            continue
        value = line.split(":", 1)[1].strip()
        value = _strip_inline_comment(value)
        # take first token in case there's any extra whitespace
        return value.split()[0]
    raise KeyError(f"Missing '{fixture_key}:' in {style_yaml_path}")


def _parse_fixture_registry_keys(registry_path: Path, key_prefix: str) -> List[str]:
    """
    Parse top-level keys like Sink001, Sink002, ... from fixture registry yaml.
    We intentionally avoid depending on PyYAML here.
    """
    keys: List[str] = []
    pat = re.compile(rf"^({re.escape(key_prefix)}[0-9]{{3}}):\s*$")
    for raw in _read_text(registry_path).splitlines():
        if raw.lstrip().startswith("#"):
            continue
        m = pat.match(raw)
        if m:
            keys.append(m.group(1))
    return keys


@dataclass(frozen=True)
class GroupEntry:
    model: str
    styles: Tuple[int, ...]  # style ids in this group (sorted)
    rep_style: int  # minimum style id (for ordering)
    rep_demo: int  # demo id for rep_style (image id)


def build_groups_for_fixture(
    *,
    repo_root: Path,
    fixture: str,
    fixtures_dir: Path,
    style_min: int = 1,
    style_max: int = 60,
    split_dirs: Iterable[str] = ("train", "test"),
    layout_id: Optional[int] = None,
) -> Tuple[List[GroupEntry], Dict[int, Tuple[int, str]]]:
    """
    Returns:
      - groups (deduped by model; ordered by rep_style)
      - style_map: style_id -> (demo_id, model)
    """
    fixture = fixture.strip().lower()
    spec = _FIXTURE_SPECS.get(fixture)
    if spec is None:
        raise NotImplementedError(
            f"Unsupported --fixture '{fixture}'. Supported: {', '.join(sorted(_FIXTURE_SPECS.keys()))}"
        )

    demo_map_path = repo_root / "robocasa" / "demo_style_mapping.json"
    data = json.loads(_read_text(demo_map_path))

    # Use layout-specific mapping if layout_id is provided
    if layout_id is not None:
        layout_key = f"layout_{layout_id}"
        if layout_key not in data:
            raise ValueError(f"Layout {layout_id} not found in demo_style_mapping.json")
        layout_data = data[layout_key]
        # Build style_to_demo from demo_to_style for layout
        style_to_demo: Dict[int, int] = {}
        for demo_name, demo_info in layout_data["demo_to_style"].items():
            style_id = int(demo_info["style_id"])
            demo_id = int(demo_name.split("_")[1])
            style_to_demo[style_id] = demo_id
    else:
        # Use default style_to_demo mapping
        style_to_demo: Dict[int, int] = {
            int(style_id): int(demo_name.split("_")[1])
            for style_id, demo_name in data["style_to_demo"].items()
        }

    # style_id -> fixture model (e.g. Sink012, Stove076)
    style_to_model: Dict[int, str] = {}
    for style_id in range(style_min, style_max + 1):
        style_path: Optional[Path] = None
        for split in split_dirs:
            p = (
                repo_root
                / "robocasa"
                / "models"
                / "assets"
                / "scenes"
                / "kitchen_styles"
                / split
                / f"style{style_id:03d}.yaml"
            )
            if p.exists():
                style_path = p
                break
        if style_path is None:
            raise FileNotFoundError(
                f"Missing kitchen style yaml for style{style_id:03d} in splits {list(split_dirs)}"
            )
        style_to_model[style_id] = _parse_style_yaml_fixture(
            style_path, spec.style_yaml_key
        )

    # Validate model is in fixture registry
    registry_path = (
        repo_root
        / "robocasa"
        / "models"
        / "assets"
        / "fixtures"
        / "fixture_registry"
        / spec.registry_yaml
    )
    valid_models = set(_parse_fixture_registry_keys(registry_path, spec.model_prefix))
    bad_models = sorted({m for m in style_to_model.values() if m not in valid_models})
    if bad_models:
        raise ValueError(
            f"Some style {fixture} models not found in {registry_path}: {bad_models}"
        )

    # Deduplicate: model -> [styles...]
    model_to_styles: Dict[str, List[int]] = defaultdict(list)
    for style_id, model in style_to_model.items():
        model_to_styles[model].append(style_id)
    for model in model_to_styles:
        model_to_styles[model].sort()

    # Build group entries
    groups: List[GroupEntry] = []
    for model, styles in model_to_styles.items():
        # Ordering rule: sort groups by their minimum style id.
        rep_style = styles[0]

        # Display rule override (docs UX): for fridge fixtures, if a group contains both
        # style 2 and style 8, prefer the style 8 image (style 2's renders tend to look
        # noticeably darker in dark mode).
        # For coffee_machine, if a group contains both style 19 and style 32, prefer style 32's image.
        # For fridge_bottom_freezer, if a group contains both style 14 and style 16, prefer style 16's image.
        # Layout-specific overrides:
        # - Fridge (Bottom Freezer) layout 1: style 15 → style 51 image
        # - Fridge (French Door) layout 6: style 3 → style 10 image, style 5 → style 6 image, style 33 → style 51 image
        # - Fridge (Side by Side) layout 7: style 3 → style 4 image
        # - Stove layout 1: style 14 → style 51 image, style 24 → style 60 image, style 29 → style 55 image, style 53 → style 56 image, style 28 → style 31 image
        # - Stovetop layout 7: style 11 → style 35 image, style 15 → style 56 image, style 26 → style 55 image
        # - Blender layout 13: style 17 → style 56 image
        # - Stove (wide) layout 13: style 11 → style 60 image, style 15 → style 55 image, style 17 → style 19 image
        rep_demo_style = rep_style
        if fixture.startswith("fridge_") and rep_style == 2 and 8 in styles:
            rep_demo_style = 8
        elif fixture == "coffee_machine" and rep_style == 19 and 32 in styles:
            rep_demo_style = 32
        elif fixture == "fridge_bottom_freezer" and rep_style == 14 and 16 in styles:
            rep_demo_style = 16
        elif (
            fixture == "fridge_bottom_freezer"
            and layout_id == 1
            and rep_style == 15
            and 51 in styles
        ):
            rep_demo_style = 51
        elif fixture == "fridge_french_door" and layout_id == 6:
            if rep_style == 3 and 10 in styles:
                rep_demo_style = 10
            elif rep_style == 5 and 6 in styles:
                rep_demo_style = 6
            elif rep_style == 33 and 51 in styles:
                rep_demo_style = 51
        elif (
            fixture == "fridge_side_by_side"
            and layout_id == 7
            and rep_style == 3
            and 4 in styles
        ):
            rep_demo_style = 4
        elif fixture == "stove" and layout_id == 1:
            if rep_style == 14 and 51 in styles:
                rep_demo_style = 51
            elif rep_style == 24 and 60 in styles:
                rep_demo_style = 60
            elif rep_style == 29 and 55 in styles:
                rep_demo_style = 55
            elif rep_style == 53 and 56 in styles:
                rep_demo_style = 56
            elif rep_style == 28 and 31 in styles:
                rep_demo_style = 31
        elif fixture == "stovetop" and layout_id == 7:
            if rep_style == 11 and 35 in styles:
                rep_demo_style = 35
            elif rep_style == 15 and 56 in styles:
                rep_demo_style = 56
            elif rep_style == 26 and 55 in styles:
                rep_demo_style = 55
        elif fixture == "blender" and layout_id == 13:
            if rep_style == 17 and 56 in styles:
                rep_demo_style = 56
        elif fixture == "stove_wide" and layout_id == 13:
            if rep_style == 11 and 60 in styles:
                rep_demo_style = 60
            elif rep_style == 15 and 55 in styles:
                rep_demo_style = 55
            elif rep_style == 17 and 19 in styles:
                rep_demo_style = 19

        rep_demo = style_to_demo[rep_demo_style]
        groups.append(
            GroupEntry(
                model=model,
                styles=tuple(styles),
                rep_style=rep_style,
                rep_demo=rep_demo,
            )
        )

    # Order by lowest style id (style order)
    groups.sort(key=lambda g: g.rep_style)

    # Ensure representative demo images exist (fallback: pick first available in group)
    missing: List[Tuple[int, str]] = []
    fixed_groups: List[GroupEntry] = []
    for g in groups:
        rep_demo = g.rep_demo
        if not (fixtures_dir / f"{rep_demo}.png").exists():
            # fallback: pick any style in group whose demo image exists
            fallback_demo = None
            for s in g.styles:
                d = style_to_demo[s]
                if (fixtures_dir / f"{d}.png").exists():
                    fallback_demo = d
                    break
            if fallback_demo is None:
                missing.append((g.rep_demo, f"{fixtures_dir / f'{g.rep_demo}.png'}"))
            else:
                rep_demo = fallback_demo
        fixed_groups.append(
            GroupEntry(
                model=g.model, styles=g.styles, rep_style=g.rep_style, rep_demo=rep_demo
            )
        )

    if missing:
        raise FileNotFoundError(
            "Missing representative images:\n"
            + "\n".join(f"- demo {d}: {p}" for d, p in missing)
        )

    style_map: Dict[int, Tuple[int, str]] = {
        s: (style_to_demo[s], style_to_model[s])
        for s in range(style_min, style_max + 1)
    }
    return fixed_groups, style_map


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build fixture viewer data-ids and data-style-groups from demo/style mapping and kitchen style yamls."
    )
    parser.add_argument(
        "--fixtures-dir",
        required=True,
        type=Path,
        help="Path to the fixture images directory (e.g. .../docs/_static/fixtures/sinks). Images are demo_id.png",
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="Fixture name (currently only 'sink'). If omitted, inferred from fixtures-dir basename.",
    )
    parser.add_argument("--style-min", type=int, default=1)
    parser.add_argument("--style-max", type=int, default=60)
    parser.add_argument(
        "--layout-id",
        type=int,
        default=None,
        help="Layout ID to use for mapping (e.g., 6 for layout_6). If not provided, uses default mapping.",
    )
    parser.add_argument(
        "--print-style-map",
        action="store_true",
        help="Also print the full style_id -> demo_id -> model mapping table.",
    )
    args = parser.parse_args(argv)

    fixtures_dir = args.fixtures_dir.expanduser().resolve()
    if not fixtures_dir.exists() or not fixtures_dir.is_dir():
        raise FileNotFoundError(
            f"--fixtures-dir does not exist or is not a directory: {fixtures_dir}"
        )

    fixture = (args.fixture or fixtures_dir.name).strip().lower()
    if fixture.endswith("s"):
        fixture = fixture[:-1]  # sinks -> sink

    repo_root = _find_repo_root(fixtures_dir)

    groups, style_map = build_groups_for_fixture(
        repo_root=repo_root,
        fixture=fixture,
        fixtures_dir=fixtures_dir,
        style_min=args.style_min,
        style_max=args.style_max,
        layout_id=args.layout_id,
    )

    data_ids = ",".join(str(g.rep_demo) for g in groups)
    data_groups = ";".join("|".join(str(s) for s in g.styles) for g in groups)

    print(f"fixture={fixture}")
    print(f"repo_root={repo_root}")
    print(f"fixtures_dir={fixtures_dir}")
    print(f"unique_models={len(groups)}")
    print("")
    print("DATA_IDS=")
    print(data_ids)
    print("")
    print("DATA_STYLE_GROUPS=")
    print(data_groups)

    if args.print_style_map:
        print("\nSTYLE_MAPPING (style_id: demo_id  model):")
        for style_id in range(args.style_min, args.style_max + 1):
            demo_id, model = style_map[style_id]
            print(f"{style_id:02d}: demo_{demo_id:02d}  {model}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
