# Kitchen Scenes

RoboCasa offers a large array of kitchen scenes with fully interactive cabinets, drawers, and appliances.
We model 10 layouts ranging from basic designs found in apartments to more elaborate designs found in high-end homes. Each layout can be configured to take on one of 12 unique styles, resulting in 120 kitchen scenes. See the breakdown of layouts and styles as follows.

Environments can be initialized with specific layouts and styles via integer enums defined [here](https://github.com/robocasa/robocasa/blob/main/robocasa/models/scenes/scene_registry.py). For example:
```
from robocasa.utils.env_utils import create_env
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

env = create_env(
    env_name="PnPCounterToCab",
    layout_ids=[LayoutType.ONE_WALL_SMALL, LayoutType.L_SHAPED_LARGE, LayoutType.WRAPAROUND],
    style_ids=[StyleType.COASTAL, StyleType.FARMHOUSE, StyleType.RUSTIC],
)
```

## Layouts

Layouts are configured by setting the arrangment of fixtures (cabinets, microwaves, counters, etc.) in a yaml file [**[Source]**](https://github.com/robocasa/robocasa/tree/main/robocasa/models/assets/scenes/kitchen_layouts).

![Kitchen Layouts](../images/layouts.png)

## Styles

Styles are configured by setting relevant textures and fixture attributes (cabinet handle types, door types, coffee machine types, etc) in a yaml file [**[Source]**](https://github.com/robocasa/robocasa/tree/main/robocasa/models/assets/scenes/kitchen_styles).

![Kitchen Styles](../images/styles.png)
