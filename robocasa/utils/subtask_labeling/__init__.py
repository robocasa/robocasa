"""
Offline per-timestep subtask labeling for composite tasks.

Pipeline (per demo):
    1. Build the env from hdf5 env_args + model.xml.
    2. For each timestep t, replay sim state via reset_to({"states": states[t]}).
    3. Evaluate task-specific predicates over privileged sim state.
    4. Emit per-frame subtask_idx (fine-grained) + subtask_phase (doc-set) + subtask_lang.

Per-task labelers register themselves in LABELERS via @register_labeler.
"""

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    LabelResult,
    LABELERS,
    register_labeler,
    get_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import BANK

# Side-effect imports so labelers self-register on package import. Per-task
# labelers live in subpackages named after the task's activity, mirroring
# robocasa/environments/kitchen/composite/<activity>/.
from robocasa.utils.subtask_labeling.making_smoothies import (  # noqa: F401
    add_ice_cubes,
    add_sweetener,
    blend_ingredients,
    place_straw,
    prepare_smoothie,
)
from robocasa.utils.subtask_labeling.toasting_bread import (  # noqa: F401
    toast_one_slot_pair,
    get_toasted_bread,
    serve_warm_croissant,
    toast_baguette,
    toast_bagel,
    pj_sandwich_prep,
    toast_on_correct_rack,
)
from robocasa.utils.subtask_labeling.adding_ice_to_beverages import (  # noqa: F401
    make_ice_lemonade,
    place_equal_ice_cubes,
    place_ice_in_cup,
    retrieve_ice_tray,
)
from robocasa.utils.subtask_labeling.arranging_buffet import (  # noqa: F401
    arrange_buffet_dessert,
    cut_buffet_pizza,
    divide_buffet_trays,
    place_beverages_together,
    tong_buffet_setup,
)
from robocasa.utils.subtask_labeling.restocking_supplies import (  # noqa: F401
    beverage_sorting,
    fresh_produce_organization,
    refill_condiment_station,
    restock_bowls,
    restock_canned_food,
    restock_pantry,
    restock_sink_supplies,
    stocking_breakfast_foods,
)
from robocasa.utils.subtask_labeling.arranging_cabinets import (  # noqa: F401
    gather_tableware,
    reset_cabinet_doors,
    stack_cans,
)
from robocasa.utils.subtask_labeling.arranging_condiments import (  # noqa: F401
    categorize_condiments,
    line_up_condiments,
    organize_condiments,
)
from robocasa.utils.subtask_labeling.baking import (  # noqa: F401
    cookie_dough_prep,
    cool_baked_cake,
    cool_baked_cookies,
    cupcake_cleanup,
    mix_cake_frosting,
    organize_baking_ingredients,
    pastry_display,
)
from robocasa.utils.subtask_labeling.boiling import (  # noqa: F401
    boil_corn,
    boil_eggs,
    boil_pot,
    cool_kettle,
    fill_kettle,
    heat_multiple_water,
    place_lid_to_boil,
    start_electric_kettle,
)
from robocasa.utils.subtask_labeling.brewing import (  # noqa: F401
    arrange_tea,
    deliver_brewed_coffee,
    kettle_boiling,
    organize_coffee_condiments,
    prepare_coffee,
    sweeten_coffee,
)
from robocasa.utils.subtask_labeling.broiling_fish import (  # noqa: F401
    oven_broil_fish,
    prepare_broiling_station,
    remove_broiled_fish,
    toaster_oven_broil_fish,
    wash_fish,
)
from robocasa.utils.subtask_labeling.chopping_vegetables import (  # noqa: F401
    cutting_tool_selection,
    gather_cutting_tools,
)
from robocasa.utils.subtask_labeling.cleaning_appliances import (  # noqa: F401
    clean_blender_jug,
    prep_fridge_for_cleaning,
    prep_sink_for_cleaning,
)
from robocasa.utils.subtask_labeling.chopping_food import (  # noqa: F401
    arrange_cutting_fruits,
    arrange_vegetables,
    bread_setup_slicing,
    clear_cutting_board,
    meat_transfer,
    organize_vegetables,
)
from robocasa.utils.subtask_labeling.cleaning_sink import (  # noqa: F401
    clear_food_waste,
    clear_sink_area,
    rinse_sink_basin,
)
from robocasa.utils.subtask_labeling.clearing_table import (  # noqa: F401
    bowl_and_cup,
    candle_cleanup,
    clear_receptacles_for_cleaning,
    cluster_items_for_clearing,
    condiment_collection,
    dessert_assembly,
    drinkware_consolidation,
    food_cleanup,
)
from robocasa.utils.subtask_labeling.defrosting_food import (  # noqa: F401
    defrost_by_category,
    microwave_thawing,
    microwave_thawing_fridge,
    move_to_counter,
    quick_thaw,
    thaw_in_sink,
)
from robocasa.utils.subtask_labeling.filling_serving_dishes import (  # noqa: F401
    build_appetizer_plate,
    display_meat_variety,
    meat_skewer_assembly,
    mixed_fruit_platter,
)
from robocasa.utils.subtask_labeling.frying import (  # noqa: F401
    assemble_cooking_array,
    distribute_steak_on_pans,
    frying_pan_adjustment,
    meal_prep_staging,
    press_chicken,
    rotate_pan,
    searing_meat,
    setup_frying,
)
from robocasa.utils.subtask_labeling.garnishing_dishes import (  # noqa: F401
    add_lemon_to_fish,
    add_sugar_cubes,
    garnish_cake,
    garnish_cupcake,
    garnish_pancake,
)
from robocasa.utils.subtask_labeling.loading_dishwasher import (  # noqa: F401
    load_dishwasher,
    prepare_dishwasher,
)
from robocasa.utils.subtask_labeling.loading_fridge import (  # noqa: F401
    create_child_friendly_fridge,
    load_condiments_in_fridge,
    load_fridge_by_type,
    load_fridge_fifo,
    load_prepared_food,
    move_freezer_to_fridge,
    place_veggies_in_drawer,
    rearrange_fridge_items,
)
from robocasa.utils.subtask_labeling.making_juice import (  # noqa: F401
    choose_ripe_fruit,
    fill_blender_jug,
    juice_fruit_reamer,
)
from robocasa.utils.subtask_labeling.making_salads import (  # noqa: F401
    prepare_cheese_station,
    wash_lettuce,
)
from robocasa.utils.subtask_labeling.making_tea import (  # noqa: F401
    arrange_tea_accompaniments,
    serve_tea,
    strainer_setup,
)
from robocasa.utils.subtask_labeling.making_toast import (  # noqa: F401
    bread_selection,
    prepare_toast,
    sweet_savory_toast_setup,
)
from robocasa.utils.subtask_labeling.managing_freezer_space import (  # noqa: F401
    clear_freezer,
    freeze_bottled_waters,
    freeze_ice_tray,
    maximize_freezer_space,
    move_fridge_to_freezer,
    move_to_freezer_drawer,
    reorganize_frozen_vegetables,
    separate_freezer_rack,
)
from robocasa.utils.subtask_labeling.measuring_ingredients import (  # noqa: F401
    choose_measuring_cup,
    organize_measuring_cups,
    weigh_ingredients,
)
from robocasa.utils.subtask_labeling.meat_preparation import (  # noqa: F401
    prep_for_tenderizing,
    prep_marinating_meat,
)
from robocasa.utils.subtask_labeling.microwaving_food import (  # noqa: F401
    filter_microwavable_item,
    microwave_correct_meal,
    microwave_defrost_meat,
    place_microwave_safe_item,
    reheat_meal,
    return_heated_food,
)
from robocasa.utils.subtask_labeling.mixing_and_blending import (  # noqa: F401
    colorful_salsa,
    make_banana_milkshake,
    spicy_marinade,
)
from robocasa.utils.subtask_labeling.mixing_ingredients import (  # noqa: F401
    blend_salsa_mix,
    blend_vegetable_sauce,
    cheese_mixing,
    make_cheesecake_filling,
    make_chocolate_milk,
    prepare_veggie_dip,
)
from robocasa.utils.subtask_labeling.organizing_dishes_and_containers import (  # noqa: F401
    empty_dish_rack,
    organize_mugs_by_handle,
    stack_bowls_cabinet,
)
from robocasa.utils.subtask_labeling.organizing_recycling import (  # noqa: F401
    recycle_bottles_by_size,
    recycle_bottles_by_type,
    recycle_soda_cans,
    recycle_stacked_yogurt,
)
from robocasa.utils.subtask_labeling.organizing_utensils import (  # noqa: F401
    arrange_utensils_by_type,
    cluster_utensils_in_drawer,
    organize_metallic_utensils,
)
from robocasa.utils.subtask_labeling.packing_lunches import (  # noqa: F401
    pack_food_by_temp,
    pack_fruit_container,
    pack_identical_lunches,
    pack_snack,
)
from robocasa.utils.subtask_labeling.plating_food import (  # noqa: F401
    balanced_meal_prep,
    plate_steak_meal,
    plate_store_dinner,
)
from robocasa.utils.subtask_labeling.portioning_meals import (  # noqa: F401
    distribute_chicken,
    portion_fruit_bowl,
    portion_hot_dogs,
    portion_in_tupperware,
    portion_on_size,
    portion_yogurt,
    scale_portioning,
)
from robocasa.utils.subtask_labeling.preparing_hot_chocolate import (  # noqa: F401
    add_marshmallow,
    sweeten_hot_chocolate,
)
from robocasa.utils.subtask_labeling.preparing_marinade import (  # noqa: F401
    blend_marinade,
    gather_marinade_ingredients,
    place_meat_in_marinade,
)
from robocasa.utils.subtask_labeling.preparing_sandwiches import (  # noqa: F401
    gather_vegetables,
    heat_kebab_sandwich,
    hot_dog_setup,
    prepare_sandwich_station,
    prepare_sausage_cheese,
    toast_heatable_ingredients,
)
from robocasa.utils.subtask_labeling.reheating_food import (  # noqa: F401
    heat_mug,
    make_loaded_potato,
    reheat_meat_on_stove,
    simmering_sauce,
    waffle_reheat,
    warm_croissant,
)
from robocasa.utils.subtask_labeling.sanitizing_cutting_board import (  # noqa: F401
    remove_cutting_board_items,
    rinse_cutting_board,
    sanitize_prep_cutting_board,
    scrub_cutting_board,
)
from robocasa.utils.subtask_labeling.sanitizing_surface import (  # noqa: F401
    arrange_sink_sanitization,
    clean_microwave,
    countertop_cleanup,
    prep_for_sanitizing,
    sanitize_sink,
    wipe_table,
)
from robocasa.utils.subtask_labeling.sauteing_vegetables import (  # noqa: F401
    adjust_heat,
    butter_on_pan,
    place_vegetables_evenly,
    preheat_pot,
    shake_pan,
    stir_vegetables,
    tilt_pan,
)
from robocasa.utils.subtask_labeling.seasoning_food import (  # noqa: F401
    lemon_seasoning_fish,
    seasoning_steak,
    setup_spice_station,
)
from robocasa.utils.subtask_labeling.serving_beverages import (  # noqa: F401
    deliver_straw,
    match_cup_and_drink,
    prepare_cocktail_station,
    prepare_drink_station,
    serve_meal_juice,
    setup_soda_bowl,
)
from robocasa.utils.subtask_labeling.serving_food import (  # noqa: F401
    alcohol_serving_prep,
    dessert_upgrade,
    pan_transfer,
    place_food_in_bowls,
    prepare_soup_serving,
    serve_steak,
)
from robocasa.utils.subtask_labeling.setting_the_table import (  # noqa: F401
    align_silverware,
    arrange_bread_basket,
    arrange_bread_bowl,
    arrange_drinkware,
    beverage_organization,
    date_night,
    seasoning_spice_setup,
    set_bowls_for_soup,
    setup_bowls,
    setup_butter_plate,
    setup_fruit_bowl,
    setup_wine_glasses,
    size_sorting,
)
from robocasa.utils.subtask_labeling.slicing_meat import (  # noqa: F401
    clean_board,
    retrieve_meat,
    set_up_cutting_station,
)
from robocasa.utils.subtask_labeling.slow_cooking import (  # noqa: F401
    add_to_soup_pot,
    begin_slow_cooking,
    stop_slow_cooking,
)
from robocasa.utils.subtask_labeling.simmering_sauces import (  # noqa: F401
    turn_off_simmered_sauce_heat,
)
from robocasa.utils.subtask_labeling.snack_preparation import (  # noqa: F401
    bread_and_cheese,
    cereal_and_bowl,
    make_fruit_bowl,
    veggie_dip_prep,
    yogurt_delight_prep,
)
from robocasa.utils.subtask_labeling.sorting_ingredients import (  # noqa: F401
    separate_raw_ingredients,
    sort_breakfast_ingredients,
)
from robocasa.utils.subtask_labeling.steaming_food import (  # noqa: F401
    multistep_steaming,
    steam_fish,
    steam_in_microwave,
)
from robocasa.utils.subtask_labeling.steaming_vegetables import (  # noqa: F401
    prepare_veggies_for_steaming,
    remove_steamed_vegetables,
    steam_veggies_with_water,
)
from robocasa.utils.subtask_labeling.storing_leftovers import (  # noqa: F401
    freeze_cooked_food,
    prepare_storing_leftovers,
    store_dumplings,
    store_leftovers_by_type,
    store_leftovers_in_bowl,
)
from robocasa.utils.subtask_labeling.tidying_cabinets_and_drawers import (  # noqa: F401
    drawer_utensil_sort,
    organize_cleaning_supplies,
    place_breakfast_items_away,
    snack_sorting,
    utensil_shuffle,
)
from robocasa.utils.subtask_labeling.washing_dishes import (  # noqa: F401
    pre_soak_pan,
    dry_dishes,
    dry_drinkware,
    clear_sink,
    collect_washing_supplies,
    dump_leftovers,
    divide_basins,
    place_dishes_by_sink,
    place_on_dish_rack,
    pre_rinse_station,
    return_washing_supplies,
    rinse_bowls,
    rinse_fragile_item,
    scrub_bowl,
    soak_sponge,
    sorting_cleanup,
    stack_bowls_in_sink,
    transport_cookware,
)
from robocasa.utils.subtask_labeling.washing_fruits_and_vegetables import (  # noqa: F401
    wash_fruit_colander,
    afterwash_sorting,
    airdry_fruit,
    clear_clutter,
    clear_sink_space,
    drain_veggies,
    gather_produce_washing,
    prepare_vegetable_roasting,
    prewash_food_assembly,
    prewash_food_sorting,
    wash_in_saucepan,
)

__all__ = [
    "BaseSubtaskLabeler",
    "LabelResult",
    "LABELERS",
    "BANK",
    "register_labeler",
    "get_labeler",
]
