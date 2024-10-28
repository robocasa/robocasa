from robosuite.environments.base import make

# Manipulation environments
from robocasa.environments.kitchen.kitchen import Kitchen, KitchenDemo
from robocasa.environments.kitchen.multi_stage.baking.cupcake_cleanup import (
    CupcakeCleanup,
)
from robocasa.environments.kitchen.multi_stage.baking.organize_baking_ingredients import (
    OrganizeBakingIngredients,
)
from robocasa.environments.kitchen.multi_stage.baking.pastry_display import (
    PastryDisplay,
)
from robocasa.environments.kitchen.multi_stage.boiling.fill_kettle import FillKettle
from robocasa.environments.kitchen.multi_stage.boiling.heat_multiple_water import (
    HeatMultipleWater,
)
from robocasa.environments.kitchen.multi_stage.boiling.veggie_boil import VeggieBoil
from robocasa.environments.kitchen.multi_stage.brewing.arrange_tea import ArrangeTea
from robocasa.environments.kitchen.multi_stage.brewing.kettle_boiling import (
    KettleBoiling,
)
from robocasa.environments.kitchen.multi_stage.brewing.prepare_coffee import (
    PrepareCoffee,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.arrange_vegetables import (
    ArrangeVegetables,
    ArrangeVegetablesSimple,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.bread_setup_slicing import (
    BreadSetupSlicing,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.clearing_the_cutting_board import (
    ClearingTheCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.meat_transfer import (
    MeatTransfer,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.organize_vegetables import (
    OrganizeVegetables,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.bowl_and_cup import (
    BowlAndCup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.candle_cleanup import (
    CandleCleanup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.clearing_cleaning_receptacles import (
    ClearingCleaningReceptacles,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.condiment_collection import (
    CondimentCollection,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.dessert_assembly import (
    DessertAssembly,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.drinkware_consolidation import (
    DrinkwareConsolidation,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.food_cleanup import (
    FoodCleanup,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.defrost_by_category import (
    DefrostByCategory,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.microwave_thawing import (
    MicrowaveThawing,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.quick_thaw import (
    QuickThaw,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.thaw_in_sink import (
    ThawInSink,
)
from robocasa.environments.kitchen.multi_stage.frying.assemble_cooking_array import (
    AssembleCookingArray,
)
from robocasa.environments.kitchen.multi_stage.frying.frying_pan_adjustment import (
    FryingPanAdjustment,
)
from robocasa.environments.kitchen.multi_stage.frying.meal_prep_staging import (
    MealPrepStaging,
)
from robocasa.environments.kitchen.multi_stage.frying.searing_meat import SearingMeat
from robocasa.environments.kitchen.multi_stage.frying.setup_frying import SetupFrying
from robocasa.environments.kitchen.multi_stage.making_toast.bread_selection import (
    BreadSelection,
)
from robocasa.environments.kitchen.multi_stage.making_toast.cheesy_bread import (
    CheesyBread,
)
from robocasa.environments.kitchen.multi_stage.making_toast.prepare_toast import (
    PrepareToast,
)
from robocasa.environments.kitchen.multi_stage.making_toast.sweet_savory_toast_setup import (
    SweetSavoryToastSetup,
)
from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_for_tenderizing import (
    PrepForTenderizing,
)
from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_marinating_meat import (
    PrepMarinatingMeat,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.colorful_salsa import (
    ColorfulSalsa,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.setup_juicing import (
    SetupJuicing,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.spicy_marinade import (
    SpicyMarinade,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.heat_mug import HeatMug
from robocasa.environments.kitchen.multi_stage.reheating_food.make_loaded_potato import (
    MakeLoadedPotato,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.simmering_sauce import (
    SimmeringSauce,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.waffle_reheat import (
    WaffleReheat,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.warm_croissant import (
    WarmCroissant,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.beverage_sorting import (
    BeverageSorting,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_bowls import (
    RestockBowls,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_pantry import (
    RestockPantry,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.stocking_breakfast_foods import (
    StockingBreakfastFoods,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.clean_microwave import (
    CleanMicrowave,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.countertop_cleanup import (
    CountertopCleanup,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.prep_for_sanitizing import (
    PrepForSanitizing,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.push_utensils_to_sink import (
    PushUtensilsToSink,
)
from robocasa.environments.kitchen.multi_stage.serving_food.dessert_upgrade import (
    DessertUpgrade,
)
from robocasa.environments.kitchen.multi_stage.serving_food.pan_transfer import (
    PanTransfer,
)
from robocasa.environments.kitchen.multi_stage.serving_food.place_food_in_bowls import (
    PlaceFoodInBowls,
)
from robocasa.environments.kitchen.multi_stage.serving_food.prepare_soup_serving import (
    PrepareSoupServing,
)
from robocasa.environments.kitchen.multi_stage.serving_food.serve_steak import (
    ServeSteak,
)
from robocasa.environments.kitchen.multi_stage.serving_food.wine_serving_prep import (
    WineServingPrep,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.arrange_bread_basket import (
    ArrangeBreadBasket,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.beverage_organization import (
    BeverageOrganization,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.date_night import (
    DateNight,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.seasoning_spice_setup import (
    SeasoningSpiceSetup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.set_bowls_for_soup import (
    SetBowlsForSoup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.size_sorting import (
    SizeSorting,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.bread_and_cheese import (
    BreadAndCheese,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.cereal_and_bowl import (
    CerealAndBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.make_fruit_bowl import (
    MakeFruitBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.veggie_dip_prep import (
    VeggieDipPrep,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.yogurt_delight_prep import (
    YogurtDelightPrep,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.multistep_steaming import (
    MultistepSteaming,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_in_microwave import (
    SteamInMicrowave,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_vegetables import (
    SteamVegetables,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.drawer_utensil_sort import (
    DrawerUtensilSort,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.organize_cleaning_supplies import (
    OrganizeCleaningSupplies,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.pantry_mishap import (
    PantryMishap,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.shaker_shuffle import (
    ShakerShuffle,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.snack_sorting import (
    SnackSorting,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_dishes import (
    DryDishes,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_drinkware import (
    DryDrinkware,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.pre_soak_pan import (
    PreSoakPan,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.sorting_cleanup import (
    SortingCleanup,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.stack_bowls import (
    StackBowlsInSink,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.afterwash_sorting import (
    AfterwashSorting,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.clear_clutter import (
    ClearClutter,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.drain_veggies import (
    DrainVeggies,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.prewash_food_assembly import (
    PrewashFoodAssembly,
)
from robocasa.environments.kitchen.single_stage.kitchen_coffee import (
    CoffeePressButton,
    CoffeeServeMug,
    CoffeeSetupMug,
)
from robocasa.environments.kitchen.single_stage.kitchen_doors import (
    CloseDoor,
    CloseDoubleDoor,
    CloseSingleDoor,
    OpenDoor,
    OpenDoubleDoor,
    OpenSingleDoor,
)
from robocasa.environments.kitchen.single_stage.kitchen_drawer import (
    CloseDrawer,
    OpenDrawer,
)
from robocasa.environments.kitchen.single_stage.kitchen_microwave import (
    TurnOffMicrowave,
    TurnOnMicrowave,
)
from robocasa.environments.kitchen.single_stage.kitchen_navigate import NavigateKitchen
from robocasa.environments.kitchen.single_stage.kitchen_pnp import (
    PnPCabToCounter,
    PnPCounterToCab,
    PnPCounterToMicrowave,
    PnPCounterToSink,
    PnPMicrowaveToCounter,
    PnPSinkToCounter,
)
from robocasa.environments.kitchen.single_stage.kitchen_sink import (
    TurnOffSinkFaucet,
    TurnOnSinkFaucet,
)
from robocasa.environments.kitchen.single_stage.kitchen_stove import (
    TurnOffStove,
    TurnOnStove,
)

try:
    import mimicgen
except ImportError:
    print(
        "WARNING: mimicgen environments not imported since mimicgen is not installed!"
    )

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS

import mujoco

assert (
    mujoco.__version__ == "3.1.1"
), "MuJoCo version must be 3.1.1. Please run pip install mujoco==3.1.1"

import numpy

assert numpy.__version__ in [
    "1.23.2",
    "1.23.3",
], "numpy version must be either 1.23.2 or 1.23.3. Please install one of these versions."

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
