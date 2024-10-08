## Composite Tasks


### Washing fruits and vegetables

| Task                  | Description                                                                                                                                               | Class File                                                                                                                                                |
|:----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PrewashFoodAssembly` | Pick the fruit/vegetable from the cabinet and place it in the bowl. Then pick the bowl and place it in the sink. Then turn on the sink facuet.            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_fruits_and_vegetables/prewash_food_assembly.py) |
| `ClearClutter`        | Pick up the fruits and vegetables and place them in the sink. Then, turn on the sink to wash them. Then, turn the sink off, put them in the tray.         | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_fruits_and_vegetables/clear_clutter.py)         |
| `DrainVeggies`        | Dump the vegetable from the pot into the sink. Then turn on the sink and wash the vegetable. Then turn off the sink and put the vegetable back in the pot | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_fruits_and_vegetables/drain_veggies.py)         |
| `AfterwashSorting`    | Pick the foods of the same kind from the sink and place them in one bowl. Place the other food in the other bowl. Then, turn off the sink                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_fruits_and_vegetables/afterwash_sorting.py)     |


### Washing dishes

| Task               | Description                                                                                                 | Class File                                                                                                                           |
|:-------------------|:------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| `StackBowlsInSink` | Stack the bowls in the sink                                                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_dishes/stack_bowls.py)     |
| `PreSoakPan`       | Pick the pan and sponge and place them into the sink. Then turn on the sink.                                | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_dishes/pre_soak_pan.py)    |
| `SortingCleanup`   | Pick the mug and place it in the sink. Pick the bowl and place it in the cabinet and then close the cabinet | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_dishes/sorting_cleanup.py) |
| `DryDrinkware`     | A wet mug is on the counter and need to be dried. Pick it up and place it upside down in the open cabinet.  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_dishes/dry_drinkware.py)   |
| `DryDishes`        | Pick the cup and bowl from the sink and place them on the counter for drying                                | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/washing_dishes/dry_dishes.py)      |


### Tidying cabinets and drawers

| Task                       | Description                                                                                                     | Class File                                                                                                                                                    |
|:---------------------------|:----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OrganizeCleaningSupplies` | Open the cabinet. Pick the cleaner and place it next to the sink. Then close the cabinet.                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/tidying_cabinets_and_drawers/organize_cleaning_supplies.py) |
| `DrawerUtensilSort`        | Open the left drawer and push the utensils inside it                                                            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/tidying_cabinets_and_drawers/drawer_utensil_sort.py)        |
| `PantryMishap`             | Open the cabinet. Pick place the vegetable on the counter and the canned food in the drawer. Close the cabinet. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/tidying_cabinets_and_drawers/pantry_mishap.py)              |
| `ShakerShuffle`            | Pick place the shaker into the drawer. Then, Close the cabinet.                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/tidying_cabinets_and_drawers/shaker_shuffle.py)             |
| `SnackSorting`             | Place the bar in the bowl and close the drawer                                                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/tidying_cabinets_and_drawers/snack_sorting.py)              |


### Steaming food

| Task                | Description                                                                                                                                                                                              | Class File                                                                                                                             |
|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| `SteamInMicrowave`  | Pick the vegetable from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button.                                   | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/steaming_food/steam_in_microwave.py) |
| `MultistepSteaming` | Turn on the sink. Then move the vegetable from the counter to the sink. Turn of the sink. Move the vegetable from the sink to the pot next to the stove. Finally, move the pot to the front left burner. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/steaming_food/multistep_steaming.py) |
| `SteamVegetables`   | Place vegetables into the pot based on the amount of time it would take to steam each. e.g. potatoes and carrots would take the longest. Then, turn off the burner beneath the pot.                      | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/steaming_food/steam_vegetables.py)   |


### Snack preparation

| Task                | Description                                                                                                     | Class File                                                                                                                                  |
|:--------------------|:----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| `CerealAndBowl`     | Open the cabinet, pick the cereal and bowl from the cabinet and place it on the counter. Then close the cabinet | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/snack_preparation/cereal_and_bowl.py)     |
| `BreadAndCheese`    | Pick the bread and cheese, place them on the cutting board,                                                     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/snack_preparation/bread_and_cheese.py)    |
| `YogurtDelightPrep` | Place the yogurt and fruit onto the counter.                                                                    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/snack_preparation/yogurt_delight_prep.py) |
| `MakeFruitBowl`     | Open the cabinet. Pick the fruits from the cabinet and place them into the bowl. Then close the cabinet.        | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/snack_preparation/make_fruit_bowl.py)     |
| `VeggieDipPrep`     | Place the two vegetables and a bowl onto the tray for setting up a vegetable dip station.                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/snack_preparation/veggie_dip_prep.py)     |


### Setting the table

| Task                   | Description                                                                                                           | Class File                                                                                                                                    |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
| `DateNight`            | Pick up the decoration and the alcohol in the cabinet and move them to the dining counter.                            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/date_night.py)            |
| `ArrangeBreadBasket`   | Open the cabinet, pick up the bread from the cabinet and place in the bowl. Then move the bowl to the dining counter. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/arrange_bread_basket.py)  |
| `SeasoningSpiceSetup`  | Move the condiments from the cabinet directly in front to the dining counter                                          | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/seasoning_spice_setup.py) |
| `SetBowlsForSoup`      | Move the bowls from the cabinet to the plates on the dining table                                                     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/set_bowls_for_soup.py)    |
| `BeverageOrganization` | Move the drinks to the dining counter                                                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/beverage_organization.py) |
| `SizeSorting`          | Stack the cup/bowl from largest to smallest                                                                           | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/setting_the_table/size_sorting.py)          |


### Serving food

| Task                 | Description                                                                                                                                        | Class File                                                                                                                              |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|
| `WineServingPrep`    | Open the cabinet directly in front. Then, move the alcohol and the cup to the counter with the decoration on it.                                   | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/wine_serving_prep.py)    |
| `ServeSteak`         | Pick up the pan with the steak in it and place it on the dining table. Then, place the steak on the plate                                          | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/serve_steak.py)          |
| `PlaceFoodInBowls`   | Pick both bowls and place them on the counter. Then pick the food and place it in one bowl and pick the other food and place it in the other bowl. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/place_food_in_bowls.py)  |
| `DessertUpgrade`     | Move the dessert items from the plate to the tray                                                                                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/dessert_upgrade.py)      |
| `PrepareSoupServing` | Open the cabinet and move the ladle to the pot. Then close the cabinet.                                                                            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/prepare_soup_serving.py) |
| `PanTransfer`        | Pick up the pan and dump the vegetables in it onto the plate. Then, return the pan to the stove.                                                   | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/serving_food/pan_transfer.py)         |


### Sanitize surface

| Task                 | Description                                                                                                                                                                    | Class File                                                                                                                                   |
|:---------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| `PushUtensilsToSink` | Push the utensils into the sink                                                                                                                                                | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/sanitize_surface/push_utensils_to_sink.py) |
| `PrepForSanitizing`  | Pick the cleaning supplies from the cabinet and place it on the counter                                                                                                        | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/sanitize_surface/prep_for_sanitizing.py)   |
| `CleanMicrowave`     | Open the microwave. Then, pick the sponge from the counter and place it in the microwave                                                                                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/sanitize_surface/clean_microwave.py)       |
| `CountertopCleanup`  | Pick  the fruit and vegetable from the counter and place it in the cabinet. Then, open the drawer and pick the cleaner and sponge from the drawer and place it on the counter. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/sanitize_surface/countertop_cleanup.py)    |


### Restocking supplies

| Task                     | Description                                                                                                              | Class File                                                                                                                                         |
|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| `RestockPantry`          | Pick the cans from the counter and place them in their designated side in the cabinet                                    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/restocking_supplies/restock_pantry.py)           |
| `StockingBreakfastFoods` | Pick the packaged food from the counter and place it in the closest cabinet to them.                                     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/restocking_supplies/stocking_breakfast_foods.py) |
| `RestockBowls`           | Open the cabinet. Pick the bowls from the counter and place it in the cabinet directly in front. Then close the cabinet. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/restocking_supplies/restock_bowls.py)            |
| `BeverageSorting`        | Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the other.                                         | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/restocking_supplies/beverage_sorting.py)         |


### Reheating food

| Task               | Description                                                                                                                            | Class File                                                                                                                              |
|:-------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|
| `HeatMug`          | Pick the mug from the cabinet and place it inside the microwave. Then close the microwave                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/reheating_food/heat_mug.py)           |
| `SimmeringSauce`   | Place the pan on the rear right burner on the stove. Then place the tomato and the onion in the pan and turn on the rear right burner. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/reheating_food/simmering_sauce.py)    |
| `WaffleReheat`     | Open the microwave, place the bowl with waffle inside the microwave, then close the microwave door and turn it on.                     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/reheating_food/waffle_reheat.py)      |
| `WarmCroissant`    | Pick the croissant and place it on the pan. Then turn on the stove to warm the croissant.                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/reheating_food/warm_croissant.py)     |
| `MakeLoadedPotato` | Retrieve the reheated potato from the microwave, then place it on the cutting board along with cheese and a bottle of condiment.       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/reheating_food/make_loaded_potato.py) |


### Mixing and blending

| Task            | Description                                                                                                         | Class File                                                                                                                               |
|:----------------|:--------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| `SetupJuicing`  | Open the cabinet, pick all 4 fruits from the cabinet and place them on the counter                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/mixing_and_blending/setup_juicing.py)  |
| `ColorfulSalsa` | Place the avocado, onion, tomato and bell pepper on the cutting board                                               | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/mixing_and_blending/colorful_salsa.py) |
| `SpicyMarinade` | Open the cabinet. Place the bowl and condiment on the counter. Then place the lime and garlic on the cutting board. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/mixing_and_blending/spicy_marinade.py) |


### Meat preparation

| Task                 | Description                                                                                                                                         | Class File                                                                                                                                  |
|:---------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| `PrepMarinatingMeat` | Pick the meat from its container and place it on the cutting board. Then pick the condiment from the cabinet and place it next to the cutting board | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/meat_preparation/prep_marinating_meat.py) |
| `PrepForTenderizing` | Retrieve a rolling pin from the cabinet and place it next to the meat on the cutting board to prepare for tenderizing.                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/meat_preparation/prep_for_tenderizing.py) |


### Making toast

| Task                    | Description                                                                                                                                                                                                                                                                                                              | Class File                                                                                                                                  |
|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| `PrepareToast`          | Open the cabinet, pick the bread, place it on the cutting board, pick the jam, place it on the counter, and close the cabinet.                                                                                                                                                                                           | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/making_toast/prepare_toast.py)            |
| `SweetSavoryToastSetup` | Pick the avocado and bread from the counter and place it on the plate. Then pick the jam from the cabinet and place it next to the plate. Lastly, close the cabinet door                                                                                                                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/making_toast/sweet_savory_toast_setup.py) |
| `CheesyBread`           | Start with a slice of bread already on a plate and a wedge of cheese on the counter. Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish.                                                                                                                            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/making_toast/cheesy_bread.py)             |
| `BreadSelection`        | Prepare to make a delicious snack by gathering the right type of bread. From the different types of pastries on the counter, select a croissant and place it on the cutting board. Additionally, retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board to complete the setup. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/making_toast/bread_selection.py)          |


### Frying

| Task                   | Description                                                                                                                             | Class File                                                                                                                          |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| `SetupFrying`          | Pick the pan from the cabinet and place it on the stove. Then turn on the stove burner for the pan.                                     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/frying/setup_frying.py)           |
| `FryingPanAdjustment`  | Pick and place the pan from the current burner to another burner and turn the burner on                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/frying/frying_pan_adjustment.py)  |
| `MealPrepStaging`      | Pick place both pans onto different burners. Then, place the vegetable and the meat on different pans                                   | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/frying/meal_prep_staging.py)      |
| `AssembleCookingArray` | Move the meat onto the pan on the stove. Then, move the condiment and vegetable from the cabinet to the counter where the plate is.     | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/frying/assemble_cooking_array.py) |
| `SearingMeat`          | Grab the pan from the cabinet and place on the front left burner on the stove. Then place the meat on the stove and turn the burner on. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/frying/searing_meat.py)           |


### Defrosting food

| Task                | Description                                                                                                                                                                                                         | Class File                                                                                                                                |
|:--------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| `ThawInSink`        | Pick the object from the counter and place it in the sink. Then turn on the sink faucet.                                                                                                                            | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/defrosting_food/thaw_in_sink.py)        |
| `MicrowaveThawing`  | Pick the food from the counter and place it in the microwave. Then turn on the microwave.                                                                                                                           | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/defrosting_food/microwave_thawing.py)   |
| `QuickThaw`         | Frozen meat rests on a plate on the counter. Retrieve the meat and place it in a pot on a burner. Then, turn the burner on.                                                                                         | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/defrosting_food/quick_thaw.py)          |
| `DefrostByCategory` | There is a mixed pile of frozen fruits and vegetables on the counter. Locate all the frozen vegetables and place the items in a bowl on the counter. Take all the frozen fruits and defrost them in a running sink. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/defrosting_food/defrost_by_category.py) |


### Clearing table

| Task                          | Description                                                                                                    | Class File                                                                                                                                         |
|:------------------------------|:---------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| `CondimentCollection`         | Pick the condiments from the counter and place it in the cabinet                                               | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/condiment_collection.py)          |
| `DessertAssembly`             | Pick up the dessert inside a container and place it on the tray. Pick up the cupcake and place it on the tray. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/dessert_assembly.py)              |
| `ClearingCleaningReceptacles` | Pick the receptacles and place them in the sink. Then, turn on the water                                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/clearing_cleaning_receptacles.py) |
| `CandleCleanup`               | Pick the decorations from the dining table and place it in the open cabinet                                    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/candle_cleanup.py)                |
| `FoodCleanup`                 | Pick the food from the counter and place it in the cabinet. Then close the cabinet                             | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/food_cleanup.py)                  |
| `DrinkwareConsolidation`      | Pick the drinks from the island and place it in the open cabinet                                               | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/drinkware_consolidation.py)       |
| `BowlAndCup`                  | Place the cup inside the bowl on the island and move it to any counter                                         | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/clearing_table/bowl_and_cup.py)                  |


### Chopping food

| Task                      | Description                                                                                                                        | Class File                                                                                                                                     |
|:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| `ArrangeVegetables`       | Pick the vegetables from the sink and place them on the cutting board                                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/chopping_food/arrange_vegetables.py)         |
| `OrganizeVegetables`      | Place the vegetables on separate cutting boards                                                                                    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/chopping_food/organize_vegetables.py)        |
| `BreadSetupSlicing`       | Place all breads on the cutting board                                                                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/chopping_food/bread_setup_slicing.py)        |
| `ClearingTheCuttingBoard` | Clear the non-vegetable object off the cutting board and place the vegetables onto it.                                             | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/chopping_food/clearing_the_cutting_board.py) |
| `MeatTransfer`            | Retrieve a container (either a pan or a bowl) from the cabinet, then place the raw meat into the container to avoid contamination. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/chopping_food/meat_transfer.py)              |


### Brewing

| Task            | Description                                                                                                                                           | Class File                                                                                                                   |
|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| `KettleBoiling` | Pick the kettle from the counter and place it on a stove burner. Then, turn the burner on                                                             | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/brewing/kettle_boiling.py) |
| `ArrangeTea`    | Pick the kettle from the counter and place it on the tray. Then pick the mug from the cabinet and place it on the tray. Then close the cabinet doors. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/brewing/arrange_tea.py)    |
| `PrepareCoffee` | Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button                                                | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/brewing/prepare_coffee.py) |


### Boiling

| Task                | Description                                                                                                                                                                                                                    | Class File                                                                                                                        |
|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| `FillKettle`        | Open the cabinet, pick the kettle from the cabinet, and place it in the sink                                                                                                                                                   | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/boiling/fill_kettle.py)         |
| `HeatMultipleWater` | Pick the kettle from the cab and place it on a stove burner. Then, pick the pot from the counter and place on another stove burner. Finally, turn both burners on                                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/boiling/heat_multiple_water.py) |
| `VeggieBoil`        | Pick up the pot and place it in the sink. Then turn on the sink and let the pot fill up with water. Then turn the sink off and move the pot to the stove. Lastly, turn on the stove and place the food in the pot for boiling. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/boiling/veggie_boil.py)         |


### Baking

| Task                        | Description                                                                                                   | Class File                                                                                                                               |
|:----------------------------|:--------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| `PastryDisplay`             | Place the pastrys on the plates                                                                               | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/baking/pastry_display.py)              |
| `OrganizeBakingIngredients` | Pick place the eggs and milk to near the bowl                                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/baking/organize_baking_ingredients.py) |
| `CupcakeCleanup`            | Move the fresh-baked cupcake off the tray onto the counter, and place the bowl used for mixing into the sink. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/multi_stage/baking/cupcake_cleanup.py)             |