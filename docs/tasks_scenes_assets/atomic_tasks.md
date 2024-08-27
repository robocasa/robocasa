## Atomic Tasks

The current release includes 25 atomic tasks for systematically training policies across eight foundational skill families. The breakdown of these tasks across skill families is as follows:

### Pick and place

| Task                          | Description                                                                                               | Class File                                                                                                         |
|:------------------------------|:----------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| `PickPlaceCounterToCabinet`   | Pick an object from the counter and place it inside the cabinet. The cabinet is already open.             | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceCabinetToCounter`   | Pick an object from the cabinet and place it on the counter. The cabinet is already open.                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceCounterToSink`      | Pick an object from the counter and place it in the sink.                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceSinkToCounter`      | Pick an object from the sink and place it on the counter area next to the sink.                           | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceCounterToMicrowave` | Pick an object from the counter and place it inside the microwave. The microwave door is already open.    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceMicrowaveToCounter` | Pick an object from inside the microwave and place it on the counter. The microwave door is already open. | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceCounterToStove`     | Pick an object from the counter and place it in a pan or pot on the stove.                                | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |
| `PickPlaceStoveToCounter`     | Pick an object from the stove (via a pot or pan) and place it on (the plate on) the counter.              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_pnp.py) |


### Opening and closing doors

| Task              | Description                                                                                          | Class File                                                                                                           |
|:------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| `OpenSingleDoor`  | Open a microwave door or a cabinet with a single door.                                               | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_doors.py) |
| `CloseSingleDoor` | Close a microwave door or a cabinet with a single door.                                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_doors.py) |
| `OpenDoubleDoor`  | Open a cabinet with two opposite-facing doors.                                                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_doors.py) |
| `CloseDoubleDoor` | Close a cabinet with two opposite-facing doors.                                                      | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_doors.py) |


### Opening and closing drawers

| Task          | Description                                                                                          | Class File                                                                                                            |
|:--------------|:-----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| `OpenDrawer`  | Open a drawer.                                                                                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_drawer.py) |
| `CloseDrawer` | Close a drawer.                                                                                      | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_drawer.py) |


### Turning levers

| Task                | Description                                                                                          | Class File                                                                                                          |
|:--------------------|:-----------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| `TurnOnSinkFaucet`  | Turn on the sink faucet to begin the flow of water.                                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_sink.py) |
| `TurnOffSinkFaucet` | Turn off the sink faucet to begin the flow of water.                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_sink.py) |
| `TurnSinkSpout`     | Turn the sink spout.                                                                                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_sink.py) |


### Twisting knobs

| Task           | Description                                                                                          | Class File                                                                                                           |
|:---------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| `TurnOnStove`  | Turn on a specified stove burner by twisting the respective stove knob.                              | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_stove.py) |
| `TurnOffStove` | Turn off a specified stove burner by twisting the respective stove knob.                             | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_stove.py) |


### Insertion

| Task             | Description                                                                                          | Class File                                                                                                            |
|:-----------------|:-----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| `CoffeeSetupMug` | Pick the mug from the counter and insert it onto the coffee machine mug holder area.                 | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_coffee.py) |
| `CoffeeServeMug` | Remove the mug from the coffee machine mug holder and place it on the counter.                       | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_coffee.py) |


### Pressing buttons

| Task                | Description                                                                                          | Class File                                                                                                               |
|:--------------------|:-----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| `CoffeePressButton` | Press the button on the coffee machine to pour coffee into the mug.                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_coffee.py)    |
| `TurnOnMicrowave`   | Turn on the microwave by pressing the start button.                                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_microwave.py) |
| `TurnOffMicrowave`  | Turn off the microwave by pressing the stop button.                                                  | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_microwave.py) |


### Navigation

| Task              | Description                                                                                          | Class File                                                                                                              |
|:------------------|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| `NavigateKitchen` | Navigate to a specified appliance in the kitchen.                                                    | [Source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/single_stage/kitchen_navigate.py) |