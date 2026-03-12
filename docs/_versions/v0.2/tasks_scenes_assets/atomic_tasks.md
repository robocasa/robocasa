## Atomic Tasks

The current release includes 25 atomic tasks for systematically training policies across eight foundational skill families. The breakdown of these tasks across skill families is as follows:

### Pick and Place

| Task                          | Description                                                                                               | Class File                                                                                                         |
|:------------------------------|:----------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| `PnPCounterToCab`        | Pick an object from the counter and place it inside the cabinet. The cabinet is already open.             | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L24) |
| `PnPCabToCounter`        | Pick an object from the cabinet and place it on the counter. The cabinet is already open.                 | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L142) |
| `PnPCounterToSink`       | Pick an object from the counter and place it in the sink.                                                 | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L258) |
| `PnPSinkToCounter`       | Pick an object from the sink and place it on the counter area next to the sink.                           | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L369) |
| `PnPCounterToMicrowave`  | Pick an object from the counter and place it inside the microwave. The microwave door is already open.    | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L481) |
| `PnPMicrowaveToCounter`  | Pick an object from inside the microwave and place it on the counter. The microwave door is already open. | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L606) |
| `PnPCounterToStove`      | Pick an object from the counter and place it in a pan or pot on the stove.                                | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L728) |
| `PnPStoveToCounter`      | Pick an object from the stove (via a pot or pan) and place it on (the plate on) the counter.              | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_pnp.py#L819) |


### Opening and closing doors

| Task              | Description                                                                                          | Class File                                                                                                           |
|:------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| `OpenSingleDoor`  | Open a microwave door or a cabinet with a single door.                                               | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_doors.py#L139) |
| `CloseSingleDoor` | Close a microwave door or a cabinet with a single door.                                              | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_doors.py#L154) |
| `OpenDoubleDoor`  | Open a cabinet with two opposite-facing doors.                                                       | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_doors.py#L144) |
| `CloseDoubleDoor` | Close a cabinet with two opposite-facing doors.                                                      | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_doors.py#L159) |


### Opening and closing drawers

| Task          | Description                                                                                          | Class File                                                                                                            |
|:--------------|:-----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| `OpenDrawer`  | Open a drawer.                                                                                       | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_drawer.py#L185) |
| `CloseDrawer` | Close a drawer.                                                                                      | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_drawer.py#L239) |


### Turning levers

| Task                | Description                                                                                          | Class File                                                                                                          |
|:--------------------|:-----------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| `TurnOnSinkFaucet`  | Turn on the sink faucet to begin the flow of water.                                                  | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_sink.py#L110) |
| `TurnOffSinkFaucet` | Turn off the sink faucet to begin the flow of water.                                                 | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_sink.py#L115) |
| `TurnSinkSpout`     | Turn the sink spout.                                                                                 | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_sink.py#L120) |


### Twisting knobs

| Task           | Description                                                                                          | Class File                                                                                                           |
|:---------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| `TurnOnStove`  | Turn on a specified stove burner by twisting the respective stove knob.                              | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_stove.py#L125) |
| `TurnOffStove` | Turn off a specified stove burner by twisting the respective stove knob.                             | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_stove.py#L130) |


### Insertion

| Task             | Description                                                                                          | Class File                                                                                                            |
|:-----------------|:-----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| `CoffeeSetupMug` | Pick the mug from the counter and insert it onto the coffee machine mug holder area.                 | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_coffee.py#L109) |
| `CoffeeServeMug` | Remove the mug from the coffee machine mug holder and place it on the counter.                       | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_coffee.py#L118) |


### Pressing buttons

| Task                | Description                                                                                          | Class File                                                                                                               |
|:--------------------|:-----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| `CoffeePressButton` | Press the button on the coffee machine to pour coffee into the mug.                                  | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_coffee.py#L127)    |
| `TurnOnMicrowave`   | Turn on the microwave by pressing the start button.                                                  | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_microwave.py#L83) |
| `TurnOffMicrowave`  | Turn off the microwave by pressing the stop button.                                                  | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_microwave.py#L88) |


### Navigation

| Task              | Description                                                                                          | Class File                                                                                                              |
|:------------------|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| `NavigateKitchen` | Navigate to a specified appliance in the kitchen.                                                    | [Source](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/environments/kitchen/single_stage/kitchen_navigate.py#L4) |