# G1 unseen composite tasks: reference and historical archive

The official benchmark reference is now [Unitree G1 Tasks](g1/tasks.md), including
the [ten unseen composite tasks](g1/tasks.md#unseen-composite-tasks), their English
instructions, implemented success conditions, and 180-second evaluation horizons.
The authoritative 50-task list is [`robocasa_eval/tasks50.json`](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa_eval/tasks50.json).

The ten unseen environments selected on **2026-10-08** are the ten entries marked
`unseen` in that manifest. They are existing benchmark environments. Their
inclusion in evaluation does not mean that demonstrations for them are part of
the 40-task training-data release. `JuiceFruitReamer` is outside this benchmark.

## Archived proposal — 2026-09-28, superseded

**The material below is a historical design record, not the current task list,
implementation backlog, evaluation protocol, or dataset inventory.** Its proposed
nine `G1...` environments and `PreRinseStation` were superseded by the October 8
selection. They must not be presented as official tasks awaiting development.
The historical 2,286-episode cleaning snapshot is also distinct from the released
40 × 50 = 2,000 demonstrations. Historical recommendations and source descriptions
below may differ from the current code; the linked [Tasks page](g1/tasks.md) is
the current source-backed reference. In particular, the archived description of
`LoadDishwasher` must not override its current door-closure requirement.

以下保留 2026-09-28 的中文历史方案；其中“推荐”“待实现”“优先实现”等表述仅描述当时的设计，不是本轮交付要求。

核对日期：2026-09-28。历史状态：任务设计，当时尚未实现新增环境或进行 G1 可执行性测试。

### 设计依据与范围

当前 NAS 已清洗数据包含 **25 个原子任务、15 个组合任务，共 2,286 条成功 episode**。
数量来自 NAS `cleaned` 数据集的 `meta/info.json`，动作和对象语义通过 `meta/tasks.jsonl`
及当前工作区任务源码核对。原子任务共 1,486 条，组合任务共 800 条。
这里描述的是可作为训练候选的数据；是否全部进入了某个模型的训练，需要以该模型的训练 manifest 为准。

“组合未见”指：组成任务的操作在训练数据中有覆盖，但完整任务的动作依赖和目标组合没有作为训练任务出现。
该定义相对于本项目确定的训练集，不代表这些任务从未进入基础模型的预训练。
现有 episode 数量也不等同于模型已掌握相应技能。

本方案采用 **1 个现有任务 + 9 个定制任务**。`PreRinseStation` 是已有环境；下文所有 `G1...`
标识都是拟定名称，当前没有注册，不能直接用于 `--environment`。
现有任务也需要先验证 SonicG1 的布局、可达性和对象采样。

基线评测应使用训练中出现过的对象实例、尺寸、fixture 样式和布局，重新采样 episode 初始状态。
本次核对确认了任务及指令层面的覆盖，尚未逐条比较 HDF5 中的资产实例交集。
实施时应从 raw 的场景元数据/XML 提取实例清单，冻结合法组合；如果实例交集为空，应补充训练覆盖或单独标记为对象泛化测试。
还应检查训练轨迹是否偶然包含整个候选流程；只有通过这一检查，才能把候选正式标记为组合未见。

官方参考：[原子任务目录](https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html)、
[组合任务目录](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html)。
任务的具体操作以本地实际执行的源码和已采集数据为准；网页上的分类标签不等于完整成功判据。

### 推荐的 10 个任务

下表的难度是设计预估，尚未通过 rollout 测量。

| 编号 | 任务及拟定环境名 | 动作流程 | 已采集任务提供的覆盖 | 主要未见组合 / 预估难度 |
| --- | --- | --- | --- | --- |
| 1 | 调整出水口并叠碗 `G1PositionSpoutAndStackBowls` | 出水口转向左侧 → 两只碗在水槽内叠放 | `TurnSinkSpout` + `StackBowlsInSink` | 调整工作空间后接续容器叠放；中等 |
| 2 | 双槽餐具分拣 `PreRinseStation`，现有任务 | 马克杯放指定一侧水槽，碗放另一侧 | `PickPlaceCounterToSink` | 对象与左右目标绑定；基础空间组合 |
| 3 | 取量杯并关抽屉 `G1SelectCupAndCloseDrawer` | 开抽屉 → 取指定大小量杯到台面 → 关抽屉 | `OpenDrawer` + `ChooseMeasuringCup` + `DrawerCakeSort` 的关抽屉动作 | 尺寸选择后完成收尾；基础至中等 |
| 4 | 配料入搅拌机并合上机头 `G1LoadMixerAndCloseHead` | 面包、芝士分别入碗 → 合上机头 | `PickPlaceCounterToStandMixer` + `CloseStandMixerHead` | 两个配料都到位后才能结束；中等 |
| 5 | 清出台面并擦拭 `G1ClearCounterAndWipe` | 马克杯、碗入水槽 → 海绵擦清空区域 → 放下海绵 | `PickPlaceCounterToSink` + `WipeTable` | 清空区域是擦拭的前置条件；中等 |
| 6 | 蘑菇入锅、晃锅、盛碗 `G1LoadShakeAndServeFood` | 蘑菇从盘入锅 → 提锅晃动 → 锅放回炉灶 → 蘑菇盛入碗 | `PickPlaceCounterToStove` + `ShakePan` + `PickPlaceStoveToCounter` | 同一对象依次经过三个状态，切换食物与锅操作；较高 |
| 7 | 取可颂并复位小烤箱 `G1ServeCroissantAndResetToasterOven` | 开门 → 拉架 → 可颂放盘 → 推回架 → 关门 | `OpenToasterOvenDoor` + `SlideToasterOvenRack` + `ServeWarmCroissant` + `CloseToasterOvenDoor` | 取出食物后继续完成设备复位；较高 |
| 8 | 完整装载洗碗机 `G1FullDishwasherLoad` | 开门 → 拉上层架 → 放杯、碗 → 推架 → 关门 | `OpenDishwasher` + `SlideDishwasherRack` + `LoadDishwasher` + `CloseDishwasher` | 串联准备、装载和复位；较高 |
| 9 | 蛋糕出炉后装餐盒 `G1CoolAndPackCake` | 蛋糕从烤箱放盘 → 关烤箱 → 蛋糕放入已有配菜的餐盒 | `CoolBakedCake` + `PackDessert` | 原任务终点成为下一段起点，中间容器和最终容器不同；中等至较高 |
| 10 | 走到冰箱并打开抽屉 `G1NavigateAndOpenFridgeDrawer` | 导航到冰箱 → 开外门 → 拉开内部抽屉 | `NavigateKitchen` + `OpenFridge` + `OpenFridgeDrawer` | 导航结束后接续门与内部抽屉操作；较高 |

其中 2 是空间目标组合，其余主要测试顺序、前置条件或流程延长。报告结果时应保留这种区别。

### 各任务的 prompt、初始化与判据

以下自定义判据为实施规格，尚未修改现有任务的 `_check_success()`。
所有涉及搬放的最终状态均应检查释放物体、合理支撑及未掉落。
“历史事件”只记录已经发生的阶段；“终点状态”必须在任务结束时仍然满足。

#### 1. G1PositionSpoutAndStackBowls

**英文 prompt**：Turn the sink spout to the left, then stack both bowls in the sink.

- 初始化：沿用叠碗任务的两个碗和台面采样；出水口尚未朝左，水流关闭。选择出水口转向左侧后能为叠放让出空间的既有水槽样式，并在试跑时验证这一几何关系。
- 历史事件：出水口转向左侧后，再把两只碗搬入水槽并叠放。
- 终点状态：`get_handle_state()["spout_ori"] == "left"`；两只碗都在水槽内，任一碗稳定套在另一碗中；两手离开物体。
- 未见之处：将转动出水口和已有叠碗流程连接，要求完成叠放后仍保留指定出水口方向。
- 不需要开水流或模拟清洗效果；转动出水口和开关水流是不同的操作。
- 实现参考：[kitchen_sink.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_sink.py) 的 `TurnSinkSpout` 和 [stack_bowls.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/washing_dishes/stack_bowls.py)。

#### 2. PreRinseStation

**英文 prompt 示例**：Place the mug in the left basin and the bowl in the right basin of the sink.

- 初始化：使用双槽水槽；马克杯和碗在水槽旁台面。左右分配均衡随机化，分配结果写入 episode 元数据和 prompt。
- 终点状态：两个对象各自在指定槽内，双手离开；放置顺序不限。
- 未见之处：训练中的单目标入水槽，组合为两个对象与两个空间目标的绑定。
- 现有实现通过 `sink.get_obj_basin_loc()` 和两个 `gripper_obj_far()` 判断成功；不要求开水龙头。
- 官方环境默认采样 `receptacle` / `stackable` 广泛类别。本方案建议限制为已采集的马克杯、碗及其实例，这是评测采样配置约束，原始默认配置不会自动满足。
- 必须遵循该任务的 `EXCLUDE_STYLES`，并检查合法双槽样式是否出现在训练中；新样式结果另报。
- 实现参考：[pre_rinse_station.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/washing_dishes/pre_rinse_station.py)。

#### 3. G1SelectCupAndCloseDrawer

**英文 prompt 模板**：Open the {left/right} drawer, take out the {larger/smaller} measuring cup and place it on the counter, then close the drawer.

- 初始化：沿用 `ChooseMeasuringCup` 的大小量杯、目标抽屉及左右站位；抽屉关闭。
- 历史事件：开抽屉 → 目标量杯释放到台面 → 关同一抽屉。
- 终点状态：正确量杯在台面；另一量杯仍在抽屉；抽屉关闭。
- 未见之处：已有选杯任务在量杯到台面后即可成功，这里增加依赖前段完成的收尾操作。
- 关抽屉的训练覆盖来自 `DrawerCakeSort`，当前清单没有独立的 `CloseDrawer` 数据集。
- 实现参考：[choose_measuring_cup.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/measuring_ingredients/choose_measuring_cup.py) 和 [kitchen_drawer.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_drawer.py)。

#### 4. G1LoadMixerAndCloseHead

**英文 prompt**：Place the bread and cheese in the stand mixer bowl, then close the mixer head.

- 初始化：机头已打开，机器未启动；面包和芝士位于附近台面。容器空间应能同时容纳两个既有尺寸的物体。
- 历史事件：两个配料都已入碗并松手，随后合上机头。配料之间没有规定先后。
- 终点状态：两件配料都通过 `check_item_in_bowl()`；机头关闭，参照现有 `CloseStandMixerHead` 的归一化位置 `< 0.01`。
- 未见之处：单配料入碗与单独合机头组合为多配料准备流程。
- 本任务目标到准备完毕为止；不包含尚未采集的旋钮启动、搅拌结果或烹饪状态。
- 实现参考：[kitchen_stand_mixer.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_stand_mixer.py) 和 `PickPlaceCounterToStandMixer`。

#### 5. G1ClearCounterAndWipe

**英文 prompt**：Move the mug and bowl from the counter into the sink, then use the sponge to wipe the cleared counter and put the sponge down.

- 初始化：马克杯、碗和海绵在水槽旁同一片可达台面；擦拭区域由两件餐具的初始区域确定，大小保持在已有擦拭任务的尺度内。
- 历史事件：两件餐具都进入水槽并释放后，才累计该区域内的海绵擦拭记录。
- 终点状态：餐具仍在水槽；擦拭达到阈值；海绵放回台面并释放。
- 初始采用 `WipeTable` 的尺度：握住海绵并接触台面时，记录至少 5 个相隔至少 2 cm 的接触位置，接触位置的包围范围对角线至少 10 cm。只累计目标清空区域内的位置。
- 未见之处：增加“先清出工作区域，再使用工具”的因果依赖。提前在其他位置擦过不应直接满足本任务。
- 实现参考：[wipe_table.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/sanitizing_surface/wipe_table.py)。

#### 6. G1LoadShakeAndServeFood

**英文 prompt**：Place the mushroom from the plate into the pan. Lift and gently shake the pan, put it back on the stove, then place the mushroom in the bowl.

- 初始化：蘑菇在盘内；空锅在炉灶；目标碗在相邻台面。蘑菇同时出现于上述三个来源任务的已采集指令。
- 历史事件：蘑菇入锅 → 握锅并提起晃动 → 锅放回炉灶 → 蘑菇盛入碗。不要把蘑菇直接从盘搬到碗计为全任务成功。
- 晃动基线参考 `ShakePan`：食物在锅内时，锅的位置变化超过 1 mm 的控制步累计达到 25 次。新增组合应排除初始化抖动，另记录真实握锅、提锅证据；具体离地阈值在试跑后冻结。
- 终点状态：蘑菇在目标碗，碗在台面，锅在原炉灶位置，手已松开。
- 蘑菇已经盛出，因此终点不能再要求“蘑菇仍在锅里”；该条件属于前序事件。
- 使用与来源任务一致的控制频率；不额外要求火候或熟度。当前 `ShakePan` 在初始化时已设置炉灶状态，不能据此认定模型会开火。
- 实现参考：[shake_pan.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/sauteing_vegetables/shake_pan.py) 及两个 stove pick-place 类。

#### 7. G1ServeCroissantAndResetToasterOven

**英文 prompt**：Open the toaster oven, pull out the rack, place the croissant on the plate, push the rack fully back in, and close the door.

- 初始化：小烤箱门关闭、目标架收回；可颂位于指定架；盘子在旁边台面。使用可颂任务与滑架任务共同支持的架/托盘样式。
- 历史事件：开门 → 拉出目标架 → 可颂到盘并释放 → 推回同一架 → 关门。
- 终点状态：可颂在盘内；盘子在台面；目标架位置 `<= 0.05`；门关闭。
- 拉出事件参照 `SlideToasterOvenRack` 的位置 `>= 0.95`。若样式有上下架，prompt 和判据必须指向同一架。
- 未见之处：已有 `ServeWarmCroissant` 的终点主要检查可颂在盘内；这里追加“推架、关门”并检查完整链条。
- 实现参考：[serve_warm_croissant.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/toasting_bread/serve_warm_croissant.py)、[kitchen_toaster_oven.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_toaster_oven.py)、[kitchen_doors.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_doors.py)。

#### 8. G1FullDishwasherLoad

**英文 prompt**：Open the dishwasher, pull out the top rack, load the cup and bowl from the counter, push the rack fully in, and close the door.

- 初始化：洗碗机关闭、上层架收回；杯和碗在台面。保留已采集 `LoadDishwasher` 的对象尺寸和已向左移动 15 cm 的采样区域。
- 历史事件：开门 → 上层架拉出 → 杯和碗装好并释放 → 推回架 → 关门。杯与碗之间不规定先后。
- 终点状态：杯和碗仍由目标架支撑；架位置 `<= 0.05`；门关闭；物体已释放。不能只以门关闭作为成功。
- 拉出事件参照 `SlideDishwasherRack` 的位置 `>= 0.95`；门关闭沿用已采集任务的判据。
- 未见之处：已有 `LoadDishwasher` 在 reset 中预先打开门并拉出架，这里要求策略从关闭状态完成整个流程。
- 重点试跑：G1 站位是否避开下翻门、装有餐具的架能否顺畅推入，餐具是否在推架后仍稳定。
- 实现参考：[prepare_dishwasher.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/loading_dishwasher/prepare_dishwasher.py)、[load_dishwasher.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/loading_dishwasher/load_dishwasher.py)、`SlideDishwasherRack`。

#### 9. G1CoolAndPackCake

**英文 prompt 模板**：Take the cake out of the oven and place it on the plate. Close the oven door, then put the cake in the lunch box that contains the {hot dog/kebab skewer}.

- 初始化：烤箱已打开，蛋糕在可达烤盘；附近台面有空盘和一个已有配菜的餐盒。烤箱架位置应允许直接关闭，避免额外引入大型烤箱推架操作。
- 蛋糕资产和餐盒容量需兼容两个来源任务。已采集 `PackDessert` 确实包含“cake + hot dog”和“cake + kebab skewer”的指令。
- 历史事件：蛋糕在盘内稳定放下并松手 → 烤箱关门 → 蛋糕从盘进入餐盒。
- 终点状态：烤箱关闭；蛋糕和原配菜都在餐盒；餐盒在台面；蛋糕已释放。
- “蛋糕曾在盘中”应记录为历史事件，不能把 `CoolBakedCake` 与 `PackDessert` 的终点判据直接做 AND。
- 基线使用一个餐盒，匹配当前 `PackDessert` 的场景结构。增加两个不同配菜的餐盒可作为后续目标选择扩展，不能宣称已有数据已经包含这一干扰项配置。
- 名称中的冷却沿用任务语义，不引入未建模的温度或冷却时长判定。
- 实现参考：[cool_baked_cake.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/composite/baking/cool_baked_cake.py) 和 `PackDessert`。

#### 10. G1NavigateAndOpenFridgeDrawer

**英文 prompt**：Go to the fridge, open its door, and fully open the inner drawer.

- 初始化：机器人从另一个 fixture 附近出发，冰箱外门和内部目标抽屉关闭；出发点与冰箱之间需要真实移动，不能 reset 在已经满足导航目标的位置。
- 历史事件：到达冰箱操作位 → 外门打开到足以避让抽屉 → 拉开指定内部抽屉。
- 到达判据沿用当前 `NavigateKitchen`：G1 骨盆/基座相对目标基座位姿的平面误差 `<= 0.25 m`，朝向余弦相似度 `>= 0.98`。25 cm 指目标站位误差，不是机器人到冰箱表面的距离。
- 终点状态：相关外门保持打开；目标抽屉达到现有 `OpenFridgeDrawer` 的 `th=0.70`。
- 导航达标是历史事件，后续操作时允许机器人为开门调整站位。
- 未见之处：现有 `OpenFridgeDrawer` 在 reset 时已完全打开冰箱门；这里把导航、开外门与内部抽屉操作连接起来。
- 仅选择存在相应内部抽屉的冰箱样式，遵循现有 `EXCLUDE_STYLES`；基础版本不搬运物体，也不引入持物行走。
- 实现参考：[kitchen_navigate.py](https://github.com/Xucxname/robocasa/blob/agent/sonic-g1-followups/robocasa/environments/kitchen/atomic/kitchen_navigate.py)、`OpenFridge`、`OpenFridgeDrawer`。

### 评测与实现建议

1. **冻结训练集与测试定义。** 用最终训练 manifest 确认上述来源数据确实进入训练。10 个完整组合任务的遥操作示范、策略轨迹及其自动切片均不加入该训练版本；新对象、新布局结果单独报告。
2. **先验证场景能执行。** 每任务做少量独立工程试跑，检查 G1 够得着、门和架不碰撞、物体容纳关系合理。工程种子和正式评测种子分开，冻结规则后再评分。
3. **采用阶段事件加终点状态。** 顺序由明确的依赖关系决定；没有规定顺序的两个搬放动作允许交换。事件在每个控制步更新一次，不能因为多次调用成功检查而重复累计。
4. **同时记录两种成功指标。** 记录首次满足完整任务判据的时间；另外要求全部终点条件连续保持建议 1 秒，报告稳定成功率。保持期可发送保持动作，不继续发出新的任务操作。试跑后固定这个时长。
5. **从每任务 30 次正式评测开始。** 10 任务共 300 次，按共同支持的已见布局分层分配新种子。报告每任务完整成功率、宏平均、各阶段到达率、时长和失败阶段；30 次只是初测，比较接近的模型时再扩充样本并报告置信区间。
6. **固定时间预算。** 建议根据独立可行性试跑的完成时长确定每任务上限并提前冻结；不要在看到某个模型失败后临时延长。不同策略使用相同种子、控制频率和上限。
7. **区分 replay 与策略评测。** raw 状态 replay 用于验证场景、标注和阶段判据；动作 replay 还受控制和物理差异影响。模型能力必须通过从初态开始的闭环 policy rollout 评估。
8. **沿用正常观测。** 策略输入使用任务 instruction、当前头部/左右腕 RGB、模型支持的本体状态，以及带时间戳的动作历史/状态。阶段真值、目标坐标和仿真器成功信号只用于评估器，不作为额外策略提示。

若继续沿用“首次成功截断”的数据处理流程，截断点必须是**完整组合任务**首次满足全部条件并通过规定保持期之后；任何中间阶段成功都不能提前结束整条任务。

优先实现 **3、5、6、8、10**：分别检查目标选择后收尾、工具使用前置条件、动态锅操作、较长装载流程和移动操作衔接。
其余任务用于补足空间调整后叠放、双目标空间绑定、多配料、小烤箱复位及中间容器转换。

### 未纳入本轮的候选及原因

- `MakeCheesecakeFilling` 还要求旋转速度旋钮启动机器，超出了当前“放入碗 + 合机头”的已采集范围。
- `ClearReceptaclesForCleaning`、`RinseBowls` 等要求开水；`TurnSinkSpout` 的左右转动出水口不能替代开关水流。
- `CheeseMixing` 要求用锅铲搅拌，和 `ShakePan` / `TiltPan` 不是同一操作。
- `CountertopCleanup`、`SortingCleanup` 等引入未核实覆盖的橱柜搬放/关门。不能只根据名称把它们当成已有擦台面技能的组合。
- `CoolBakedCookies` 与已有 `CoolBakedCake` 的动作组合可能高度重合，更适合作为对象泛化对照。
- “面包放砧板后再把芝士放在面包上”可能与 `BreadAndCheese` 中的实际示范重合；仅新增任务名不足以确立组合未见，因此本轮未选入。
- “大型烤箱放两种食物后关门”需要检查伸出的架能否收回；现有 `PickPlaceCounterToOven` 初始化会拉架，而已采集的 `SlideToasterOvenRack` 操作的是小烤箱。暂不把两者视为同一个已覆盖操作。
- `PrepareDishwasher` 的“开门 + 拉架”适合用作任务 8 的短流程诊断，不另外占用本轮 10 个名额。

### 已采集清洗集快照

NAS 原始成功数据根目录：
`/volume1/unitree_ly_robocasa_data/datasets/raw/accepted/{atomic_tasks,composite_tasks}`。

本次数量与指令核对目录：
`/volume1/unitree_ly_robocasa_data/datasets/lerobot/{atomic_tasks,composite_tasks}/cleaned/robocasa_<Task>_g1_3cam/meta/`。

当时新加入的 CloseOven 和 NavigateKitchen 状态还曾引用本地发布报告
`datasets/log/new_tasks_release_20260928/report.md`；该历史报告不随本指南分发。
清洗集已在 NAS 并不代表该集合已全部发布到 Hugging Face。

#### 原子任务：25 个，1,486 条

| 任务 | 成功 episode |
| --- | ---: |
| CheesyBread | 50 |
| CloseDishwasher | 54 |
| CloseElectricKettleLid | 55 |
| CloseFridgeDrawer | 54 |
| CloseFridge | 60 |
| CloseOven | 54 |
| CloseStandMixerHead | 54 |
| CloseToasterOvenDoor | 71 |
| CoffeeServeMug | 54 |
| NavigateKitchen | 51 |
| OpenDishwasher | 55 |
| OpenDrawer | 78 |
| OpenFridgeDrawer | 56 |
| OpenFridge | 54 |
| OpenToasterOvenDoor | 53 |
| PackDessert | 52 |
| PickPlaceCounterToOven | 56 |
| PickPlaceCounterToSink | 71 |
| PickPlaceCounterToStandMixer | 62 |
| PickPlaceCounterToStove | 86 |
| PickPlaceStoveToCounter | 78 |
| SlideDishwasherRack | 69 |
| SlideToasterOvenRack | 54 |
| TurnOnToaster | 51 |
| TurnSinkSpout | 54 |

#### 组合任务：15 个，800 条

| 任务 | 成功 episode |
| --- | ---: |
| BoilCorn | 50 |
| BreadAndCheese | 52 |
| ChooseMeasuringCup | 53 |
| CleanBlenderJug | 57 |
| CoolBakedCake | 53 |
| DrawerCakeSort | 54 |
| DryDishes | 51 |
| DumpLeftovers | 62 |
| LoadDishwasher | 52 |
| PortionPeaches | 51 |
| ServeWarmCroissant | 59 |
| ShakePan | 53 |
| StackBowlsInSink | 50 |
| TiltPan | 52 |
| WipeTable | 51 |

注意：`CheesyBread` 和 `PackDessert` 在当前官方目录及本地实现中属于原子任务，名称听起来像组合任务也不应重新分类。
