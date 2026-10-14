"""Manual subtask-label overrides for flagged demos a human has reviewed.

The flagged-demo workflow: the auto-labeler flags a demo it can't confidently
segment (a genuine anomaly), a person watches the video, and specifies the
correct segmentation by hand here. The CLI applies the override in place of the
auto-resolve for that (env_name, demo_key) and records the demo as manually
labeled, so it no longer sits in the "investigate" queue.

Each override is an ordered list of contiguous segments
    (end_frame_exclusive, subtask_name, stage, lang)
-- exactly what BaseSubtaskLabeler.build_result() consumes. Segment i covers
[previous_end, end); the final segment should be the terminal "done".
"""

from robocasa.utils.subtask_labeling.base import (  # noqa: F401
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# AddIceCubes demo_82 (== LeRobot episode_000091, T=741): a multi-grab the
# auto-labeler flagged. Reviewed: cube1 is placed individually (lift 147,
# release 230), then the robot returns and scoops the two REMAINING cubes
# together (lift 474, released ~688). Labeled as two pick/place subtasks, the
# second one covering both remaining cubes.
_ADD_ICE_CUBES_DEMO82 = [
    (147, "PickPlaceCounterToBlender", STAGE_PICK, "pick up an ice cube from the bowl"),
    (235, "PickPlaceCounterToBlender", STAGE_PLACE, "place the ice cube in the blender"),
    (474, "PickPlaceCounterToBlender", STAGE_PICK, "pick up the remaining ice cubes from the bowl"),
    (689, "PickPlaceCounterToBlender", STAGE_PLACE, "place the ice cubes in the blender"),
    (741, DONE, STAGE_DONE, "task complete"),
]

# RestockBowls demo_2 (== LeRobot episode_000001, T=1543): the auto-labeler
# detected the second bowl's departure late (frame 1088), so the pick stage ran
# long and the place stage was only ~20 frames. Reviewed against the video (77s
# @20fps): the second bowl leaves the counter ~48s = frame 960, so the pick ->
# place boundary is set there. All other boundaries match the auto-resolve.
_RESTOCK_BOWLS_DEMO2 = [
    (468, "OpenCabinet", STAGE_EXECUTE, "open the cabinet"),
    (671, "PickPlaceCounterToCabinet", STAGE_PICK, "pick up a bowl from the counter"),
    (822, "PickPlaceCounterToCabinet", STAGE_PLACE, "place the bowl in the cabinet"),
    (960, "PickPlaceCounterToCabinet", STAGE_PICK, "pick up the bowl from the counter"),
    (1110, "PickPlaceCounterToCabinet", STAGE_PLACE, "place the bowl in the cabinet"),
    (1525, "CloseCabinet", STAGE_EXECUTE, "close the cabinet"),
    (1543, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_12 (pretrain, == LeRobot episode T=1453): an anomaly
# the auto-labeler silently mis-segments. The labeler only tracks the three
# "middle" bottles, but in this demo the robot rearranges FOUR glass bottles --
# including the two end-cluster bottles (bottle_glass1, bottle_glass2) -- so the
# auto-resolve sees only 2 of the 4 pick/places. Reviewed against the video and
# hand-segmented by the user. Boundaries are the gripper finger-width grasp
# (close) / release (open) frames: closes 126/428/665/1022, opens 304/529/832/
# 1401, matching the four glass bottles' contact-grasp runs.
_RECYCLE_BOTTLES_DEMO12 = [
    (126,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (304,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle in its own cluster to the right or left"),
    (428,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up another glass bottle from the middle"),
    (529,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle in the glass bottle cluster"),
    (665,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up another glass bottle from the middle"),
    (832,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle in the glass bottle cluster"),
    (1022, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up another glass bottle from the middle"),
    (1401, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle in the glass bottle cluster"),
    (1453, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_49 (pretrain, == LeRobot episode T=593): another
# silent anomaly. The robot relocates bottle_plastic_middle (a tracked middle,
# contact 139-212) and then bottle_glass2 -- one of the two END-CLUSTER glass
# bottles (maxdisp 0.45, contact 334-519), NOT a tracked middle. The labeler
# only watches the three middle bottles, so it sees a single moving bottle and
# mis-segments as a one-bottle plastic relocation. Reviewed and hand-segmented
# by the user. Boundaries are the gripper finger-width close/open frames: closes
# 139/331, opens 214/525.
_RECYCLE_BOTTLES_DEMO49 = [
    (139, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up the plastic bottle from the middle"),
    (214, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle to the right side to form a cluster"),
    (331, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up the glass bottle from the middle"),
    (525, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle to the left side to form a cluster"),
    (593, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_21 (pretrain, == LeRobot episode T=1900): a six-bottle
# rearrange the auto-labeler flags (relocates more than its 1-5 cap). The robot
# sorts ALL bottles into two fresh clusters -- glass on one side, plastic on the
# other -- moving four glass (glass_middle, mystery, glass2, glass1) and two
# plastic (plastic_middle, plastic1), in that departure order. Hand-segmented by
# the user; place wording is the user's (the first bottle of a type starts its own
# cluster, the second forms the pair, the rest join). Pick ends at the contact
# grasp; place ends at the gripper-open (release) frame -- the auto-resolve's
# motion set-down fires early on these short, slow carries, truncating the place.
_RECYCLE_BOTTLES_DEMO21 = [
    (114,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (226,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle in its own cluster to the right or left"),
    (497,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (632,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottle to form a cluster"),
    (796,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (953,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle in its own cluster to the right"),
    (1094, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (1222, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle next to the other plastic bottle to form a cluster"),
    (1375, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (1569, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottles"),
    (1738, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (1838, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottles"),
    (1900, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_28 (pretrain, == LeRobot episode T=1187): an
# end-cluster rearrange. Four bottles are relocated in departure order --
# plastic1, mystery(glass), glass2, plastic_middle -- forming two fresh clusters
# (plastic right, glass left). The auto-resolve segments this correctly (gripper-
# open boundaries below); only the first-placed bottle of each type gets the
# user's "to the <side> side to form a cluster" wording, the rest join an existing
# cluster. Boundaries are the auto-resolve's grasp / gripper-open release frames.
_RECYCLE_BOTTLES_DEMO28 = [
    (205,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (359,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle to the right side to form a cluster"),
    (458,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (562,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle to the left side to form a cluster"),
    (752,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (854,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottles"),
    (998,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (1152, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle next to the other plastic bottles"),
    (1187, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_64 (pretrain, == LeRobot episode T=1571): same
# end-cluster rearrange pattern. Four bottles in departure order -- plastic1,
# glass2, mystery(glass), plastic_middle -- form two fresh clusters (plastic
# right, glass left). First-placed bottle of each type gets the user's "to the
# <side> side to form a cluster" wording; the rest join. Boundaries are the
# auto-resolve's grasp / gripper-open release frames.
_RECYCLE_BOTTLES_DEMO64 = [
    (125,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (300,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle to the right side to form a cluster"),
    (492,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (731,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle to the left side to form a cluster"),
    (918,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a glass bottle from the middle"),
    (1029, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottles"),
    (1209, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (1488, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle next to the other plastic bottles"),
    (1571, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_99 (pretrain, == LeRobot episode T=2961): a large,
# spread-out layout (navigation between the middle and each cluster). The robot
# relocates two end-cluster glass bottles and the plastic middle. The auto-resolve
# segments it correctly; the only change is the user's wording for the plastic
# pick (seg 7), which is grabbed out of the glass cluster rather than the middle.
# All boundaries are the auto-resolve's grasp / contact-end release / nav frames.
_RECYCLE_BOTTLES_DEMO99 = [
    (312,  "PickPlaceCounterToCounter", STAGE_PICK,     "pick up a glass bottle from the middle"),
    (617,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the cluster of glass bottles"),
    (822,  "PickPlaceCounterToCounter", STAGE_PLACE,    "place the glass bottle next to the other glass bottles"),
    (1098, "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the middle"),
    (1291, "PickPlaceCounterToCounter", STAGE_PICK,     "pick up a glass bottle from the middle"),
    (1655, "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the cluster of glass bottles"),
    (1870, "PickPlaceCounterToCounter", STAGE_PLACE,    "place the glass bottle next to the other glass bottles"),
    (2439, "PickPlaceCounterToCounter", STAGE_PICK,     "pick up the plastic bottle in the cluster of glass bottles"),
    (2707, "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the cluster of plastic bottles"),
    (2914, "PickPlaceCounterToCounter", STAGE_PLACE,    "place the plastic bottle next to the other plastic bottles"),
    (2961, DONE, STAGE_DONE, "task complete"),
]

# RecycleBottlesByType demo_268 (TARGET split, == LeRobot episode T=786): a normal
# three-bottle demo, but the contact-grasp check badly mis-fired -- glass_middle
# registered zero contact and the third bottle (plastic_middle) only two single-
# frame blips (704, 708) -- so the contact-end release landed at 716 and collapsed
# the third place to 12 frames. The finger width shows the bottle is held until the
# gripper fully opens at ~770. Only the third place end is corrected; every other
# boundary and all wording match the auto-resolve. (Target-split key; the T-guard
# in label_subtasks keeps it off the pretrain demo_268, which does not exist.)
_RECYCLE_BOTTLES_DEMO268 = [
    (175, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up the glass bottle from the middle"),
    (311, "PickPlaceCounterToCounter", STAGE_PLACE, "place the glass bottle next to the other glass bottles"),
    (471, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (551, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle next to the other plastic bottles"),
    (704, "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a plastic bottle from the middle"),
    (770, "PickPlaceCounterToCounter", STAGE_PLACE, "place the plastic bottle next to the other plastic bottles"),
    (786, DONE, STAGE_DONE, "task complete"),
]

# PackIdenticalLunches anomalies (all TARGET split; T-guard keeps them off the
# same-key pretrain demos, which have different lengths). The user watched each
# and described what happened.
#
# demo_5 (T=884) and demo_483 (T=671): MULTI-GRAB -- the last vegetable and meat
# are picked up in a single grasp and placed together in the empty tupperware, so
# the final trip is one pick / one place covering both items.
_PACK_LUNCH_DEMO5 = [
    (67,   "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a corn from the fridge"),
    (181,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to a tupperware on the counter"),
    (215,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the corn in a tupperware"),
    (306,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (404,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a fish from the fridge"),
    (493,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with the corn"),
    (556,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the fish in the same tupperware as the corn"),
    (644,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (729,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a corn and a fish from the fridge"),
    (804,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the empty tupperware on the counter"),
    (852,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the corn and fish in the empty tupperware"),
    (884,  DONE, STAGE_DONE, "task complete"),
]

_PACK_LUNCH_DEMO483 = [
    (52,   "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a garlic from the fridge"),
    (117,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to a tupperware on the counter"),
    (160,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the garlic in a tupperware"),
    (225,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (287,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a fish from the fridge"),
    (344,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with the garlic"),
    (391,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the fish in the same tupperware as the garlic"),
    (479,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (541,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a fish and a garlic from the fridge"),
    (599,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the empty tupperware on the counter"),
    (639,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the fish and garlic in the empty tupperware"),
    (671,  DONE, STAGE_DONE, "task complete"),
]

# demo_358 (T=1148): the first item (potato) never got a place subtask -- the
# carry nav ran right up to the release. Place inserted at the potato's set-down.
_PACK_LUNCH_DEMO358 = [
    (62,   "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a potato from the fridge"),
    (160,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to a tupperware on the counter"),
    (218,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the potato in a tupperware"),
    (277,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (334,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a fish from the fridge"),
    (416,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with the potato"),
    (464,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the fish in the same tupperware as the potato"),
    (561,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (637,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a fish from the fridge"),
    (734,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the empty tupperware on the counter"),
    (780,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the fish in the empty tupperware"),
    (887,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (964,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a potato from the fridge"),
    (1072, "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with only the fish in it"),
    (1121, "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the potato in the tupperware with only fish in it"),
    (1148, DONE, STAGE_DONE, "task complete"),
]

# demo_386 (T=968): the last place (steak) started too late -- the carry nav ate
# it. Place start pulled back ~30 frames so the lower-and-release is its own segment.
_PACK_LUNCH_DEMO386 = [
    (49,   "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a steak from the fridge"),
    (130,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to a tupperware on the counter"),
    (158,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the steak in a tupperware"),
    (221,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (278,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a garlic from the fridge"),
    (351,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with the steak"),
    (385,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the garlic in the same tupperware as the steak"),
    (471,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (534,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a garlic from the fridge"),
    (615,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the empty tupperware on the counter"),
    (648,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the garlic in the empty tupperware"),
    (734,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (812,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a steak from the fridge"),
    (890,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with only the garlic in it"),
    (931,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the steak in the tupperware with only garlic in it"),
    (968,  DONE, STAGE_DONE, "task complete"),
]

# demo_387 (T=1386): the first steak never got a pick subtask -- the robot drove
# back to the fridge and grasped in one motion. Pick carved out before the grasp.
_PACK_LUNCH_DEMO387 = [
    (93,   "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a mushroom from the fridge"),
    (178,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to a tupperware on the counter"),
    (235,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the mushroom in a tupperware"),
    (341,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (440,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a mushroom from the fridge"),
    (513,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the empty tupperware on the counter"),
    (573,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the mushroom in the empty tupperware"),
    (680,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (831,  "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a steak from the fridge"),
    (868,  "NavigateKitchen",           STAGE_NAVIGATE, "navigate to either tupperware with mushroom in it"),
    (938,  "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the steak in either tupperware with mushroom in it"),
    (1049, "NavigateKitchen",           STAGE_NAVIGATE, "navigate back to the fridge"),
    (1167, "PickPlaceFridgeToCounter", STAGE_PICK,     "pick up a steak from the fridge"),
    (1276, "NavigateKitchen",           STAGE_NAVIGATE, "navigate to the tupperware with only the mushroom in it"),
    (1356, "PickPlaceFridgeToCounter", STAGE_PLACE,    "place the steak in the tupperware with only mushroom in it"),
    (1386, DONE, STAGE_DONE, "task complete"),
]

# PortionHotDogs demo_45 (PRETRAIN, T=1017; T-guard keeps it off the same-key
# target demo). hotdog_bun1 is grabbed with a loose grasp (no contact) and a
# collision knocks it early, so the auto-resolve emitted a spurious 5-frame place
# for it. Rebuilt from the clean finger-width close/open + contact events: four
# pick/places (no nav in this demo), bun2/bun1 then sausage1/sausage2.
_PORTION_HOTDOGS_DEMO45 = [
    (154,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a bun from the bowl"),
    (265,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the bun on either plate"),
    (391,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a bun from the bowl"),
    (484,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the bun on the plate without a bun"),
    (610,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a sausage from the bowl"),
    (733,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the sausage on either plate"),
    (850,  "PickPlaceCounterToCounter", STAGE_PICK,  "pick up a sausage from the bowl"),
    (967,  "PickPlaceCounterToCounter", STAGE_PLACE, "place the sausage on the plate without a sausage"),
    (1017, DONE, STAGE_DONE, "task complete"),
]

# PortionHotDogs TARGET flagged demos, reconstructed from the clean contact /
# finger-width grasp-release events + base-drive windows (the auto-resolve was
# tripped by collisions / loose grasps on the no-contact buns). Each is four
# pick/places in grasp order; language is by type ("either plate" first, "the
# plate without a X" second). Reviewed by the user.
_PH = "PickPlaceCounterToCounter"
_NAV = "NavigateKitchen"
_PORTION_HOTDOGS_DEMO19 = [
    (54,  _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (135, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (327, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (377, _NAV, STAGE_NAVIGATE, "navigate to either plate"),
    (406, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (446, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (509, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (557, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (598, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (640, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (703, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (752, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (787, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO76 = [
    (100, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (159, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (199, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (253, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (312, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (370, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (432, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (493, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (542, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (583, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (611, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO116 = [
    (129, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (201, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (304, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (364, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (484, _PH,  STAGE_PICK,     "pick up the remaining bun in the bowl"),
    (561, _NAV, STAGE_NAVIGATE, "navigate to the plate without a bun"),
    (607, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (668, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (729, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (799, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (851, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (882, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO289 = [
    (115, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (166, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (248, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (315, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (394, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (461, _NAV, STAGE_NAVIGATE, "navigate to the plate without a bun"),
    (493, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (547, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (608, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (665, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (715, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (744, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO308 = [
    (68,  _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (155, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (227, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (284, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (357, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (413, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (489, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (550, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (580, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO318 = [
    (122, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (200, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (315, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (368, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (469, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (599, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (682, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (771, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (806, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO324 = [
    (120, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (180, _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (287, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (353, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (415, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (484, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (593, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (696, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (726, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO412 = [
    (110,  _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (221,  _NAV, STAGE_NAVIGATE, "navigate to either plate"),
    (282,  _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (450,  _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (536,  _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (661,  _NAV, STAGE_NAVIGATE, "navigate to either plate"),
    (740,  _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (838,  _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (1007, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (1124, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (1222, _NAV, STAGE_NAVIGATE, "navigate back to the bowl"),
    (1302, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (1416, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (1498, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (1529, DONE, STAGE_DONE, "task complete"),
]
_PORTION_HOTDOGS_DEMO500 = [
    (51,  _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (93,  _PH,  STAGE_PLACE,    "place the bun on either plate"),
    (139, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (185, _PH,  STAGE_PLACE,    "place the sausage on either plate"),
    (223, _PH,  STAGE_PICK,     "pick up a sausage from the bowl"),
    (249, _NAV, STAGE_NAVIGATE, "navigate to the plate without a sausage"),
    (283, _PH,  STAGE_PLACE,    "place the sausage on the plate without a sausage"),
    (361, _PH,  STAGE_PICK,     "pick up a bun from the bowl"),
    (414, _PH,  STAGE_PLACE,    "place the bun on the plate without a bun"),
    (444, DONE, STAGE_DONE, "task complete"),
]

# HeatKebabSandwich demo_29 (PRETRAIN, T=1239; single-rack oven). An anomalous
# two-phase rack pull the auto-resolve can't place: the rack is drawn out early
# (to ~half, frames ~150-213, before the kebab is picked), sits through the kebab
# handling, then topped up the rest of the way *between* the two items (~440), and
# pushed back in after the baguette. The auto-slide anchored on the mid-sequence
# top-up and collapsed to zero length. Reviewed with the user: the slide-out is
# the FIRST (early, larger) pull; the mid-sequence top-up is folded into the kebab
# place / baguette pick. Hand-segmented: slide out, pick/place kebab, pick/place
# baguette, slide back in, close, timer. Boundaries are the first pull's plateau
# (213), the auto-resolve's grasp/contact frames (kebab 308/446, baguette 580/692),
# the rack's return below 0.05 (757), door-closed (917), and timer-set (1237).
_HKS_DEMO29 = [
    (213,  "SlideToasterOvenRack",          STAGE_EXECUTE, "slide the toaster oven rack out"),
    (308,  "PickPlaceCounterToToasterOven", STAGE_PICK,    "pick up the kebab skewer from the plate"),
    (446,  "PickPlaceCounterToToasterOven", STAGE_PLACE,   "place the kebab skewer in the toaster oven"),
    (580,  "PickPlaceCounterToToasterOven", STAGE_PICK,    "pick up the baguette from the plate"),
    (692,  "PickPlaceCounterToToasterOven", STAGE_PLACE,   "place the baguette in the toaster oven"),
    (757,  "SlideToasterOvenRack",          STAGE_EXECUTE, "slide the toaster oven rack back in"),
    (917,  "CloseToasterOvenDoor",          STAGE_EXECUTE, "close the toaster oven door"),
    (1237, "TurnOnToasterOven",             STAGE_EXECUTE, "set the toaster oven timer"),
    (1239, DONE, STAGE_DONE, "task complete"),
]

# HeatKebabSandwich TARGET multi-grab anomalies (T-guard scopes each to the
# target split). In these 7 demos the robot grabs BOTH the kebab skewer and the
# baguette together and carries them to the oven in one motion -- and the thin
# items never register a contact-grasp, so the auto-resolve collapsed one item's
# pick/place ("1 pick, 2 places"). Reviewed by the user: all are multi-grabs, so
# each is a single combined pick + place. The slide-out / slide-in / close /
# timer boundaries are the auto-resolve's (correct); the pick->place split is the
# combined lift (earliest item motion) -> settle in the oven (block end).
_HK_MOVE, _HK_SLIDE = "PickPlaceCounterToToasterOven", "SlideToasterOvenRack"
_HK_CLOSE, _HK_TIMER = "CloseToasterOvenDoor", "TurnOnToasterOven"
_HK_PICK_BOTH = "pick up the kebab skewer and baguette from the plate"
_HK_PLACE_BOTH = "place the kebab skewer and baguette in the toaster oven"
_HK_SLIDE_OUT = "slide the toaster oven rack out"
_HK_SLIDE_IN = "slide the toaster oven rack back in"
_HK_CLOSE_LANG = "close the toaster oven door"
_HK_TIMER_LANG = "set the toaster oven timer"

_HKS_DEMO55 = [
    (199,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (405,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (607,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (827,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (1034, _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (1187, _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (1191, DONE, STAGE_DONE, "task complete"),
]
_HKS_DEMO109 = [
    (240,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (454,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (610,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (825,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (1049, _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (1263, _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (1266, DONE, STAGE_DONE, "task complete"),
]
# demo_210 has no rack pull (no slide-out/in) -- just the multi-grab then close.
_HKS_DEMO210 = [
    (130,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (342,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (606,  _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (978,  _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (985,  DONE, STAGE_DONE, "task complete"),
]
_HKS_DEMO214 = [
    (150,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (268,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (453,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (610,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (799,  _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (991,  _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (995,  DONE, STAGE_DONE, "task complete"),
]
_HKS_DEMO323 = [
    (214,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (396,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (562,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (763,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (939,  _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (1240, _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (1244, DONE, STAGE_DONE, "task complete"),
]
_HKS_DEMO378 = [
    (175,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (323,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (640,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (1124, _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (1332, _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (1531, _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (1535, DONE, STAGE_DONE, "task complete"),
]
_HKS_DEMO464 = [
    (174,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_OUT),
    (278,  _HK_MOVE,  STAGE_PICK,    _HK_PICK_BOTH),
    (456,  _HK_MOVE,  STAGE_PLACE,   _HK_PLACE_BOTH),
    (618,  _HK_SLIDE, STAGE_EXECUTE, _HK_SLIDE_IN),
    (806,  _HK_CLOSE, STAGE_EXECUTE, _HK_CLOSE_LANG),
    (984,  _HK_TIMER, STAGE_EXECUTE, _HK_TIMER_LANG),
    (988,  DONE, STAGE_DONE, "task complete"),
]

# ToasterOvenBroilFish demo_103 (T=774): the robot barely nudges the rack (a ~5-frame
# auto-slide) and the DOOR-CLOSE does essentially all the sliding-in, so per user
# review there is NO separate SlideToasterOvenRack subtask -- those frames fold into
# CloseToasterOvenDoor. pick/place/close/temp/done.
_TOBF_DEMO103 = [
    (69,  "PickPlaceCounterToToasterOven", STAGE_PICK,  "pick up the fish from the counter"),
    (161, "PickPlaceCounterToToasterOven", STAGE_PLACE, "place the fish on the rack of the toaster oven"),
    (296, "CloseToasterOvenDoor",          STAGE_EXECUTE, "close the toaster oven door"),
    (754, "AdjustToasterOvenTemperature",  STAGE_EXECUTE, "set the toaster oven temperature to the max"),
    (774, DONE, STAGE_DONE, "task complete"),
]

# AddSweetener demo_40 (PRETRAIN, T=448): a MULTI-GRAB the auto-labeler flagged
# ("expected 3 picks, found 2"). Reviewed with the user: the gripper grabs TWO sugar
# cubes at once (both lift ~frame 82), carries them to the blender, and drops them one
# by one (releases 166 then 185); the third cube is a separate pick/place (300/423). So
# the first trip is one pick of two cubes -> two individual places.
_ADD_SWEETENER_DEMO40 = [
    (82,  "PickPlaceCounterToBlender", STAGE_PICK,  "pick up two sugar cubes from the plate"),
    (166, "PickPlaceCounterToBlender", STAGE_PLACE, "place a sugar cube in the blender"),
    (185, "PickPlaceCounterToBlender", STAGE_PLACE, "place the other sugar cube in the blender"),
    (300, "PickPlaceCounterToBlender", STAGE_PICK,  "pick up the last sugar cube from the plate"),
    (423, "PickPlaceCounterToBlender", STAGE_PLACE, "place the last sugar cube in the blender"),
    (448, DONE, STAGE_DONE, "task complete"),
]

# BlendIngredients demo_11 (PRETRAIN, T=679, kiwi): the operator set the lid down right
# next to the fruit, so the auto-labeler's "gripper left the lid" and "gripper grabbed
# the fruit" collapsed to one instant and the fruit pick went zero-length. Reviewed with
# the user (20fps video): the lid drops (open done) at 0:09 = frame 180, and the close
# ends at ~0:28-0:29 = frame 580 (the gripper releases the re-seated lid). The kiwi
# pick/place split is the fruit's own motion (departs ~248, released ~360).
_BLEND_INGREDIENTS_DEMO11 = [
    (180, "OpenBlenderLid",            STAGE_EXECUTE, "open the blender lid"),
    (248, "PickPlaceCounterToBlender", STAGE_PICK,    "pick up the kiwi from the counter"),
    (360, "PickPlaceCounterToBlender", STAGE_PLACE,   "place the kiwi in the blender"),
    (580, "CloseBlenderLid",           STAGE_EXECUTE, "close the blender lid"),
    (678, "TurnOnBlender",             STAGE_EXECUTE, "start the blender"),
    (679, DONE, STAGE_DONE, "task complete"),
]

# SimmeringSauce demo_73 (PRETRAIN, T=958): a slow pan placement. The pan is held
# in one long grasp (156-361) as the robot picks it up, carries it, and sets it on
# the front-right burner -- and while it is still setting the pan down it nudges
# the onion ~9cm at frame ~180. The auto-resolve read that bump as the onion's
# pick starting, so it truncated the pan-place to ~10 frames and collapsed the
# onion pick. Reviewed with the user (the short pan-place is just the onion nudge
# and should be gone): the pan is really placed at ~361, the onion is grasped ~500
# and dropped in the pan ~620, the tomato is grasped ~748 and dropped ~851, and the
# burner turns on ~942. Boundaries from the finger-width grasp/release, the onion/
# tomato motion, and the knob-on frame.
_SIMMERING_SAUCE_DEMO73 = [
    (170, "PickPlaceCounterToStove", STAGE_PICK,    "pick up the pan from the counter"),
    (361, "PickPlaceCounterToStove", STAGE_PLACE,   "place the pan on the front right burner"),
    (500, "PickPlaceCounterToStove", STAGE_PICK,    "pick up the onion from the counter"),
    (620, "PickPlaceCounterToStove", STAGE_PLACE,   "place the onion in the pan"),
    (748, "PickPlaceCounterToStove", STAGE_PICK,    "pick up the tomato from the counter"),
    (851, "PickPlaceCounterToStove", STAGE_PLACE,   "place the tomato in the pan"),
    (942, "TurnOnStove",             STAGE_EXECUTE, "turn on the front right burner"),
    (958, DONE, STAGE_DONE, "task complete"),
]

# RefillCondimentStation demo_9 (PRETRAIN, T=1035): a very slow cabinet open. The
# door-open detection over-extended OpenCabinet to frame 522 -- right up to the
# ketchup's departure -- collapsing the pick to a single frame. Reviewed with the
# user: OpenCabinet ends at 0:13 (=frame 260 @20fps) and the ketchup pick begins
# there; the carry nav / place / done boundaries match the auto-resolve. The cabinet
# is double-door here (auto lang "open the cabinet doors").
_REFILL_CONDIMENT_DEMO9 = [
    (260,  "OpenCabinet",              STAGE_EXECUTE,  "open the cabinet doors"),
    (524,  "PickPlaceCabinetToDining", STAGE_PICK,     "pick up the ketchup from the cabinet"),
    (815,  "NavigateKitchen",          STAGE_NAVIGATE, "carry the ketchup to the dining table"),
    (998,  "PickPlaceCabinetToDining", STAGE_PLACE,    "place the ketchup on the dining table"),
    (1035, DONE, STAGE_DONE, "task complete"),
]

# RemoveCuttingBoardItems demo_82 (PRETRAIN, T=251): the sweet-potato clear collided
# with the fish -- setting the potato down knocked the fish ~0.20 (frame ~105), a
# false first-move that made the auto-resolve collapse the fish's pick ("expected 2
# picks, found 1"). Reviewed with the user (20fps): the sweet potato is set down by
# 0:06 (frame 120); the fish's pick subtask runs 0:06->0:09 (120-180) and its place
# 0:09->end. Boundaries hand-set to the user's timestamps.
_REMOVE_CB_ITEMS_DEMO82 = [
    (85,  "PickPlaceCuttingBoardToCounter", STAGE_PICK,  "pick up the sweet potato from the cutting board"),
    (120, "PickPlaceCuttingBoardToCounter", STAGE_PLACE, "place the sweet potato on the counter"),
    (180, "PickPlaceCuttingBoardToCounter", STAGE_PICK,  "pick up the fish from the cutting board"),
    (235, "PickPlaceCuttingBoardToCounter", STAGE_PLACE, "place the fish on the counter"),
    (251, DONE, STAGE_DONE, "task complete"),
]

# RestockSinkSupplies demo_102 (PRETRAIN, T=834): a scrambled two-item demo the
# auto-resolve mis-ordered (it labeled the first pick "soap" and ran it 87->644).
# From the finger-width grasp + object z-lift: the SPONGE is handled first (grasped
# ~150, lifted ~210, set down ~320) and the SOAP second (the robot returns to the
# drawer, grasps ~540, lifts ~630, sets down ~780). Reviewed with the user: the first
# pick ends at 0:09 (frame 180). Drawer opens by frame 87; the base barely moves (no
# nav). Later boundaries are the grasp/lift/release frames.
_RESTOCK_SINK_DEMO102 = [
    (87,  "OpenDrawer",               STAGE_EXECUTE, "fully open the drawer"),
    (180, "PickPlaceDrawerToCounter", STAGE_PICK,    "pick up the sponge from the drawer"),
    (330, "PickPlaceDrawerToCounter", STAGE_PLACE,   "place the sponge on the counter next to the sink"),
    (620, "PickPlaceDrawerToCounter", STAGE_PICK,    "pick up the soap dispenser from the drawer"),
    (810, "PickPlaceDrawerToCounter", STAGE_PLACE,   "place the soap dispenser upright on the counter next to the sink"),
    (834, DONE, STAGE_DONE, "task complete"),
]

# CountertopCleanup demo_36 (PRETRAIN, T=1145): the fork pickup collides with the
# bowl (nudges it ~0.10 at frame ~540), a false first-move that made the auto-resolve
# order bowl-before-fork and then miss the real bowl->fridge move entirely (bury it in
# "done"). From the finger-width grasp + z-lift: the true order is cup->cabinet
# (grasp 107, placed ~305), fork->drawer (grasp 465, placed ~720), then bowl->fridge
# (rim-grasped ~840 so gd stays ~0.11, lifted ~900, driven ~2.7m, set down ~1080).
# demo_36 cabinet is "above" (cab_z 1.84), drawer is "in front" (base-y +0.10).
_COUNTERTOP_CLEANUP_DEMO36 = [
    (180,  "PickPlaceCounterToCabinet", STAGE_PICK,     "pick up the cup from the counter"),
    (305,  "PickPlaceCounterToCabinet", STAGE_PLACE,    "place the cup in the cabinet above"),
    (600,  "PickPlaceCounterToDrawer",  STAGE_PICK,     "pick up the fork from the counter"),
    (740,  "PickPlaceCounterToDrawer",  STAGE_PLACE,    "place the fork in the drawer in front"),
    (900,  "PickPlaceCounterToFridge",  STAGE_PICK,     "pick up the bowl of food from the counter"),
    (1020, "NavigateKitchen",           STAGE_NAVIGATE, "carry the bowl of food to the fridge"),
    (1080, "PickPlaceCounterToFridge",  STAGE_PLACE,    "place the bowl of food in the fridge"),
    (1145, DONE, STAGE_DONE, "task complete"),
]

# (env_name, demo_key) -> list of (end_exclusive, subtask_name, stage, lang).
# Frames are absolute and specific to that demo's trajectory length.
MANUAL_OVERRIDES = {
    ("AddSweetener", "demo_40"): _ADD_SWEETENER_DEMO40,
    ("BlendIngredients", "demo_11"): _BLEND_INGREDIENTS_DEMO11,
    ("ToasterOvenBroilFish", "demo_103"): _TOBF_DEMO103,
    ("AddIceCubes", "demo_82"): _ADD_ICE_CUBES_DEMO82,
    ("RestockBowls", "demo_2"): _RESTOCK_BOWLS_DEMO2,
    ("RecycleBottlesByType", "demo_12"): _RECYCLE_BOTTLES_DEMO12,
    ("RecycleBottlesByType", "demo_49"): _RECYCLE_BOTTLES_DEMO49,
    ("RecycleBottlesByType", "demo_21"): _RECYCLE_BOTTLES_DEMO21,
    ("RecycleBottlesByType", "demo_28"): _RECYCLE_BOTTLES_DEMO28,
    ("RecycleBottlesByType", "demo_64"): _RECYCLE_BOTTLES_DEMO64,
    ("RecycleBottlesByType", "demo_99"): _RECYCLE_BOTTLES_DEMO99,
    ("RecycleBottlesByType", "demo_268"): _RECYCLE_BOTTLES_DEMO268,
    ("PackIdenticalLunches", "demo_5"): _PACK_LUNCH_DEMO5,
    ("PackIdenticalLunches", "demo_483"): _PACK_LUNCH_DEMO483,
    ("PackIdenticalLunches", "demo_358"): _PACK_LUNCH_DEMO358,
    ("PackIdenticalLunches", "demo_386"): _PACK_LUNCH_DEMO386,
    ("PackIdenticalLunches", "demo_387"): _PACK_LUNCH_DEMO387,
    ("PortionHotDogs", "demo_45"): _PORTION_HOTDOGS_DEMO45,
    ("PortionHotDogs", "demo_19"): _PORTION_HOTDOGS_DEMO19,
    ("PortionHotDogs", "demo_76"): _PORTION_HOTDOGS_DEMO76,
    ("PortionHotDogs", "demo_116"): _PORTION_HOTDOGS_DEMO116,
    ("PortionHotDogs", "demo_289"): _PORTION_HOTDOGS_DEMO289,
    ("PortionHotDogs", "demo_308"): _PORTION_HOTDOGS_DEMO308,
    ("PortionHotDogs", "demo_318"): _PORTION_HOTDOGS_DEMO318,
    ("PortionHotDogs", "demo_324"): _PORTION_HOTDOGS_DEMO324,
    ("PortionHotDogs", "demo_412"): _PORTION_HOTDOGS_DEMO412,
    ("PortionHotDogs", "demo_500"): _PORTION_HOTDOGS_DEMO500,
    ("HeatKebabSandwich", "demo_29"): _HKS_DEMO29,
    ("HeatKebabSandwich", "demo_55"): _HKS_DEMO55,
    ("HeatKebabSandwich", "demo_109"): _HKS_DEMO109,
    ("HeatKebabSandwich", "demo_210"): _HKS_DEMO210,
    ("HeatKebabSandwich", "demo_214"): _HKS_DEMO214,
    ("HeatKebabSandwich", "demo_323"): _HKS_DEMO323,
    ("HeatKebabSandwich", "demo_378"): _HKS_DEMO378,
    ("HeatKebabSandwich", "demo_464"): _HKS_DEMO464,
    ("SimmeringSauce", "demo_73"): _SIMMERING_SAUCE_DEMO73,
    ("RefillCondimentStation", "demo_9"): _REFILL_CONDIMENT_DEMO9,
    ("RemoveCuttingBoardItems", "demo_82"): _REMOVE_CB_ITEMS_DEMO82,
    ("RestockSinkSupplies", "demo_102"): _RESTOCK_SINK_DEMO102,
    ("CountertopCleanup", "demo_36"): _COUNTERTOP_CLEANUP_DEMO36,
}

# Same overrides keyed by LeRobot episode_index (the LeRobot dataset has no
# demo_key, and its episode order does NOT match the hdf5 demo order). Identify
# the episode by matching the trajectory length: AddIceCubes demo_82 == T=741 ==
# episode_000091; RestockBowls demo_2 == T=1543 == episode_000017.
LEROBOT_OVERRIDES = {
    ("AddIceCubes", 91): _ADD_ICE_CUBES_DEMO82,
    ("RestockBowls", 17): _RESTOCK_BOWLS_DEMO2,
}


def get_override(env_name: str, demo_key: str):
    """Return the manual segment list for an hdf5 demo, or None."""
    return MANUAL_OVERRIDES.get((env_name, demo_key))


def get_lerobot_override(env_name: str, episode_index: int):
    """Return the manual segment list for a LeRobot episode, or None."""
    return LEROBOT_OVERRIDES.get((env_name, int(episode_index)))
