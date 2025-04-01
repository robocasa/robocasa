import argparse
import os
import random

import imageio
import robosuite
from robosuite import load_controller_config
from tqdm import tqdm

ENVS = [
    "KitchenPnPCounterToCab",
    "KitchenPnPCabToCounter",
    "KitchenPnPCounterToSink",
    "KitchenPnPSinkToCounter",
    "KitchenPnPCounterToMicrowave",
    "KitchenPnPMicrowaveToCounter",
    "KitchenCoffeePnPMug",
    "KitchenOpenDoor",
    "KitchenCloseDoor",
    "KitchenTurnOnSinkFaucet",
    "KitchenTurnOffSinkFaucet",
    "KitchenTurnOnStove",
    "KitchenTurnOffStove",
]

video_writer = imageio.get_writer(os.path.expanduser("~/tmp/env_montage.mp4"), fps=2)

for i in tqdm(range(50)):
    env_name = random.choice(ENVS)
    env = robosuite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera="robot0_frontview",
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        camera_heights=512,
        camera_widths=512,
    )
    env.reset()

    img = env.sim.render(height=1024, width=1536, camera_name="robot0_frontview")[::-1]
    video_writer.append_data(img)

    env.close()
    del env

video_writer.close()
