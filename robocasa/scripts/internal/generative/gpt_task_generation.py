import asyncio
import datetime
import os
import time

from openai import OpenAI

client = OpenAI(
    api_key="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",
)


def chat_with_chatgpt(prompt: str, model: str = "gpt-3.5-turbo"):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":

    prompt_file = "activity_ego4d"
    activity = "Steaming Vegetables"

    recommended_objects = {
        "Defrosting Food": "stove, microwave, sink, meat, packaged food",
        "Boiling Water": "stove, sink, mug, pot, kettle, teapot, jug",
        "Meat Preparation": "stove, meat, condiments, cutting board, vegetables",
        "Setting the Table": "receptacles (bowls, plates, and mugs are good), food items, decoration",
        "Clearing the Table": "receptacles (bowls, plates, and mugs are good), food items, decoration",
        "Santize the Surface": "cleaner, sponge, food items, utensils",
        "Snack Preparation": "microwave, cabinets, bowls, plates, packaged food, fruits, cheese, drinks",
        "Tidying Cabinets and Drawers": "cabinets, drawers, bowls, utensils, condiments",
        "Washing Fruits and Vegetables": "cabinets, drawers, sink, fruits, vegetables",
        "Frying": "cabinets, stove, pans, vegetables, meat",
        "Mixing and Blending": "cabinets, blender, food items",
        "Unloading Groceries": "cabinets, drawer, food items",
        "Baking": "cabinets, receptacles, pastry, dairy, sweets, food items",
        "Serving Food": "stove, microwave, receptacles (bowls, plates, and mugs are good), food items",
        "Reheating Food": "stove, microwave, receptacles (bowls, plates, and mugs are good), food items",
        "Measuring Ingredients": "eggs, cheese, fruits",
        "Making Toast": "stove, toaster, bread, butter, jam",
        "Melting Food": "microwave, bowls, butter, chocolate",
        "Steaming Vegetables": "sink, stove, micrwoave, pot, bowl, vegetables",
    }

    recommended_skills = {
        "Clearing the Table": "pushing and sweeping objects, opening doors",
        "Santize the Surface": "pushing and sweeping objects",
        "Snack Preparation": "opening / closing doors, opening / closing drawers, pressing buttons, pushing",
        "Tidying Cabinets and Drawers": "opening / closing doors, opening / closing drawers",
        "Washing Fruits and Vegetables": "opening / closing doors, opening / closing drawers, turning levers",
        "Frying": "opening / closing doors, opening / closing drawers, twisting knobs",
        "Mixing and Blending": "opening / closing doors, opening / closing drawers",
        "Reheating Food": "swapping",
    }

    # model = "gpt-3.5-turbo" # cost effective
    model = "gpt-4-1106-preview"
    with open(f"prompts/{prompt_file}.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("{ACTIVITY}", f"{activity}")

    prompt += f"\n Here are a list of recommended fixtures and objects \
                to use for this activity: {recommended_objects[activity]}"

    # prompt += f"\n Also make sure some of your tasks use these skills \
    #             along with picking and placing objects: {recommended_skills[activity]}"

    msg = chat_with_chatgpt(prompt=prompt, model=model)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if prompt_file == "activity_ego4d":
        output_dir = os.path.join(
            os.getcwd(), f"gpt-outputs/activity-outputs/{activity}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"gpt-outputs/activity-outputs/{activity}/{time_str}.txt", "w") as f:
            f.write(msg)

    elif prompt_file == "activity_generation":
        output_dir = os.path.join(os.getcwd(), f"gpt-outputs/activity-generation")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"gpt-outputs/activity-generation/{time_str}.txt", "w") as f:
            f.write(msg)
