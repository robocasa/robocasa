import asyncio
import datetime
import os
import openai
from openai import OpenAI  # OpenAI for GPT models
from anthropic import Anthropic  # Anthropic for Claude models
import google.generativeai as genai  # Google Gemini models
import argparse
import robocasa

key_folder = os.path.expanduser("~/tmp/keys")
OPENAI_KEY = open(os.path.join(key_folder, "OPENAI_KEY.txt"), "r").readline().strip()
ANTHROPIC_KEY = (
    open(os.path.join(key_folder, "ANTHROPIC_KEY.txt"), "r").readline().strip()
)
GEMINI_KEY = open(os.path.join(key_folder, "GEMINI_KEY.txt"), "r").readline().strip()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)
genai.configure(api_key=GEMINI_KEY)

# Models to use
models = {
    "Claude": [
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
    ],
    # "Gemini": ["gemini-2.0-flash", "gemini-1.5-pro"],
    # "GPT": ["gpt-4o", "o1-preview", "o1-mini"],
}

# Load prompt template
def load_prompt(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return ""


# Generate responses for a single model and task
async def generate_responses(model_family, model_name, prompt, output_path):
    try:
        if model_family == "GPT":
            response = openai_client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
        elif model_family == "Claude":
            message = anthropic_client.messages.create(
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
            result = message.content[0].text
        elif model_family == "Gemini":
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            result = response.text
    except Exception as e:
        return f"Error: {e}"

    with open(output_path, "w") as file:
        file.write(f"PROMPT: \n\n{prompt}\n")
        file.write(f"\nMODEL: {model_family}-{model_name}\n")

        if isinstance(result, list):
            result = "".join(item.text for item in result)

        file.write(f"\nRESPONSE: \n\n{result}")
        print(f"Generated and saved result for {model_family} - {model_name}")


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            robocasa.__path__[0],
            "scripts/internal/taskgen/activity_prompting/outputs/local",
        )
    os.makedirs(output_dir, exist_ok=True)

    # Load prompt template
    prompt_path = args.prompt_path
    base_prompt = load_prompt(prompt_path)

    if not base_prompt:
        print("No valid prompt loaded. Exiting.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tasks = []
    for model_family, model_list in models.items():
        for model in model_list:
            filename = f"{timestamp}_{model_family}_{model}.txt"
            output_path = os.path.join(output_dir, filename)
            tasks.append((model_family, model, base_prompt, output_path))

    results = await asyncio.gather(
        *[generate_responses(*task) for task in tasks], return_exceptions=True
    )
    # # Save responses
    # for (model_family, model, _), result in zip(tasks, results):
    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     filename = f"{model_family}_{model}_{timestamp}.txt"
    #     filepath = os.path.join(output_dir, filename)

    #     with open(filepath, "w") as file:
    #         file.write(f"PROMPT: \n\n{base_prompt}\n")
    #         file.write(f"\nMODEL: {model_family}-{model}\n")

    #         if isinstance(result, list):
    #             result = "".join(item.text for item in result)

    #         file.write(f"\nRESPONSE: \n\n{result}")
    #     print(f"Generated and saved result for {model_family} - {model}")


if __name__ == "__main__":
    asyncio.run(main())
