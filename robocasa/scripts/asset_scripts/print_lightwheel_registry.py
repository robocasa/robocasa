import os
import re
import robocasa

FIXTURES_BASE_PATH = os.path.join(robocasa.__path__[0], "models/assets/fixtures")

for fixture_type in os.listdir(FIXTURES_BASE_PATH):
    fixture_type_dir = os.path.join(FIXTURES_BASE_PATH, fixture_type)
    if not os.path.isdir(fixture_type_dir):
        continue

    valid_model_names = []
    for model_name in os.listdir(fixture_type_dir):
        is_lightwheel_format = bool(re.search(r"\d{3}$", model_name))
        if is_lightwheel_format:
            valid_model_names.append(model_name)

    print(f"===== {fixture_type.upper()} =====")
    valid_model_names.sort()
    for model_name in valid_model_names:
        print(f"{model_name}:")
        print(f"  xml: fixtures/{fixture_type}/{model_name}")
        print()

    print("\n\n\n")
