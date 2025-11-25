import os
import yaml
import shutil

BASE_DIR = "/cluster/home/sgutjahr/MasterThesis/constrained-generation-geom/aa_experiments"
FOLDER_NAME = "baseline_dipole_energy_final"  # change this if your root folder is different
BASE_DIR = os.path.join(BASE_DIR, FOLDER_NAME)
PARAMETER_NAME = "lambda_init"

for name in os.listdir(BASE_DIR):
    folder = os.path.join(BASE_DIR, name)
    if not os.path.isdir(folder):
        continue
    
    # Expect format: "{seed}_{runid}"
    if "_" not in name:
        continue

    seed, _ = name.split("_", 1)

    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found in {folder}, skipping.")
        continue

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parameter_value = (
        config.get("augmented_lagrangian", {}).get(PARAMETER_NAME)
        or config.get(PARAMETER_NAME)
    )

    if parameter_value is None:
        print(f"{PARAMETER_NAME} not found in {config_path}, skipping.")
        continue

    new_name = f"{parameter_value}_{seed}"
    new_path = os.path.join(BASE_DIR, new_name)

    # Avoid overwriting existing folders
    if os.path.exists(new_path):
        print(f"Cannot rename {folder} → {new_path}; target exists.")
        continue

    print(f"Renaming {folder} → {new_path}")
    shutil.move(folder, new_path)
