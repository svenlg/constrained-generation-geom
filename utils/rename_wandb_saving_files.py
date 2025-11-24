import os
import yaml
import shutil

BASE_DIR = "/Users/svlg/MasterThesis/v03_geom/aa_experiments/"
FOLDER_NAME = "al_rho_init"  # change this if your root folder is different
BASE_DIR = os.path.join(BASE_DIR, FOLDER_NAME)
PARAMETER_NAME = "rho_init"

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

    try:
        parameter_value = config["augmented_lagrangian"][PARAMETER_NAME]
    except KeyError:
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
