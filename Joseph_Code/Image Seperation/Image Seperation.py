import os
import tifffile as tiff
import numpy as np

# =========================
# USER VARIABLES
# =========================

input_folder = r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_19/Raw"
output_base_folder = r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_19/Raw/Split"

separation_step = 2   # change to 3 for every third slice

# =========================
# CREATE OUTPUT FOLDERS
# =========================

output_folders = []

for i in range(separation_step):
    folder = os.path.join(output_base_folder, f"group_{i+1}")
    os.makedirs(folder, exist_ok=True)
    output_folders.append(folder)

# =========================
# PROCESS TIFF FILES
# =========================

for file in os.listdir(input_folder):

    if file.lower().endswith((".tif", ".tiff")):

        file_path = os.path.join(input_folder, file)

        # Read Z-stack
        stack = tiff.imread(file_path)

        # Split stack into groups
        separated = [stack[i::separation_step] for i in range(separation_step)]

        # Save each separated stack
        for i, group in enumerate(separated):

            output_path = os.path.join(
                output_folders[i],
                file.replace(".tif", f"_group{i+1}.tif").replace(".tiff", f"_group{i+1}.tiff")
            )

            tiff.imwrite(output_path, group)

        print(f"Processed: {file}")

print("Done!")