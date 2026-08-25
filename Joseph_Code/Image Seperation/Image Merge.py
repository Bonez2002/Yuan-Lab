import os
import tifffile as tiff
import numpy as np

# =========================
# USER VARIABLES
# =========================

input_folder = r"E:/2026_04_09/Raw"
output_folder = r"E:/2026_04_09/Raw/Composites"

separation_step = 2   # change to 3 for every third slice

# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs(output_folder, exist_ok=True)

# =========================
# CREATE LUTS (ImageJ format)
# =========================

def create_lut(color):
    lut = np.zeros((3, 256), dtype=np.uint8)

    if color == "red":
        lut[0] = np.arange(256)
    elif color == "green":
        lut[1] = np.arange(256)
    elif color == "blue":
        lut[2] = np.arange(256)

    return lut

blue_lut = create_lut("blue")
red_lut = create_lut("red")

# =========================
# PROCESS TIFF FILES
# =========================

for file in os.listdir(input_folder):

    if file.lower().endswith((".tif", ".tiff")):

        file_path = os.path.join(input_folder, file)

        # --- Safe read ---
        try:
            stack = tiff.imread(file_path)
        except Exception as e:
            print(f"Skipping {file} (not a valid TIFF): {e}")
            continue

        print(f"{file} shape: {stack.shape}")

        # --- Expecting (Z, Y, X) ---
        if stack.ndim != 3:
            print(f"Skipping {file} (unexpected shape)")
            continue

        # --- Split into groups ---
        separated = [stack[i::separation_step] for i in range(separation_step)]

        # --- Match Z depth ---
        min_z = min(group.shape[0] for group in separated)
        separated = [group[:min_z] for group in separated]

        # --- Stack into channels (C, Z, Y, X) ---
        composite = np.stack(separated, axis=0)

        # --- Convert to ImageJ order (Z, C, Y, X) ---
        composite = np.moveaxis(composite, 0, 1)

        print(f"Composite shape: {composite.shape}")  # should be (Z, C, Y, X)

        # --- Output path ---
        output_path = os.path.join(
            output_folder,
            file.replace(".tif", "_composite.tif").replace(".tiff", "_composite.tiff")
        )

        # --- Save with composite mode + LUTs ---
        tiff.imwrite(
            output_path,
            composite.astype(stack.dtype),
            imagej=True,
            metadata={
                'axes': 'ZCYX',
                'mode': 'composite',   # forces color display
                'LUTs': [blue_lut, red_lut]
            }
        )

        print(f"Processed: {file}")

print("Done!")