"""
3D paired-image cell segmentation and per-object intensity analysis.

EDIT THE "USER SETTINGS" SECTION BELOW, THEN RUN THIS FILE.

Workflow
--------
1. Match TIFF stacks in group1/ and group2/ by filename.
2. Segment cells from each Group 1 3D stack.
3. Assign each segmented cell an Object_ID.
4. Apply the same labeled objects to the matching Group 2 stack.
5. Measure mean intensity for every object in both images.
6. Write all objects from all image pairs to one Excel workbook.

Expected folder structure
-------------------------
experiment/
    segmentation_analysis_in_code_settings.py
    group1/
        sample001_group1.tif
        sample002_group1.tif
    group2/
        sample001_group2.tif
        sample002_group2.tif

Install dependencies once
-------------------------
pip install numpy scipy scikit-image tifffile openpyxl

Then edit the USER SETTINGS section and run:
python segmentation_analysis_in_code_settings.py

The script processes one image pair at a time to reduce RAM use.
"""


from __future__ import annotations

import gc
import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


TIFF_EXTENSIONS = {".tif", ".tiff"}


def get_settings():
    """
    ================================================================
    USER SETTINGS - CHANGE VALUES IN THIS SECTION ONLY
    ================================================================
    """

    class Settings:
        pass

    args = Settings()

    # ---------------------------------------------------------------
    # FOLDERS / FILE NAMES
    # ---------------------------------------------------------------

    # Main experiment folder. It must contain the group1 and group2 folders.
    #
    # Windows example:
    # args.root = Path(r"C:\Users\YourName\Desktop\experiment")
    #
    # Mac/Linux example:
    # args.root = Path("/home/yourname/experiment")
    #
    # If the Python file is inside the experiment folder, leave this as:
    args.root = Path(r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split")
    args.group1_folder = "S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split/group_1"
    args.group2_folder = "S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split/group_2"

    # Expected file names:
    # sample001_group1.tif
    # sample001_group2.tif
    args.group1_suffix = "_group1"
    args.group2_suffix = "_group2"

    # ---------------------------------------------------------------
    # OUTPUT LOCATION
    # ---------------------------------------------------------------

    # Choose exactly where the Excel results file will be saved.
    #
    # WINDOWS EXAMPLE:
    # args.output_folder = Path(r"C:\Users\YourName\Desktop\Results")
    #
    # MAC/LINUX EXAMPLE:
    # args.output_folder = Path("/home/yourname/Desktop/Results")
    #
    # Example: save results in a folder named "Results" inside the
    # experiment folder:
    args.output_folder = args.root / "Results"

    # Name of the Excel results file:
    args.output_filename = "3D_cell_intensity_results.xlsx"

    # Complete output path. Normally you do not need to change this line.
    args.output = args.output_folder / args.output_filename

    # ---------------------------------------------------------------
    # SEGMENTATION SETTINGS
    # ---------------------------------------------------------------

    # "watershed" = attempts to split touching cells.
    # "connected" = lower RAM usage, but touching cells remain one object.
    args.segmentation = "watershed"

    # Intensity threshold for Group 1.
    # None = automatically calculate an Otsu threshold for each image.
    # Or enter a number such as 30, 50, 100, etc.
    args.threshold = None

    # False = cells are brighter than background.
    # True = cells are darker than background.
    args.invert = False

    # 3D Gaussian smoothing before thresholding.
    # 0 disables smoothing.
    args.gaussian_sigma = 1.0

    # Objects smaller than this many voxels are removed.
    args.min_size = 3000

    # ---------------------------------------------------------------
    # WATERSHED SETTINGS
    # These only matter when args.segmentation = "watershed"
    # ---------------------------------------------------------------

    # Controls how close two detected cell centers are allowed to be.
    # Increase this if one cell is being split into many objects.
    # Decrease it if neighboring cells are being merged.
    args.min_peak_distance = 30

    # Minimum distance-transform height required for a watershed seed.
    args.peak_threshold = 1.0

    # Relative voxel dimensions.
    # If Z spacing is physically larger than XY spacing, enter the ratio.
    # Example: Z = 1.0 um and XY = 0.25 um -> z_spacing=4.0, xy_spacing=1.0
    args.z_spacing = 1.0
    args.xy_spacing = 1.0

    # ---------------------------------------------------------------
    # OPTIONAL OUTPUT
    # ---------------------------------------------------------------

    # True = save numbered 3D label TIFF files in a "labels" folder.
    # Recommended while checking/tuning segmentation.
    args.save_labels = False

    # Save a colored RGB version of the labels for easier visual inspection.
    # This is visualization only; the raw label TIFF remains the authoritative
    # Object_ID image used for measurements.
    args.save_colored_labels = True

    # Save a colored-label overlay on top of the original Group 1 image.
    args.save_overlay = False

    # Opacity of the colored labels in the overlay.
    # 0.0 = only Group 1 image, 1.0 = only label colors.
    args.overlay_alpha = 0.45

    # ---------------------------------------------------------------
    # MASK REPAIR / HOLE FILLING
    # ---------------------------------------------------------------

    # Fill completely enclosed holes inside segmented cells.
    args.fill_holes = True

    # Close small gaps caused by dim/missed voxels during thresholding.
    args.close_gaps = True

    # Number of 3D binary-closing iterations.
    # Start with 1 or 2. Larger values can accidentally join nearby cells.
    args.closing_iterations = 2

    return args


def normalize_pair_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if not stem.lower().endswith(suffix.lower()):
        raise ValueError(
            f'Filename "{path.name}" does not end with expected suffix "{suffix}".'
        )
    return stem[: -len(suffix)]


def find_pairs(
    group1_dir: Path,
    group2_dir: Path,
    group1_suffix: str,
    group2_suffix: str,
) -> list[tuple[str, Path, Path]]:
    if not group1_dir.is_dir():
        raise FileNotFoundError(f"Group 1 folder not found: {group1_dir}")
    if not group2_dir.is_dir():
        raise FileNotFoundError(f"Group 2 folder not found: {group2_dir}")

    def collect(folder: Path, suffix: str) -> dict[str, Path]:
        result = {}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS:
                if p.stem.lower().endswith(suffix.lower()):
                    key = normalize_pair_name(p, suffix)
                    if key in result:
                        raise ValueError(f'Duplicate pair name "{key}" in {folder}')
                    result[key] = p
        return result

    g1 = collect(group1_dir, group1_suffix)
    g2 = collect(group2_dir, group2_suffix)

    if not g1:
        raise FileNotFoundError(
            f'No TIFF files ending in "{group1_suffix}.tif/.tiff" found in {group1_dir}'
        )

    missing_g2 = sorted(set(g1) - set(g2))
    missing_g1 = sorted(set(g2) - set(g1))
    if missing_g2 or missing_g1:
        parts = []
        if missing_g2:
            parts.append("Missing Group 2 matches: " + ", ".join(missing_g2))
        if missing_g1:
            parts.append("Missing Group 1 matches: " + ", ".join(missing_g1))
        raise ValueError("\n".join(parts))

    return [(name, g1[name], g2[name]) for name in sorted(g1)]


def open_stack(path: Path) -> np.ndarray:
    """
    Prefer memory mapping. Fall back to tifffile.imread if the TIFF layout
    cannot be memory mapped.
    """
    try:
        arr = tifffile.memmap(path)
    except Exception:
        arr = tifffile.imread(path)

    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"{path.name}: expected a 3D TIFF stack after squeezing, got shape {arr.shape}"
        )
    return arr


def estimate_otsu_threshold(image: np.ndarray, max_samples: int = 2_000_000) -> float:
    """
    Estimate Otsu threshold from an evenly strided sample to avoid copying
    the entire 3D stack into a temporary 1D array.
    """
    flat = image.reshape(-1)
    step = max(1, flat.size // max_samples)
    sample = np.asarray(flat[::step])
    return float(threshold_otsu(sample))


def make_foreground_mask(
    image: np.ndarray,
    threshold: float | None,
    sigma: float,
    min_size: int,
    invert: bool,
    fill_holes: bool,
    close_gaps: bool,
    closing_iterations: int,
) -> tuple[np.ndarray, float]:
    """
    Create a 3D foreground mask.

    Smoothing is stored as float32 to reduce RAM. If sigma == 0, no floating
    point copy of the complete stack is created.
    """
    if sigma > 0:
        print(f"    Gaussian smoothing (sigma={sigma})...")
        smoothed = np.empty(image.shape, dtype=np.float32)
        ndi.gaussian_filter(image, sigma=sigma, output=smoothed, mode="nearest")
        threshold_source = smoothed
    else:
        smoothed = None
        threshold_source = image

    if threshold is None:
        used_threshold = estimate_otsu_threshold(threshold_source)
    else:
        used_threshold = float(threshold)

    print(f"    Threshold: {used_threshold:.4f}")

    if invert:
        mask = np.asarray(threshold_source < used_threshold, dtype=bool)
    else:
        mask = np.asarray(threshold_source > used_threshold, dtype=bool)

    del smoothed
    gc.collect()

    if close_gaps and closing_iterations > 0:
        print(
            f"    Closing small 3D mask gaps "
            f"({closing_iterations} iteration(s))..."
        )
        structure = ndi.generate_binary_structure(3, 1)
        mask = ndi.binary_closing(
            mask,
            structure=structure,
            iterations=closing_iterations,
        )

    if fill_holes:
        print("    Filling enclosed 3D holes...")
        mask = ndi.binary_fill_holes(mask)

    if min_size > 1:
        print(f"    Removing foreground components smaller than {min_size} voxels...")
        # remove_small_objects works directly on the Boolean mask.
        mask = remove_small_objects(mask, min_size=min_size, connectivity=1)

    return np.asarray(mask, dtype=bool), used_threshold


def connected_component_segmentation(mask: np.ndarray, min_size: int) -> np.ndarray:
    structure = ndi.generate_binary_structure(3, 1)
    labels, count = ndi.label(mask, structure=structure)
    labels = labels.astype(np.int32, copy=False)
    print(f"    Initial connected objects: {count}")

    if min_size > 1 and count > 0:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        labels[~keep[labels]] = 0
        labels, count = ndi.label(labels > 0, structure=structure)
        labels = labels.astype(np.int32, copy=False)

    print(f"    Final objects: {int(labels.max())}")
    return labels


def watershed_segmentation(
    mask: np.ndarray,
    min_size: int,
    min_peak_distance: int,
    peak_threshold: float,
    z_spacing: float,
    xy_spacing: float,
) -> np.ndarray:
    """
    Split touching 3D objects using a distance-transform watershed.

    The distance transform is the largest temporary array in this script.
    SciPy creates it as float64, so watershed mode can require substantial RAM
    for very large stacks.
    """
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    print("    Computing 3D distance transform...")
    distance = ndi.distance_transform_edt(
        mask,
        sampling=(z_spacing, xy_spacing, xy_spacing),
    )

    # A maximum filter provides local maxima without constructing a large
    # coordinate list. The window size controls how close two seeds may be.
    size = max(1, 2 * int(min_peak_distance) + 1)
    print(f"    Finding watershed seeds (maximum-filter size={size})...")
    local_max = distance == ndi.maximum_filter(distance, size=size, mode="constant")
    local_max &= mask
    local_max &= distance >= float(peak_threshold)

    seed_structure = ndi.generate_binary_structure(3, 1)
    markers, marker_count = ndi.label(local_max, structure=seed_structure)
    markers = markers.astype(np.int32, copy=False)
    del local_max
    gc.collect()

    print(f"    Watershed seeds: {marker_count}")

    if marker_count == 0:
        print("    No watershed seeds found; falling back to connected components.")
        del distance, markers
        gc.collect()
        return connected_component_segmentation(mask, min_size)

    print("    Running 3D watershed...")
    labels = watershed(-distance, markers, mask=mask)
    labels = labels.astype(np.int32, copy=False)

    del distance, markers
    gc.collect()

    # Remove watershed fragments below the requested volume and then relabel
    # consecutively so Object_ID values are compact: 1, 2, 3, ...
    max_label = int(labels.max())
    if min_size > 1 and max_label > 0:
        sizes = np.bincount(labels.ravel(), minlength=max_label + 1)
        keep = sizes >= min_size
        keep[0] = False
        labels[~keep[labels]] = 0

    unique = np.unique(labels)
    unique = unique[unique != 0]
    if unique.size:
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        lut[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
        labels = lut[labels]
    else:
        labels.fill(0)

    print(f"    Final objects: {int(labels.max())}")
    return labels


def measure_objects(
    labels: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return voxel_count, Group 1 intensity sum, and Group 2 intensity sum.

    Measurement is performed one Z plane at a time. This avoids making
    flattened copies of both full intensity stacks.
    """
    n_objects = int(labels.max())
    counts = np.zeros(n_objects + 1, dtype=np.int64)
    sum_g1 = np.zeros(n_objects + 1, dtype=np.float64)
    sum_g2 = np.zeros(n_objects + 1, dtype=np.float64)

    for z in range(labels.shape[0]):
        lab = np.asarray(labels[z]).ravel()
        g1 = np.asarray(group1[z]).ravel()
        g2 = np.asarray(group2[z]).ravel()

        counts += np.bincount(lab, minlength=n_objects + 1)
        sum_g1 += np.bincount(lab, weights=g1, minlength=n_objects + 1)
        sum_g2 += np.bincount(lab, weights=g2, minlength=n_objects + 1)

    return counts, sum_g1, sum_g2


def object_bounding_boxes(labels: np.ndarray) -> list[tuple[slice, slice, slice] | None]:
    """
    scipy.ndimage.find_objects is efficient and lets us report each object's
    3D bounding box without creating coordinate arrays for every voxel.
    """
    return ndi.find_objects(labels)



def make_label_colors(labels: np.ndarray) -> np.ndarray:
    """
    Create an RGB visualization of integer object labels.

    Colors are deterministic: the same Object_ID receives the same color.
    Background (label 0) is black.

    This visualization does NOT replace the raw integer label TIFF.
    """
    n_objects = int(labels.max())

    # A compact, high-contrast palette. IDs cycle through this palette.
    palette = np.array([
        [230,  25,  75],
        [ 60, 180,  75],
        [255, 225,  25],
        [  0, 130, 200],
        [245, 130,  48],
        [145,  30, 180],
        [ 70, 240, 240],
        [240,  50, 230],
        [210, 245,  60],
        [250, 190, 212],
        [  0, 128, 128],
        [220, 190, 255],
        [170, 110,  40],
        [255, 250, 200],
        [128,   0,   0],
        [170, 255, 195],
        [128, 128,   0],
        [255, 215, 180],
        [  0,   0, 128],
        [128, 128, 128],
    ], dtype=np.uint8)

    lut = np.zeros((n_objects + 1, 3), dtype=np.uint8)
    if n_objects > 0:
        ids = np.arange(1, n_objects + 1)
        lut[1:] = palette[(ids - 1) % len(palette)]

    return lut[labels]


def normalize_group1_for_overlay(group1: np.ndarray) -> np.ndarray:
    """
    Convert Group 1 to uint8 grayscale for visualization.

    Percentile scaling prevents a few extreme pixels from making the
    remainder of the image appear too dark.
    """
    sample = group1.reshape(-1)
    step = max(1, sample.size // 1_000_000)
    sampled = np.asarray(sample[::step], dtype=np.float32)

    low, high = np.percentile(sampled, (1.0, 99.5))
    if high <= low:
        low = float(np.min(sampled))
        high = float(np.max(sampled))

    if high <= low:
        return np.zeros(group1.shape, dtype=np.uint8)

    output = np.empty(group1.shape, dtype=np.uint8)
    scale = 255.0 / (high - low)

    # Plane-by-plane conversion reduces temporary memory usage.
    for z in range(group1.shape[0]):
        plane = np.asarray(group1[z], dtype=np.float32)
        plane = np.clip((plane - low) * scale, 0, 255)
        output[z] = plane.astype(np.uint8)

    return output


def make_overlay(
    group1: np.ndarray,
    colored_labels: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Overlay RGB labels on Group 1. Background remains grayscale.
    """
    gray = normalize_group1_for_overlay(group1)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    del gray

    foreground = labels > 0

    # Work one Z plane at a time to limit temporary memory.
    for z in range(labels.shape[0]):
        fg = foreground[z]
        if not np.any(fg):
            continue

        base = overlay[z]
        colors = colored_labels[z]

        blended = (
            (1.0 - alpha) * base[fg].astype(np.float32)
            + alpha * colors[fg].astype(np.float32)
        )
        base[fg] = np.clip(blended, 0, 255).astype(np.uint8)

    return overlay

def create_workbook(output_path: Path):
    wb = Workbook(write_only=False)

    ws = wb.active
    ws.title = "Object Results"
    headers = [
        "Image",
        "Object_ID",
        "Voxel_Count",
        "Group1_Mean_Intensity",
        "Group2_Mean_Intensity",
        "Z_Min",
        "Z_Max",
        "Y_Min",
        "Y_Max",
        "X_Min",
        "X_Max",
    ]
    ws.append(headers)

    summary = wb.create_sheet("Image Summary")
    summary_headers = [
        "Image",
        "Group1_File",
        "Group2_File",
        "Threshold",
        "Object_Count",
        "Total_Object_Voxels",
        "Group1_Mean_Of_Object_Means",
        "Group2_Mean_Of_Object_Means",
        "Status",
    ]
    summary.append(summary_headers)

    # Formatting is deliberately lightweight so the workbook stays responsive
    # even when many thousands of objects are present.
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for sheet in (ws, summary):
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    widths_results = {
        "A": 42, "B": 12, "C": 14, "D": 23, "E": 23,
        "F": 10, "G": 10, "H": 10, "I": 10, "J": 10, "K": 10,
    }
    for col, width in widths_results.items():
        ws.column_dimensions[col].width = width

    widths_summary = {
        "A": 42, "B": 48, "C": 48, "D": 14, "E": 14,
        "F": 20, "G": 28, "H": 28, "I": 18,
    }
    for col, width in widths_summary.items():
        summary.column_dimensions[col].width = width

    return wb, ws, summary


def main() -> int:
    args = get_settings()
    root = args.root.expanduser().resolve()
    group1_dir = root / args.group1_folder
    group2_dir = root / args.group2_folder
    output_path = Path(args.output).expanduser().resolve()

    if args.min_size < 1:
        raise ValueError("--min-size must be >= 1")
    if args.gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be >= 0")
    if args.min_peak_distance < 0:
        raise ValueError("--min-peak-distance must be >= 0")
    if args.z_spacing <= 0 or args.xy_spacing <= 0:
        raise ValueError("z_spacing and xy_spacing must be > 0")
    if args.closing_iterations < 0:
        raise ValueError("closing_iterations must be >= 0")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0.0 and 1.0")

    pairs = find_pairs(
        group1_dir,
        group2_dir,
        args.group1_suffix,
        args.group2_suffix,
    )
    print(f"Found {len(pairs)} matched image pair(s).")
    print(f"Segmentation mode: {args.segmentation}")
    print(f"Output: {output_path}")

    # Create the requested output folder automatically if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Saved label TIFFs go into a "labels" subfolder beside the Excel file.
    labels_dir = output_path.parent / "labels"
    if args.save_labels or args.save_colored_labels or args.save_overlay:
        labels_dir.mkdir(parents=True, exist_ok=True)

    wb, results_ws, summary_ws = create_workbook(output_path)

    total_objects = 0
    failed = 0

    for pair_index, (pair_name, g1_path, g2_path) in enumerate(pairs, start=1):
        print()
        print("=" * 72)
        print(f"[{pair_index}/{len(pairs)}] {pair_name}")
        print(f"  Group 1: {g1_path.name}")
        print(f"  Group 2: {g2_path.name}")

        try:
            group1 = open_stack(g1_path)
            group2 = open_stack(g2_path)

            if group1.shape != group2.shape:
                raise ValueError(
                    f"Shape mismatch: Group 1 {group1.shape} vs Group 2 {group2.shape}"
                )

            print(f"  Shape (Z,Y,X): {group1.shape}")
            print(f"  Group 1 dtype: {group1.dtype}; Group 2 dtype: {group2.dtype}")

            mask, used_threshold = make_foreground_mask(
                group1,
                threshold=args.threshold,
                sigma=args.gaussian_sigma,
                min_size=args.min_size,
                invert=args.invert,
                fill_holes=args.fill_holes,
                close_gaps=args.close_gaps,
                closing_iterations=args.closing_iterations,
            )

            foreground_voxels = int(np.count_nonzero(mask))
            print(f"    Foreground voxels: {foreground_voxels:,}")

            if args.segmentation == "watershed":
                labels = watershed_segmentation(
                    mask,
                    min_size=args.min_size,
                    min_peak_distance=args.min_peak_distance,
                    peak_threshold=args.peak_threshold,
                    z_spacing=args.z_spacing,
                    xy_spacing=args.xy_spacing,
                )
            else:
                labels = connected_component_segmentation(mask, args.min_size)

            del mask
            gc.collect()

            n_objects = int(labels.max())
            if n_objects == 0:
                summary_ws.append([
                    pair_name,
                    g1_path.name,
                    g2_path.name,
                    used_threshold,
                    0,
                    0,
                    None,
                    None,
                    "No objects detected",
                ])
                print("  WARNING: no objects detected.")
                del labels, group1, group2
                gc.collect()
                continue

            print("    Measuring object intensities...")
            counts, sum_g1, sum_g2 = measure_objects(labels, group1, group2)
            boxes = object_bounding_boxes(labels)

            mean_g1 = np.divide(
                sum_g1,
                counts,
                out=np.full_like(sum_g1, np.nan, dtype=np.float64),
                where=counts > 0,
            )
            mean_g2 = np.divide(
                sum_g2,
                counts,
                out=np.full_like(sum_g2, np.nan, dtype=np.float64),
                where=counts > 0,
            )

            for object_id in range(1, n_objects + 1):
                box = boxes[object_id - 1] if object_id - 1 < len(boxes) else None
                if box is None or counts[object_id] == 0:
                    continue

                z_slice, y_slice, x_slice = box
                # Excel receives 0-based voxel coordinates. Max values are inclusive.
                results_ws.append([
                    pair_name,
                    object_id,
                    int(counts[object_id]),
                    float(mean_g1[object_id]),
                    float(mean_g2[object_id]),
                    int(z_slice.start),
                    int(z_slice.stop - 1),
                    int(y_slice.start),
                    int(y_slice.stop - 1),
                    int(x_slice.start),
                    int(x_slice.stop - 1),
                ])

            if args.save_labels:
                label_path = labels_dir / f"{pair_name}_labels.tif"
                print(f"    Saving raw Object_ID labels: {label_path.name}")
                tifffile.imwrite(
                    label_path,
                    labels,
                    photometric="minisblack",
                    metadata={"axes": "ZYX"},
                    compression="zlib",
                )

            if args.save_colored_labels or args.save_overlay:
                print("    Creating colored label visualization...")
                colored_labels = make_label_colors(labels)

                if args.save_colored_labels:
                    color_path = labels_dir / f"{pair_name}_labels_COLOR.tif"
                    print(f"    Saving colored labels: {color_path.name}")
                    tifffile.imwrite(
                        color_path,
                        colored_labels,
                        photometric="rgb",
                        metadata={"axes": "ZYXC"},
                        compression="zlib",
                    )

                if args.save_overlay:
                    overlay_path = labels_dir / f"{pair_name}_labels_OVERLAY.tif"
                    print(f"    Saving Group 1 overlay: {overlay_path.name}")
                    overlay = make_overlay(
                        group1,
                        colored_labels,
                        labels,
                        args.overlay_alpha,
                    )
                    tifffile.imwrite(
                        overlay_path,
                        overlay,
                        photometric="rgb",
                        metadata={"axes": "ZYXC"},
                        compression="zlib",
                    )
                    del overlay

                del colored_labels
                gc.collect()

            valid = counts[1:] > 0
            summary_ws.append([
                pair_name,
                g1_path.name,
                g2_path.name,
                used_threshold,
                n_objects,
                int(counts[1:].sum()),
                float(np.mean(mean_g1[1:][valid])) if np.any(valid) else None,
                float(np.mean(mean_g2[1:][valid])) if np.any(valid) else None,
                "OK",
            ])

            total_objects += n_objects
            print(f"  Completed: {n_objects} object(s).")

            del (
                labels,
                group1,
                group2,
                counts,
                sum_g1,
                sum_g2,
                mean_g1,
                mean_g2,
                boxes,
            )
            gc.collect()

        except Exception as exc:
            failed += 1
            print(f"  ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()

            summary_ws.append([
                pair_name,
                g1_path.name,
                g2_path.name,
                None,
                None,
                None,
                None,
                None,
                f"ERROR: {exc}",
            ])
            gc.collect()

    # Number formatting after all rows are written.
    for row in results_ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for cell in row:
            cell.number_format = "0.0000"

    for row in summary_ws.iter_rows(min_row=2):
        row[3].number_format = "0.0000"  # Threshold
        row[6].number_format = "0.0000"
        row[7].number_format = "0.0000"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)

    print()
    print("=" * 72)
    print("Analysis complete.")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Objects measured: {total_objects}")
    print(f"Failed pairs: {failed}")
    print(f"Excel file: {output_path}")

    if args.save_labels or args.save_colored_labels or args.save_overlay:
        print(f"Label/visualization TIFF folder: {labels_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)