"""
GPU-oriented 3D paired-image cell segmentation and per-object intensity analysis.

This version keeps the same folder matching and Excel output workflow, but the
segmentation path uses CUDA-friendly operations when CuPy/CuPyX are available:

- 3D Gaussian smoothing on GPU
- thresholding on GPU
- gap closing and hole filling on GPU
- connected-component labeling on GPU

Important:
- This is a GPU-connected-component workflow, not a GPU watershed workflow.
- It is best when cells are reasonably separated already.
- If cells are strongly touching and must be split, the CPU watershed version
  is still the better choice.
- If CuPy is not installed, the script falls back to the CPU version.

Edit the USER SETTINGS section below, then run the file.
"""

from __future__ import annotations

from pathlib import Path
import gc
import sys
import traceback

import numpy as np
import tifffile
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cndi = None
    GPU_AVAILABLE = False


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
    # FOLDERS / FILE NAME PATTERNS
    # ---------------------------------------------------------------
    args.root = Path(r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split")
    args.group1_folder = "S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split/group_1"
    args.group2_folder = "S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_06/Raw/Split/group_2"
    args.group1_suffix = "_group1"
    args.group2_suffix = "_group2"

    # ---------------------------------------------------------------
    # OUTPUT LOCATION
    # ---------------------------------------------------------------
    args.output_folder = args.root / "Results_GPU"
    args.output_filename = "3D_cell_intensity_results_GPU.xlsx"
    args.output = args.output_folder / args.output_filename

    # ---------------------------------------------------------------
    # GPU / SEGMENTATION
    # ---------------------------------------------------------------
    # True = use GPU segmentation when CuPy is available.
    # False = always use CPU segmentation.
    args.use_gpu = True

    # GPU index to use when more than one GPU is available.
    args.gpu_device = 0

    # Segmentation style:
    # "gpu_connected" -> GPU threshold + cleanup + connected components
    # "cpu_connected"  -> CPU threshold + cleanup + connected components
    args.segmentation = "gpu_connected"

    # Threshold on Group 1 intensity:
    # None = automatic Otsu threshold per image.
    # Or enter a fixed value like 50, 80, 120, etc.
    args.threshold = None

    # True if cells are darker than background.
    args.invert = False

    # 3D Gaussian smoothing before thresholding.
    # 0 disables smoothing.
    args.gaussian_sigma = 1.0

    # Remove objects smaller than this number of voxels.
    args.min_size = 200

    # Fill fully enclosed holes inside segmented cells.
    args.fill_holes = True

    # Close small gaps caused by gating/thresholding before labeling.
    args.close_gaps = True

    # Strength of the gap closing. Start at 1 or 2.
    args.closing_iterations = 2

    # Optional extra cleanup on CPU after segmentation.
    # Usually leave True.
    args.remove_small_objects_after_labeling = False

    # ---------------------------------------------------------------
    # VISUALIZATION OUTPUTS
    # ---------------------------------------------------------------
    args.save_labels = True
    args.save_colored_labels = True
    args.save_overlay = True
    args.overlay_alpha = 0.45

    return args


def normalize_group1_for_overlay(group1: np.ndarray) -> np.ndarray:
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
    for z in range(group1.shape[0]):
        plane = np.asarray(group1[z], dtype=np.float32)
        plane = np.clip((plane - low) * scale, 0, 255)
        output[z] = plane.astype(np.uint8)
    return output


def make_label_colors(labels: np.ndarray) -> np.ndarray:
    n_objects = int(labels.max())
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


def make_overlay(
    group1: np.ndarray,
    colored_labels: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> np.ndarray:
    gray = normalize_group1_for_overlay(group1)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    foreground = labels > 0

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


def normalize_pair_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if not stem.lower().endswith(suffix.lower()):
        raise ValueError(f'Filename "{path.name}" does not end with expected suffix "{suffix}".')
    return stem[: -len(suffix)]


def find_pairs(group1_dir: Path, group2_dir: Path, group1_suffix: str, group2_suffix: str):
    if not group1_dir.is_dir():
        raise FileNotFoundError(f"Group 1 folder not found: {group1_dir}")
    if not group2_dir.is_dir():
        raise FileNotFoundError(f"Group 2 folder not found: {group2_dir}")

    def collect(folder: Path, suffix: str) -> dict[str, Path]:
        out = {}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS and p.stem.lower().endswith(suffix.lower()):
                key = normalize_pair_name(p, suffix)
                if key in out:
                    raise ValueError(f'Duplicate pair name "{key}" in {folder}')
                out[key] = p
        return out

    g1 = collect(group1_dir, group1_suffix)
    g2 = collect(group2_dir, group2_suffix)

    if not g1:
        raise FileNotFoundError(f'No TIFF files ending in "{group1_suffix}.tif/.tiff" found in {group1_dir}')

    missing_g2 = sorted(set(g1) - set(g2))
    missing_g1 = sorted(set(g2) - set(g1))
    if missing_g2 or missing_g1:
        msg = []
        if missing_g2:
            msg.append("Missing Group 2 matches: " + ", ".join(missing_g2))
        if missing_g1:
            msg.append("Missing Group 1 matches: " + ", ".join(missing_g1))
        raise ValueError("\n".join(msg))

    return [(name, g1[name], g2[name]) for name in sorted(g1)]


def open_stack(path: Path) -> np.ndarray:
    try:
        arr = tifffile.memmap(path)
    except Exception:
        arr = tifffile.imread(path)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected 3D TIFF after squeezing, got shape {arr.shape}")
    return arr


def estimate_otsu_threshold_cpu(image: np.ndarray, max_samples: int = 2_000_000) -> float:
    flat = image.reshape(-1)
    step = max(1, flat.size // max_samples)
    sample = np.asarray(flat[::step])
    return float(threshold_otsu(sample))


def estimate_otsu_threshold_gpu(image_gpu, max_samples: int = 2_000_000) -> float:
    flat = image_gpu.ravel()
    step = max(1, flat.size // max_samples)
    sample = flat[::step]
    sample_cpu = cp.asnumpy(sample)
    return float(threshold_otsu(sample_cpu))


def gpu_connected_segmentation(group1: np.ndarray, args) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU libraries (CuPy/CuPyX) are not installed. "
            "Install a CuPy build that matches your CUDA setup, or use the CPU version."
        )

    with cp.cuda.Device(int(args.gpu_device)):
        g = cp.asarray(group1)

        if float(args.gaussian_sigma) > 0:
            smoothed = cndi.gaussian_filter(g, sigma=float(args.gaussian_sigma), mode="nearest")
        else:
            smoothed = g

        if args.threshold is None:
            used_threshold = estimate_otsu_threshold_gpu(smoothed)
        else:
            used_threshold = float(args.threshold)

        mask = smoothed < used_threshold if args.invert else smoothed > used_threshold

        if args.close_gaps and int(args.closing_iterations) > 0:
            structure = cndi.generate_binary_structure(3, 1)
            mask = cndi.binary_closing(
                mask,
                structure=structure,
                iterations=int(args.closing_iterations),
            )

        if args.fill_holes:
            mask = cndi.binary_fill_holes(mask)

        structure = cndi.generate_binary_structure(3, 1)
        labels, _ = cndi.label(mask, structure=structure)
        labels = labels.astype(cp.int32, copy=False)

        if int(args.min_size) > 1 and labels.size > 0:
            counts = cp.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
            keep = counts >= int(args.min_size)
            keep[0] = False
            labels = cp.where(keep[labels], labels, 0).astype(cp.int32, copy=False)

        if int(labels.max()) > 0:
            unique = cp.unique(labels)
            unique = unique[unique != 0]
            if unique.size:
                lut = cp.zeros(int(labels.max()) + 1, dtype=cp.int32)
                lut[unique] = cp.arange(1, unique.size + 1, dtype=cp.int32)
                labels = lut[labels]

        return cp.asnumpy(labels.astype(cp.int32, copy=False)), used_threshold


def cpu_connected_segmentation(group1: np.ndarray, args) -> tuple[np.ndarray, float]:
    if float(args.gaussian_sigma) > 0:
        smoothed = np.empty(group1.shape, dtype=np.float32)
        ndi.gaussian_filter(group1, sigma=float(args.gaussian_sigma), output=smoothed, mode="nearest")
        source = smoothed
    else:
        source = group1

    if args.threshold is None:
        used_threshold = estimate_otsu_threshold_cpu(source)
    else:
        used_threshold = float(args.threshold)

    mask = source < used_threshold if args.invert else source > used_threshold

    if args.close_gaps and int(args.closing_iterations) > 0:
        structure = ndi.generate_binary_structure(3, 1)
        mask = ndi.binary_closing(mask, structure=structure, iterations=int(args.closing_iterations))

    if args.fill_holes:
        mask = ndi.binary_fill_holes(mask)

    structure = ndi.generate_binary_structure(3, 1)
    labels, _ = ndi.label(mask, structure=structure)
    labels = labels.astype(np.int32, copy=False)

    if int(args.min_size) > 1 and int(labels.max()) > 0:
        counts = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
        keep = counts >= int(args.min_size)
        keep[0] = False
        labels = np.where(keep[labels], labels, 0).astype(np.int32, copy=False)

    if int(labels.max()) > 0:
        unique = np.unique(labels)
        unique = unique[unique != 0]
        if unique.size:
            lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
            lut[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
            labels = lut[labels]
    return labels, used_threshold


def segment_group1(group1: np.ndarray, args) -> tuple[np.ndarray, float, str]:
    if args.segmentation == "gpu_connected" and args.use_gpu and GPU_AVAILABLE:
        labels, threshold = gpu_connected_segmentation(group1, args)
        return labels, threshold, "gpu"
    labels, threshold = cpu_connected_segmentation(group1, args)
    return labels, threshold, "cpu"


def measure_objects(labels: np.ndarray, group1: np.ndarray, group2: np.ndarray):
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


def object_bounding_boxes(labels: np.ndarray):
    return ndi.find_objects(labels)


def create_workbook():
    wb = Workbook(write_only=False)

    ws = wb.active
    ws.title = "Object Results"
    ws.append([
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
    ])

    summary = wb.create_sheet("Image Summary")
    summary.append([
        "Image",
        "Group1_File",
        "Group2_File",
        "Threshold",
        "Segmentation_Device",
        "Object_Count",
        "Total_Object_Voxels",
        "Group1_Mean_Of_Object_Means",
        "Group2_Mean_Of_Object_Means",
        "Status",
    ])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for sheet in (ws, summary):
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    for col, width in {
        "A": 42, "B": 12, "C": 14, "D": 23, "E": 23,
        "F": 10, "G": 10, "H": 10, "I": 10, "J": 10, "K": 10,
    }.items():
        ws.column_dimensions[col].width = width

    for col, width in {
        "A": 42, "B": 48, "C": 48, "D": 14, "E": 18,
        "F": 14, "G": 20, "H": 28, "I": 28, "J": 18,
    }.items():
        summary.column_dimensions[col].width = width

    return wb, ws, summary


def main() -> int:
    args = get_settings()
    root = args.root.expanduser().resolve()
    group1_dir = root / args.group1_folder
    group2_dir = root / args.group2_folder
    output_path = Path(args.output).expanduser().resolve()

    if int(args.min_size) < 1:
        raise ValueError("min_size must be >= 1")
    if float(args.gaussian_sigma) < 0:
        raise ValueError("gaussian_sigma must be >= 0")
    if int(args.closing_iterations) < 0:
        raise ValueError("closing_iterations must be >= 0")
    if not 0.0 <= float(args.overlay_alpha) <= 1.0:
        raise ValueError("overlay_alpha must be between 0.0 and 1.0")

    pairs = find_pairs(group1_dir, group2_dir, args.group1_suffix, args.group2_suffix)
    print(f"Found {len(pairs)} matched image pair(s).")
    print(f"Requested segmentation: {args.segmentation}")
    print(f"CuPy available: {GPU_AVAILABLE}")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dir = output_path.parent / "labels"
    if args.save_labels or args.save_colored_labels or args.save_overlay:
        labels_dir.mkdir(parents=True, exist_ok=True)

    wb, results_ws, summary_ws = create_workbook()

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
                raise ValueError(f"Shape mismatch: Group 1 {group1.shape} vs Group 2 {group2.shape}")

            print(f"  Shape (Z,Y,X): {group1.shape}")
            print(f"  Group 1 dtype: {group1.dtype}; Group 2 dtype: {group2.dtype}")

            labels, used_threshold, device_used = segment_group1(group1, args)
            print(f"  Segmentation device: {device_used}")
            print(f"  Threshold: {used_threshold:.4f}")
            print(f"  Objects found: {int(labels.max())}")

            if args.remove_small_objects_after_labeling and int(labels.max()) > 0:
                labels = remove_small_objects(labels > 0, min_size=int(args.min_size), connectivity=1)
                labels, _ = ndi.label(labels, structure=ndi.generate_binary_structure(3, 1))
                labels = labels.astype(np.int32, copy=False)

            n_objects = int(labels.max())
            if n_objects == 0:
                summary_ws.append([
                    pair_name,
                    g1_path.name,
                    g2_path.name,
                    used_threshold,
                    device_used,
                    0,
                    0,
                    None,
                    None,
                    "No objects detected",
                ])
                del group1, group2, labels
                gc.collect()
                continue

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
                tifffile.imwrite(
                    label_path,
                    labels,
                    photometric="minisblack",
                    metadata={"axes": "ZYX"},
                    compression="zlib",
                )

            if args.save_colored_labels or args.save_overlay:
                colored_labels = make_label_colors(labels)

                if args.save_colored_labels:
                    color_path = labels_dir / f"{pair_name}_labels_COLOR.tif"
                    tifffile.imwrite(
                        color_path,
                        colored_labels,
                        photometric="rgb",
                        metadata={"axes": "ZYXC"},
                        compression="zlib",
                    )

                if args.save_overlay:
                    overlay = make_overlay(group1, colored_labels, labels, float(args.overlay_alpha))
                    overlay_path = labels_dir / f"{pair_name}_labels_OVERLAY.tif"
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
                device_used,
                n_objects,
                int(counts[1:].sum()),
                float(np.mean(mean_g1[1:][valid])) if np.any(valid) else None,
                float(np.mean(mean_g2[1:][valid])) if np.any(valid) else None,
                "OK",
            ])

            total_objects += n_objects
            print(f"  Completed: {n_objects} object(s).")

            del group1, group2, labels, counts, sum_g1, sum_g2, mean_g1, mean_g2, boxes
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
                "gpu" if GPU_AVAILABLE and args.use_gpu else "cpu",
                None,
                None,
                None,
                None,
                f"ERROR: {exc}",
            ])
            gc.collect()

    for row in results_ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for cell in row:
            cell.number_format = "0.0000"

    for row in summary_ws.iter_rows(min_row=2):
        if row[3].value is not None:
            row[3].number_format = "0.0000"
        if row[7].value is not None:
            row[7].number_format = "0.0000"
        if row[8].value is not None:
            row[8].number_format = "0.0000"

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