"""
Cellpose-SAM / PyTorch GPU version of the paired 3D cell analysis workflow.

Group 1:
    segmented in 3D with Cellpose-SAM using PyTorch/CUDA.

Group 2:
    measured using the exact same Object_ID masks generated from Group 1.

Outputs:
    - One Excel workbook containing every object from every matched image pair
    - Raw Object_ID 3D TIFF
    - Colored label 3D TIFF
    - Colored overlay 3D TIFF

IMPORTANT:
    Edit only the USER SETTINGS section below.

Recommended environment:
    Python 3.10+
    NVIDIA GPU + a PyTorch build with CUDA support

Core packages:
    pip install cellpose tifffile numpy scipy openpyxl tqdm

Cellpose 4 uses PyTorch internally. The model selected below defaults to
"cpsam_v2", the current Cellpose-SAM model family.
"""

from __future__ import annotations

from pathlib import Path
import gc
import sys
import traceback
import inspect
import time
import platform
import atexit
import re
import os
import subprocess

import numpy as np
import tifffile
from scipy import ndimage as ndi
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

try:
    from cellpose import models
except Exception:
    models = None


TIFF_EXTENSIONS = {".tif", ".tiff"}



# ===================================================================
# WINDOWS ANTI-SLEEP
# ===================================================================

# Windows execution-state flags.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def enable_anti_sleep(keep_display_on=False):
    """
    Prevent Windows from automatically sleeping while this Python process
    is running.

    keep_display_on=False:
        Keeps the computer/system awake, but still allows the monitor to turn
        off normally.

    keep_display_on=True:
        Keeps both the computer and display awake.

    On non-Windows systems this function safely does nothing.
    """
    if platform.system() != "Windows":
        print("Anti-sleep: non-Windows system detected; Windows anti-sleep skipped.")
        return False

    try:
        import ctypes

        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED

        if keep_display_on:
            flags |= ES_DISPLAY_REQUIRED

        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)

        if result == 0:
            raise OSError("SetThreadExecutionState returned 0.")

        if keep_display_on:
            print("Anti-sleep enabled: computer and display will stay awake.")
        else:
            print("Anti-sleep enabled: computer will stay awake; display may turn off.")

        return True

    except Exception as error:
        print(f"WARNING: could not enable Windows anti-sleep: {error}")
        return False


def disable_anti_sleep():
    """
    Restore normal Windows sleep behavior.
    """
    if platform.system() != "Windows":
        return

    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS
        )

        print("Anti-sleep disabled: normal Windows sleep behavior restored.")

    except Exception as error:
        print(f"WARNING: could not restore Windows sleep settings: {error}")

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
    # INPUT FOLDERS
    # ---------------------------------------------------------------

    # Add every experiment root that should be processed, in run order.
    # Each root must contain the relative group_1 and group_2 folders below.
    # One complete Excel workbook and QC-output folder are created per root.
    args.experiment_roots = [
        Path(r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_12/Raw/Split"),
        # Path(r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_13/Raw/Split"),
        # Path(r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_08_14/Raw/Split"),
    ]

    # The batch controller replaces this automatically for each root.
    args.root = args.experiment_roots[0]

    # ---------------------------------------------------------------
    # ANTI-SLEEP SETTINGS
    # ---------------------------------------------------------------

    # True = prevent Windows from automatically putting the computer to sleep
    # while this analysis script is running.
    args.prevent_sleep = True

    # False = computer stays awake, but monitor is still allowed to turn off.
    # True  = keep both computer AND monitor awake.
    #
    # False is recommended for long overnight analyses.
    args.keep_display_on = False

    args.group1_folder = "group_1"
    args.group2_folder = "group_2"

    # Expected matching filenames:
    # sample001_group1.tif
    # sample001_group2.tif
    args.group1_suffix = "_group1"
    args.group2_suffix = "_group2"

    # ---------------------------------------------------------------
    # EXPERIMENT GROUP NAME MATCHING
    # ---------------------------------------------------------------
    # Search each image name for one of these underscore-separated tokens.
    # Example image: 2026_08_12_Cos7_Gwiz_2BS_Tam_1_003 -> Group = Tam
    args.group_names = ["NP", "UM", "JQ1", "Tam", "SV40", "VP16"]

    # Saved when no configured group name is found.
    args.unmatched_group_name = "Unmatched"

    # Stop if a filename contains more than one configured group.
    args.error_on_multiple_group_matches = True

    # ---------------------------------------------------------------
    # OUTPUT LOCATION
    # ---------------------------------------------------------------

    # Each experiment saves into this subfolder inside its own root.
    args.output_subfolder_name = "CellposeSAM_Results"
    args.output_folder = args.root / args.output_subfolder_name

    args.output_filename = "3D_cell_intensity_results_CellposeSAM.xlsx"
    args.output = args.output_folder / args.output_filename

    # When two or more roots are configured, a small batch summary is saved
    # beside the first experiment root.
    args.batch_summary_filename = "CellposeSAM_Batch_Summary.xlsx"

    # ---------------------------------------------------------------
    # CELLPOSE-SAM MODEL
    # ---------------------------------------------------------------

    # Current Cellpose-SAM model.
    args.pretrained_model = "cpsam_v2"

    # Use NVIDIA CUDA GPU when available.
    args.use_gpu = True

    # GPU index. Usually 0.
    args.gpu_device = 0

    # bfloat16 lowers GPU memory use on supported GPUs.
    args.use_bfloat16 = True

    # ---------------------------------------------------------------
    # TRUE 3D SEGMENTATION
    # ---------------------------------------------------------------

    # True = Cellpose performs 3D segmentation.
    args.do_3D = True

    # TIFFs in this workflow are expected as:
    # Z x Y x X
    args.z_axis = 0

    # None because Group 1 is a single-channel 3D stack.
    args.channel_axis = None

    # ---------------------------------------------------------------
    # VOXEL ANISOTROPY
    # ---------------------------------------------------------------

    # Set this if Z spacing differs from XY pixel spacing.
    #
    # anisotropy = Z voxel size / XY pixel size
    #
    # Example:
    # XY = 0.25 um/pixel
    # Z  = 1.00 um/plane
    # anisotropy = 1.00 / 0.25 = 4.0
    #
    # 63x
    # XY = 0.1801953125 um/pixel
    # Z  = 0.30 um/pixel
    # anisotropy = 0.30 / 0.1801953125 = 1.6648601777585085627574246694125
    # Leave None if voxels are close to isotropic or spacing is unknown.
    args.anisotropy = 1.6648601777585085627574246694125

    # ---------------------------------------------------------------
    # CELL SIZE
    # ---------------------------------------------------------------

    # Cellpose-SAM is relatively robust to cell size.
    #
    # None = do not force a diameter.
    #
    # If cells are much larger than the model's typical scale, you can try
    # a value such as 40, 60, 80, etc.
    args.diameter = None

    # ---------------------------------------------------------------
    # SEGMENTATION SENSITIVITY
    # ---------------------------------------------------------------

    # Cell probability threshold.
    #
    # LOWER:
    #   finds more / larger masks
    #
    # HIGHER:
    #   is more conservative and can reduce weak false-positive regions
    #
    # Good starting point:
    args.cellprob_threshold = 0.0

    # Flow consistency threshold.
    #
    # This is primarily a mask-quality filter.
    args.flow_threshold = 0.4

    # Remove Cellpose objects smaller than this many voxels/pixels.
    args.min_size = 200

    # Maximum fraction of the entire image allowed for one mask.
    # Useful for rejecting catastrophic giant masks.
    args.max_size_fraction = 0.4

    # Number of Cellpose dynamics iterations.
    # None = automatic.
    # Increase for very large/long cells if needed.
    args.niter = None

    # ---------------------------------------------------------------
    # 3D FRAGMENTATION / OVERSEGMENTATION CONTROLS
    # ---------------------------------------------------------------

    # Smooth Cellpose flows in Z, Y, X before final 3D mask generation.
    #
    # This can help with fragmentation along Z.
    #
    # Examples:
    # args.flow3D_smooth = 0
    # args.flow3D_smooth = 1
    # args.flow3D_smooth = [2, 1, 1]   # stronger smoothing in Z
    args.flow3D_smooth = [2, 1, 1]

    # ---------------------------------------------------------------
    # INPUT NORMALIZATION
    # ---------------------------------------------------------------

    # Cellpose normalization.
    args.normalize = True

    # If cells are dark on a bright background, change to True.
    args.invert = False

    # ---------------------------------------------------------------
    # GPU BATCH / MEMORY
    # ---------------------------------------------------------------

    # Number of tiles evaluated at once.
    #
    # If GPU runs out of memory:
    #   reduce to 4, 2, or 1.
    #
    # If you have lots of VRAM:
    #   try 16 or 32.
    args.batch_size = 128

    # Tile overlap.
    args.tile_overlap = 0.1

    # For SAM models this should remain 256.
    args.bsize = 256

    # Test-time augmentation.
    # False is faster and recommended initially.
    args.augment = False

    # ---------------------------------------------------------------
    # OPTIONAL POST-PROCESSING
    # ---------------------------------------------------------------

    # Fill completely enclosed holes inside each final Cellpose object.
    args.fill_holes_after_cellpose = True

    # Maximum hole size to fill, in voxels.
    #
    # 0 means use complete binary hole filling inside each object.
    # Current implementation uses enclosed-hole filling.
    args.max_hole_size = 0

    # ---------------------------------------------------------------
    # NUCLEAR / CYTOSOL COMPARTMENTS
    # ---------------------------------------------------------------
    # Group 1 Cellpose objects are treated as nuclei.
    # Choose how simulated cytosol is generated:
    #   "until_touching" = expand throughout the image to the nearest nucleus.
    #   "fixed_distance" = expand only the requested physical distance.
    # In both modes, every background voxel can belong to at most one nucleus,
    # so expansions stop at the nearest-neighbor boundary and never overlap.
    args.cytosol_expansion_mode = "until_touching"

    # Used only when cytosol_expansion_mode == "fixed_distance".
    # Expansion distance outward from the nuclear surface, in microns.
    args.cytosol_expansion_distance_um = 2.0

    # IMPORTANT: change these to your microscope calibration.
    args.xy_pixel_size_um = 184.52/1024
    args.z_spacing_um = 0.30

    # ---------------------------------------------------------------
    # GROUP 2 PUNCTA / HUB DETECTION
    # ---------------------------------------------------------------
    args.detect_group2_puncta = True

    # Difference-of-Gaussians spot sizes in microns.
    args.puncta_sigma_small_um = 0.35
    args.puncta_sigma_large_um = 0.80

    # Higher = fewer/brighter puncta; lower = more/dimmer puncta.
    args.puncta_threshold_std = 3.0

    # None = automatic robust threshold.
    args.puncta_fixed_threshold = None

    # ---------------------------------------------------------------
    # LOW-SIGNAL / EMPTY GROUP 2 PROTECTION
    # ---------------------------------------------------------------

    # Skip puncta detection when Group 2 contains too little usable signal.
    args.skip_low_signal_images = True

    # If this fraction (or more) of Group 2 voxels are exactly zero, the
    # image is considered low-signal UNLESS it still contains sufficiently
    # strong raw signal according to the max/std safeguards below.
    args.max_zero_fraction = 0.96

    # Absolute raw Group 2 safeguards. Tune these for your image intensity
    # scale. For uint8 images, values such as 3-10 are reasonable starting
    # points. Set either to 0 to disable that individual safeguard.
    args.min_group2_max_intensity = 5.0
    args.min_group2_std = 1.0

    # Every punctum candidate must also contain raw Group 2 signal at or above
    # this value. This prevents tiny numerical/noise fluctuations near zero
    # from becoming puncta. Set to 0 to disable.
    args.puncta_min_raw_intensity = 15

    # Hard lower bound for the DoG threshold. This prevents the automatic
    # threshold from collapsing toward zero in nearly empty images.
    # Set to 0 to disable.
    args.puncta_min_dog_threshold = 0.05

    args.puncta_min_volume_um3 = 0.05

    # Detection-stage maximum volume. Components larger than this are removed
    # from the puncta mask itself. None = no detection-stage maximum.
    args.puncta_max_volume_um3 = 25.0

    # Export-stage maximum punctum volume in um^3. A detected punctum larger
    # than this is NOT included in Excel, per-cell puncta summaries, or counts.
    # It can still remain visible in the puncta QC mask/overlay so you can see
    # what was detected but rejected from quantitative export. None = disabled.
    args.puncta_export_max_volume_um3 = 25.0

    # ---------------------------------------------------------------
    # PUNCTA 3D SHAPE FILTERING
    # ---------------------------------------------------------------
    # Bright connected components can be rejected when their 3D geometry is
    # too elongated/flat to look punctum-like. Shape is calculated in physical
    # microns, so the different Z and XY voxel sizes are accounted for.
    args.puncta_shape_filter = True

    # Maximum PCA principal-axis aspect ratio (longest / shortest).
    # 1.0 is sphere-like; larger values are increasingly elongated/flat.
    # None disables this individual criterion.
    args.puncta_max_aspect_ratio = 2.00

    # PCA-based 3D sphericity/isotropy score = sqrt(lambda_min/lambda_max).
    # 1.0 is sphere-like; values approaching 0 are elongated or sheet-like.
    # This is intentionally a voxel/PCA metric and does not require scikit-image.
    # None disables this individual criterion.
    args.puncta_min_shape_sphericity = 0.40

    # Very tiny components do not contain enough voxels for stable 3D shape.
    # Components below this size bypass shape rejection but still obey the
    # normal puncta minimum-volume threshold.
    args.puncta_shape_min_voxels = 4

    # ---------------------------------------------------------------
    # GROUP 2 DIFFUSE SIGNAL DETECTION
    # ---------------------------------------------------------------
    # Detect broad Group 2 signal independently of puncta. Diffuse-positive
    # voxels must exceed a locally estimated background and are then stripped
    # of all voxels belonging to detected puncta.
    args.detect_group2_diffuse = True

    # Gaussian scale (microns) used to estimate broad local background.
    # This should be larger than the puncta scale.
    args.diffuse_background_sigma_um = 2.0

    # Robust threshold above the residual/background distribution.
    args.diffuse_threshold_std = 1.5

    # Hard lower limit for the automatically calculated, background-subtracted
    # diffuse threshold. This prevents low-noise/negative-control images from
    # producing a threshold close to zero and classifying background as signal.
    #
    # The effective threshold is:
    #   max(automatic_threshold, diffuse_min_residual_threshold)
    #
    # This is applied to (raw Group 2 - estimated local background), not to the
    # raw image. Tune this value using the residual/noise range in your negative
    # controls. Set to 0 to disable the hard floor.
    args.diffuse_min_residual_threshold = 5.0

    # Optional absolute raw-intensity floor. Set to 0 to disable.
    args.diffuse_min_raw_intensity = 0.05

    # Remove very small diffuse-positive islands. Physical units are used so
    # behavior remains consistent when voxel calibration changes.
    args.diffuse_min_volume_um3 = 0.25

    # ---------------------------------------------------------------
    # PLASMID COMPARISON METRICS
    # ---------------------------------------------------------------
    # A separate "Plasmid Metrics" worksheet reports background-corrected
    # uptake, nuclear localization, puncta organization, diffuse fractions,
    # and punctum-to-nucleus distance summaries for every Object_ID.

    # None = estimate one image-level Group 2 background value from the
    # configured percentile. Otherwise, use this fixed raw-intensity value for
    # every image. A fixed value is most comparable when acquisition settings
    # and negative-control background are stable across the experiment.
    args.plasmid_fixed_background_intensity = None

    # Percentile of the raw Group 2 stack used when the fixed value is None.
    # This preserves the TIFF's native intensity scale (0-1, uint8, uint16...).
    args.plasmid_background_percentile = 20.0

    # Reference volume used for compartment-normalized puncta density.
    args.puncta_density_reference_um3 = 100.0

    # Distance bands used for per-object punctum localization summaries.
    args.perinuclear_distance_um = 1.0
    args.deep_nuclear_distance_um = 1.0

    # Split a hub at the nuclear boundary so nuclear and cytosolic portions
    # are measured separately.
    args.split_puncta_at_nuclear_boundary = True

    # Extra QC outputs.
    args.save_compartment_labels = False
    args.save_puncta_labels = False
    args.save_puncta_overlay = True

    # ---------------------------------------------------------------
    # FAST PROCESSING SETTINGS
    # ---------------------------------------------------------------
    # Production mode: skip expensive medians and optionally skip QC TIFFs.
    args.fast_mode = True
    args.calculate_median = False
    args.compress_output_tiffs = False

    # Set these False for maximum batch speed after segmentation is tuned.
    args.save_colored_labels = False
    args.save_overlay = True
    args.save_compartment_labels = False
    args.save_puncta_overlay = True

    # ---------------------------------------------------------------
    # EXCEL LARGE-DATA SAFETY
    # ---------------------------------------------------------------

    # Excel allows at most 1,048,576 rows per worksheet.
    # Use a lower rollover point so there is always room for the header.
    args.excel_rows_per_puncta_sheet = 1_000_000

    # Save a checkpoint workbook every N completed images.
    # Set to 0 to disable checkpoint saves.
    args.save_excel_every_n_images = 5

    # Checkpoint file is overwritten each time and removed after a successful
    # final save.
    args.checkpoint_filename = "3D_cell_intensity_results_CHECKPOINT.xlsx"

    # ---------------------------------------------------------------
    # OUTPUT IMAGES
    # ---------------------------------------------------------------

    # Raw integer Object_ID TIFF.
    # This is the authoritative label image.
    args.save_labels = False

    # Easy-to-see RGB label TIFF.
    args.save_colored_labels = False

    # Colored labels over original Group 1.
    args.save_overlay = True

    # Overlay opacity.
    args.overlay_alpha = 0.45

    return args


def check_dependencies(args):
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Install a PyTorch build appropriate "
            "for your computer/GPU before running this script."
        )

    if models is None:
        raise ImportError(
            "Cellpose is not installed. Install it with:\n"
            "pip install cellpose"
        )

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")

    if args.use_gpu:
        if torch.cuda.is_available():
            gpu_index = int(args.gpu_device)
            if gpu_index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested GPU {gpu_index}, but PyTorch sees only "
                    f"{torch.cuda.device_count()} CUDA GPU(s)."
                )
            print(f"Using CUDA GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
        else:
            print(
                "WARNING: use_gpu=True, but CUDA is not available to PyTorch.\n"
                "Cellpose will be initialized without CUDA acceleration."
            )


def normalize_pair_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if not stem.lower().endswith(suffix.lower()):
        raise ValueError(
            f'Filename "{path.name}" does not end with expected suffix "{suffix}".'
        )
    return stem[:-len(suffix)]



def find_group_name(pair_name, args):
    """Return the configured group token found in the image name."""
    tokens = {
        token.casefold()
        for token in str(pair_name).split("_")
        if token
    }

    matches = [
        str(group)
        for group in args.group_names
        if str(group).casefold() in tokens
    ]

    if len(matches) > 1:
        message = (
            f'Image "{pair_name}" matched multiple groups: '
            + ", ".join(matches)
        )
        if args.error_on_multiple_group_matches:
            raise ValueError(message)
        tqdm.write("WARNING: " + message + f". Using {matches[0]}.")

    return matches[0] if matches else str(args.unmatched_group_name)

def find_pairs(group1_dir, group2_dir, group1_suffix, group2_suffix):
    if not group1_dir.is_dir():
        raise FileNotFoundError(f"Group 1 folder not found: {group1_dir}")

    if not group2_dir.is_dir():
        raise FileNotFoundError(f"Group 2 folder not found: {group2_dir}")

    def collect(folder, suffix):
        result = {}

        for path in sorted(folder.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in TIFF_EXTENSIONS
                and path.stem.lower().endswith(suffix.lower())
            ):
                pair_name = normalize_pair_name(path, suffix)

                if pair_name in result:
                    raise ValueError(
                        f'Duplicate pair name "{pair_name}" in {folder}'
                    )

                result[pair_name] = path

        return result

    group1 = collect(group1_dir, group1_suffix)
    group2 = collect(group2_dir, group2_suffix)

    if not group1:
        raise FileNotFoundError(
            f'No TIFF files ending with "{group1_suffix}" found in {group1_dir}'
        )

    missing_group2 = sorted(set(group1) - set(group2))
    missing_group1 = sorted(set(group2) - set(group1))

    if missing_group2 or missing_group1:
        messages = []

        if missing_group2:
            messages.append(
                "Missing Group 2 matches: " + ", ".join(missing_group2)
            )

        if missing_group1:
            messages.append(
                "Missing Group 1 matches: " + ", ".join(missing_group1)
            )

        raise ValueError("\n".join(messages))

    return [
        (name, group1[name], group2[name])
        for name in sorted(group1)
    ]


def open_stack(path):
    try:
        image = tifffile.memmap(path)
    except Exception:
        image = tifffile.imread(path)

    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim != 3:
        raise ValueError(
            f"{path.name}: expected a 3D TIFF stack, got shape {image.shape}"
        )

    return image


def create_cellpose_model(args):
    """
    Create the Cellpose-SAM model once and reuse it for every image.

    Cellpose expects a torch.device object, not an integer GPU index.
    """

    print()
    print("Loading Cellpose-SAM model...")
    print(f"Requested model: {args.pretrained_model}")

    if args.use_gpu and torch.cuda.is_available():
        gpu_index = int(args.gpu_device)

        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu_index} was requested, but PyTorch sees only "
                f"{torch.cuda.device_count()} CUDA GPU(s)."
            )

        device = torch.device(f"cuda:{gpu_index}")

        print(
            f"Using GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)}"
        )
    else:
        device = torch.device("cpu")

        if args.use_gpu:
            print(
                "WARNING: GPU requested, but CUDA is unavailable. "
                "Using CPU."
            )
        else:
            print("Using CPU.")

    model_kwargs = {
        "gpu": bool(args.use_gpu and torch.cuda.is_available()),
        "pretrained_model": args.pretrained_model,
        "device": device,
        "use_bfloat16": bool(args.use_bfloat16),
    }

    # Filter constructor arguments too, because Cellpose releases can differ.
    constructor_parameters = inspect.signature(
        models.CellposeModel
    ).parameters

    filtered_model_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in constructor_parameters
    }

    ignored = sorted(
        set(model_kwargs) - set(filtered_model_kwargs)
    )

    if ignored:
        print(
            "Cellpose constructor does not support these settings "
            f"in this installation; ignoring: {', '.join(ignored)}"
        )

    model = models.CellposeModel(
        **filtered_model_kwargs
    )

    print(f"Cellpose device: {getattr(model, 'device', device)}")

    return model


def run_cellpose_3d(model, image, args):
    """
    Run Cellpose-SAM 3D segmentation.

    This function inspects the installed CellposeModel.eval() signature and
    sends only arguments supported by that exact Cellpose installation. This
    avoids errors such as:
        unexpected keyword argument 'invert'
    """

    tqdm.write("    Preparing Cellpose-SAM 3D segmentation...")

    eval_parameters = inspect.signature(
        model.eval
    ).parameters

    # Newer Cellpose versions may place invert inside the normalization
    # dictionary rather than accepting invert= directly.
    if "invert" in eval_parameters:
        normalization_setting = bool(args.normalize)
    else:
        normalization_setting = {
            "normalize": bool(args.normalize),
            "norm3D": True,
            "invert": bool(args.invert),
        }

    requested_kwargs = {
        "batch_size": int(args.batch_size),
        "channel_axis": args.channel_axis,
        "z_axis": args.z_axis,
        "normalize": normalization_setting,
        "invert": bool(args.invert),
        "diameter": args.diameter,
        "flow_threshold": float(args.flow_threshold),
        "cellprob_threshold": float(args.cellprob_threshold),
        "do_3D": bool(args.do_3D),
        "anisotropy": args.anisotropy,
        "min_size": int(args.min_size),
        "max_size_fraction": float(args.max_size_fraction),
        "niter": args.niter,
        "augment": bool(args.augment),
        "tile_overlap": float(args.tile_overlap),
        "bsize": int(args.bsize),
        "flow3D_smooth": args.flow3D_smooth,
        "progress": True,
    }

    eval_kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in eval_parameters
    }

    ignored = sorted(
        set(requested_kwargs) - set(eval_kwargs)
    )

    if ignored:
        tqdm.write(
            "    Installed Cellpose does not support these eval settings; "
            f"ignoring: {', '.join(ignored)}"
        )

    tqdm.write(
        "    Cellpose parameters being used: "
        + ", ".join(sorted(eval_kwargs))
    )

    start_time = time.perf_counter()

    result = model.eval(
        image,
        **eval_kwargs,
    )

    elapsed = time.perf_counter() - start_time

    tqdm.write(
        f"    Cellpose model.eval() finished in {elapsed:.1f} seconds."
    )

    # Different Cellpose releases can return different tuple lengths.
    # We only need the first return value: masks.
    if isinstance(result, tuple):
        masks = result[0]
    else:
        masks = result

    labels = np.asarray(
        masks,
        dtype=np.int32,
    )

    if labels.shape != image.shape:
        raise ValueError(
            f"Cellpose returned mask shape {labels.shape}, "
            f"but input image shape is {image.shape}."
        )

    return labels


def fill_holes_per_object(labels):
    """
    Fill enclosed holes while preserving Object_ID boundaries.

    Processing is done object-by-object within bounding boxes so that filling
    one cell does not merge it with a neighboring cell.
    """

    if int(labels.max()) == 0:
        return labels

    print("    Filling enclosed holes inside Cellpose objects...")

    output = labels.copy()
    boxes = ndi.find_objects(labels)

    for object_id, box in enumerate(boxes, start=1):
        if box is None:
            continue

        region = labels[box]
        object_mask = region == object_id

        filled = ndi.binary_fill_holes(object_mask)

        # Only add newly filled voxels where no other label currently exists.
        target = output[box]
        add = filled & (target == 0)
        target[add] = object_id

    return output


def relabel_consecutively(labels):
    unique = np.unique(labels)
    unique = unique[unique != 0]

    if unique.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)

    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[unique] = np.arange(
        1,
        unique.size + 1,
        dtype=np.int32,
    )

    return lut[labels]


def measure_objects(labels, group1, group2):
    """
    Measure voxel count and intensity sums one Z plane at a time to avoid
    unnecessary full-stack intensity copies.
    """

    n_objects = int(labels.max())

    counts = np.zeros(n_objects + 1, dtype=np.int64)
    group1_sum = np.zeros(n_objects + 1, dtype=np.float64)
    group2_sum = np.zeros(n_objects + 1, dtype=np.float64)

    for z in range(labels.shape[0]):
        lab = np.asarray(labels[z]).ravel()
        g1 = np.asarray(group1[z]).ravel()
        g2 = np.asarray(group2[z]).ravel()

        counts += np.bincount(
            lab,
            minlength=n_objects + 1,
        )

        group1_sum += np.bincount(
            lab,
            weights=g1,
            minlength=n_objects + 1,
        )

        group2_sum += np.bincount(
            lab,
            weights=g2,
            minlength=n_objects + 1,
        )

    return counts, group1_sum, group2_sum


def make_label_colors(labels):
    """
    Make deterministic RGB colors for visualization only.
    """

    n_objects = int(labels.max())

    palette = np.array([
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ], dtype=np.uint8)

    lookup = np.zeros(
        (n_objects + 1, 3),
        dtype=np.uint8,
    )

    if n_objects > 0:
        object_ids = np.arange(
            1,
            n_objects + 1,
        )

        lookup[1:] = palette[
            (object_ids - 1) % len(palette)
        ]

    return lookup[labels]


def normalize_group1_for_overlay(group1):
    """
    Percentile-normalize Group 1 to uint8 for visualization.
    """

    flat = group1.reshape(-1)
    step = max(
        1,
        flat.size // 1_000_000,
    )

    sample = np.asarray(
        flat[::step],
        dtype=np.float32,
    )

    low, high = np.percentile(
        sample,
        (1.0, 99.5),
    )

    if high <= low:
        low = float(np.min(sample))
        high = float(np.max(sample))

    if high <= low:
        return np.zeros(
            group1.shape,
            dtype=np.uint8,
        )

    output = np.empty(
        group1.shape,
        dtype=np.uint8,
    )

    scale = 255.0 / (high - low)

    for z in range(group1.shape[0]):
        plane = np.asarray(
            group1[z],
            dtype=np.float32,
        )

        plane = np.clip(
            (plane - low) * scale,
            0,
            255,
        )

        output[z] = plane.astype(
            np.uint8,
        )

    return output


def make_overlay(group1, colored_labels, labels, alpha):
    gray = normalize_group1_for_overlay(group1)

    overlay = np.repeat(
        gray[..., None],
        3,
        axis=-1,
    )

    del gray

    for z in range(labels.shape[0]):
        foreground = labels[z] > 0

        if not np.any(foreground):
            continue

        base = overlay[z]
        colors = colored_labels[z]

        blended = (
            (1.0 - alpha)
            * base[foreground].astype(np.float32)
            +
            alpha
            * colors[foreground].astype(np.float32)
        )

        base[foreground] = np.clip(
            blended,
            0,
            255,
        ).astype(np.uint8)

    return overlay



def voxel_volume_um3(args):
    return (
        float(args.xy_pixel_size_um)
        * float(args.xy_pixel_size_um)
        * float(args.z_spacing_um)
    )


def cytosol_expansion_description(args):
    mode = str(args.cytosol_expansion_mode).strip().casefold()
    if mode == "fixed_distance":
        return f"Nearest nucleus up to {float(args.cytosol_expansion_distance_um):g} um"
    return "Nearest nucleus until touching"


def expand_nuclei(labels, args):
    """Expand nuclei in physical 3D space without allowing overlap.

    Every background voxel is first assigned to its nearest nuclear voxel.
    In fixed-distance mode, that assignment is retained only when the voxel is
    within the requested distance of a nucleus. This means two expanding cells
    can meet at their nearest-neighbor boundary but can never overlap.
    """
    if int(labels.max()) == 0:
        empty = np.zeros(labels.shape, dtype=np.int32)
        return empty, empty.copy()

    mode = str(args.cytosol_expansion_mode).strip().casefold()
    if mode == "fixed_distance":
        max_distance = float(args.cytosol_expansion_distance_um)
        print(
            f"    Expanding nuclei by up to {max_distance:g} um "
            "without overlapping neighboring territories..."
        )
    else:
        max_distance = None
        print("    Expanding nuclei until neighboring territories touch...")

    background = labels == 0
    distances, indices = ndi.distance_transform_edt(
        background,
        sampling=(
            float(args.z_spacing_um),
            float(args.xy_pixel_size_um),
            float(args.xy_pixel_size_um),
        ),
        return_indices=True,
    )

    nearest = labels[indices[0], indices[1], indices[2]]

    if max_distance is None:
        whole = nearest.astype(np.int32, copy=True)
    else:
        whole = np.zeros(labels.shape, dtype=np.int32)
        grow = background & (distances <= max_distance)
        whole[grow] = nearest[grow]

    # Preserve the original nuclear labels exactly.
    whole[labels > 0] = labels[labels > 0]

    cytosol = whole.copy()
    cytosol[labels > 0] = 0
    return whole, cytosol


def region_stats(region_labels, image, calculate_median=False):
    """Fast per-object statistics; median is optional."""
    n = int(region_labels.max())
    count = np.zeros(n + 1, dtype=np.int64)
    total = np.zeros(n + 1, dtype=np.float64)
    total2 = np.zeros(n + 1, dtype=np.float64)

    for z in range(region_labels.shape[0]):
        lab = region_labels[z].ravel()
        val = np.asarray(image[z], dtype=np.float64).ravel()
        count += np.bincount(lab, minlength=n + 1)
        total += np.bincount(lab, weights=val, minlength=n + 1)
        total2 += np.bincount(lab, weights=val * val, minlength=n + 1)

    mean = np.divide(total, count, out=np.full(n + 1, np.nan), where=count > 0)
    var = np.divide(total2, count, out=np.full(n + 1, np.nan), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    median = np.full(n + 1, np.nan)
    minimum = np.full(n + 1, np.nan)
    maximum = np.full(n + 1, np.nan)

    if n:
        ids = np.arange(1, n + 1)
        minimum[1:] = ndi.minimum(image, labels=region_labels, index=ids)
        maximum[1:] = ndi.maximum(image, labels=region_labels, index=ids)

    if calculate_median:
        for oid, box in enumerate(ndi.find_objects(region_labels), start=1):
            if box is None:
                continue
            mask = region_labels[box] == oid
            if np.any(mask):
                median[oid] = np.median(np.asarray(image[box])[mask])

    return {
        "count": count, "sum": total, "mean": mean, "median": median,
        "std": std, "min": minimum, "max": maximum,
    }


def group2_signal_qc(group2, args):
    """
    Fast image-level Group 2 signal quality check.

    Returns a dictionary used both for puncta gating and Excel reporting.
    Sampling is used for standard deviation on very large stacks, while
    zero fraction and maximum are calculated exactly.
    """
    total_voxels = int(group2.size)
    nonzero_voxels = int(np.count_nonzero(group2))
    zero_fraction = (
        1.0 - (nonzero_voxels / total_voxels)
        if total_voxels > 0
        else 1.0
    )

    max_intensity = float(np.max(group2)) if total_voxels else 0.0
    mean_intensity = float(np.mean(group2)) if total_voxels else 0.0

    flat = group2.reshape(-1)
    step = max(1, flat.size // 2_000_000)
    sample = np.asarray(flat[::step], dtype=np.float32)
    std_intensity = float(np.std(sample)) if sample.size else 0.0

    too_many_zeros = (
        zero_fraction >= float(args.max_zero_fraction)
    )
    max_too_low = (
        float(args.min_group2_max_intensity) > 0
        and max_intensity < float(args.min_group2_max_intensity)
    )
    std_too_low = (
        float(args.min_group2_std) > 0
        and std_intensity < float(args.min_group2_std)
    )

    # A high zero fraction alone does not reject an image if it still has
    # convincingly strong sparse signal. This preserves truly sparse puncta.
    low_signal = (
        max_too_low
        or std_too_low
        or (
            too_many_zeros
            and max_too_low
        )
    )

    return {
        "zero_fraction": zero_fraction,
        "nonzero_fraction": 1.0 - zero_fraction,
        "mean": mean_intensity,
        "max": max_intensity,
        "std": std_intensity,
        "low_signal": bool(low_signal),
    }


def component_shape_metrics(component_mask, z_um, xy_um):
    """Return physical-coordinate PCA aspect ratio and isotropy/sphericity.

    The metric is rotation-invariant and uses calibrated Z/Y/X coordinates.
    Sphericity here is a PCA isotropy score, not mesh surface-area sphericity.
    """
    coords = np.argwhere(component_mask)
    if coords.shape[0] < 4:
        return np.nan, np.nan

    xyz = coords.astype(np.float64, copy=False)
    xyz[:, 0] *= float(z_um)
    xyz[:, 1] *= float(xy_um)
    xyz[:, 2] *= float(xy_um)
    xyz -= np.mean(xyz, axis=0, keepdims=True)

    cov = (xyz.T @ xyz) / max(1, xyz.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    largest = float(eig[-1])
    smallest = float(eig[0])

    if largest <= 0.0:
        return np.nan, np.nan
    if smallest <= 1e-15:
        return np.inf, 0.0

    aspect_ratio = float(np.sqrt(largest / smallest))
    shape_sphericity = float(np.sqrt(smallest / largest))
    return aspect_ratio, shape_sphericity


def detect_puncta(group2, whole_labels, nucleus_labels, args):
    """
    Fast puncta pipeline: one DoG, one 3D connected-component pass, vectorized
    size/intensity measurements, centroid-based cell/compartment assignment,
    and one global signed nuclear-distance map.
    """
    t0 = time.perf_counter()

    qc = group2_signal_qc(
        group2,
        args,
    )

    tqdm.write(
        "    Group 2 QC: "
        f"zero={100.0 * qc['zero_fraction']:.2f}%, "
        f"mean={qc['mean']:.4g}, "
        f"max={qc['max']:.4g}, "
        f"std={qc['std']:.4g}"
    )

    if (
        bool(args.skip_low_signal_images)
        and qc["low_signal"]
    ):
        tqdm.write(
            "    LOW SIGNAL: puncta detection skipped; reporting 0 puncta."
        )

        empty = np.zeros(
            group2.shape,
            dtype=np.int32,
        )

        rejected = np.zeros(group2.shape, dtype=np.uint8)
        return empty, [], np.nan, rejected

    z_um = float(args.z_spacing_um)
    xy_um = float(args.xy_pixel_size_um)
    s1 = float(args.puncta_sigma_small_um)
    s2 = float(args.puncta_sigma_large_um)
    sigma1 = (s1 / z_um, s1 / xy_um, s1 / xy_um)
    sigma2 = (s2 / z_um, s2 / xy_um, s2 / xy_um)

    tqdm.write("    Enhancing Group 2 puncta with 3D Difference-of-Gaussians...")
    ts = time.perf_counter()
    img = np.asarray(group2, dtype=np.float32)
    dog = ndi.gaussian_filter(img, sigma1, mode="nearest")
    dog -= ndi.gaussian_filter(img, sigma2, mode="nearest")
    del img
    tqdm.write(f"    DoG filtering: {time.perf_counter() - ts:.1f} s")

    valid = whole_labels > 0
    values = dog[valid]
    step = max(1, values.size // 2_000_000)
    sample = values[::step]

    if args.puncta_fixed_threshold is None:
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 0:
            robust_sigma = float(np.std(sample))
        threshold = med + float(args.puncta_threshold_std) * robust_sigma
    else:
        threshold = float(args.puncta_fixed_threshold)

    automatic_threshold = float(threshold)

    if float(args.puncta_min_dog_threshold) > 0:
        threshold = max(
            automatic_threshold,
            float(args.puncta_min_dog_threshold),
        )

    if threshold > automatic_threshold:
        tqdm.write(
            f"    Automatic DoG threshold {automatic_threshold:.6g} "
            f"raised to floor {threshold:.6g}"
        )
    else:
        tqdm.write(
            f"    Puncta DoG threshold: {threshold:.6g}"
        )

    candidate = (dog > threshold) & valid

    # Require an absolute raw-intensity floor as a second independent test.
    if float(args.puncta_min_raw_intensity) > 0:
        candidate &= (
            np.asarray(group2)
            >= float(args.puncta_min_raw_intensity)
        )
    del dog, values, sample, valid
    gc.collect()
    tqdm.write(f"    Thresholded puncta voxels: {np.count_nonzero(candidate):,}")

    ts = time.perf_counter()
    components, n0 = ndi.label(candidate, structure=ndi.generate_binary_structure(3, 1))
    del candidate
    sizes = np.bincount(components.ravel(), minlength=n0 + 1)

    vv = voxel_volume_um3(args)
    min_vox = max(1, int(np.ceil(float(args.puncta_min_volume_um3) / vv)))
    # Track why candidate components are excluded so the QC overlay can show them.
    # 1 = size/volume rejection, 2 = shape rejection, 3 = export-max-volume rejection.
    rejected_component_reason = np.zeros(n0 + 1, dtype=np.uint8)

    keep = sizes >= min_vox
    keep[0] = False
    rejected_component_reason[(np.arange(n0 + 1) != 0) & (sizes < min_vox)] = 1
    if args.puncta_max_volume_um3 is not None:
        max_vox = int(np.floor(float(args.puncta_max_volume_um3) / vv))
        too_large_detection = sizes > max_vox
        rejected_component_reason[too_large_detection & keep] = 1
        keep &= ~too_large_detection

    # Optional 3D physical-shape filter. This rejects broad elongated/sheet-like
    # bright regions before they become reported puncta.
    shape_metrics_by_component = {}
    shape_rejected = 0
    if bool(args.puncta_shape_filter) and np.any(keep):
        boxes = ndi.find_objects(components)
        min_shape_vox = max(1, int(args.puncta_shape_min_voxels))
        for cid in np.flatnonzero(keep):
            if cid == 0 or sizes[cid] < min_shape_vox:
                continue
            box = boxes[cid - 1]
            if box is None:
                continue
            component_mask = components[box] == cid
            aspect, sphericity = component_shape_metrics(
                component_mask, z_um, xy_um
            )
            shape_metrics_by_component[int(cid)] = (aspect, sphericity)

            reject = False
            if args.puncta_max_aspect_ratio is not None:
                reject |= (
                    np.isfinite(aspect)
                    and aspect > float(args.puncta_max_aspect_ratio)
                )
            if args.puncta_min_shape_sphericity is not None:
                reject |= (
                    np.isfinite(sphericity)
                    and sphericity < float(args.puncta_min_shape_sphericity)
                )
            if reject:
                keep[cid] = False
                rejected_component_reason[cid] = 2
                shape_rejected += 1

        tqdm.write(
            f"    3D shape filter rejected: {shape_rejected:,} component(s)"
        )

    kept = np.flatnonzero(keep)
    kept_shape_metrics = [
        shape_metrics_by_component.get(int(cid), (np.nan, np.nan))
        for cid in kept
    ]
    lut = np.zeros(n0 + 1, dtype=np.int32)
    lut[kept] = np.arange(1, kept.size + 1, dtype=np.int32)
    puncta_labels = lut[components]

    # Rejection overlay labels are categorical, not Object_ID labels.
    rejected_labels = rejected_component_reason[components]

    del components, sizes, keep, lut, rejected_component_reason
    gc.collect()

    n = int(puncta_labels.max())
    tqdm.write(f"    Initial components: {n0:,}; retained: {n:,}")
    tqdm.write(f"    Label + size filter: {time.perf_counter() - ts:.1f} s")

    if n == 0:
        return puncta_labels, [], threshold, rejected_labels

    ts = time.perf_counter()
    counts = np.bincount(puncta_labels.ravel(), minlength=n + 1)
    sums = np.zeros(n + 1, dtype=np.float64)
    for z in range(puncta_labels.shape[0]):
        sums += np.bincount(
            puncta_labels[z].ravel(),
            weights=np.asarray(group2[z]).ravel(),
            minlength=n + 1,
        )
    means = np.divide(sums, counts, out=np.full(n + 1, np.nan), where=counts > 0)
    ids = np.arange(1, n + 1)
    maxima = ndi.maximum(group2, labels=puncta_labels, index=ids)
    centers = ndi.center_of_mass(
        np.ones(puncta_labels.shape, dtype=np.uint8),
        labels=puncta_labels,
        index=ids,
    )

    nuclear_mask = nucleus_labels > 0
    outside = ndi.distance_transform_edt(
        ~nuclear_mask, sampling=(z_um, xy_um, xy_um)
    )
    inside = ndi.distance_transform_edt(
        nuclear_mask, sampling=(z_um, xy_um, xy_um)
    )

    records = []
    for pid, center in enumerate(centers, start=1):
        cz, cy, cx = center
        z = int(np.clip(round(cz), 0, puncta_labels.shape[0] - 1))
        y = int(np.clip(round(cy), 0, puncta_labels.shape[1] - 1))
        x = int(np.clip(round(cx), 0, puncta_labels.shape[2] - 1))
        cell_id = int(whole_labels[z, y, x])
        if cell_id == 0:
            continue

        if nucleus_labels[z, y, x] == cell_id:
            compartment = "Nuclear"
            distance = -float(inside[z, y, x])
        else:
            compartment = "Cytosolic"
            distance = float(outside[z, y, x])

        volume_um3 = float(counts[pid] * vv)

        # Optional export-only upper-volume filter. This intentionally happens
        # after detection so oversized candidates may remain visible in QC
        # overlays while being excluded from Excel and quantitative summaries.
        export_max = args.puncta_export_max_volume_um3
        if export_max is not None and volume_um3 > float(export_max):
            rejected_labels[puncta_labels == pid] = 3
            continue

        records.append({
            "Punctum_ID": pid,
            "Cell_Object_ID": cell_id,
            "Compartment": compartment,
            "Voxel_Count": int(counts[pid]),
            "Volume_um3": volume_um3,
            "Mean_Intensity": float(means[pid]),
            "Max_Intensity": float(maxima[pid - 1]),
            "Integrated_Intensity": float(sums[pid]),
            "Centroid_Z_vox": float(cz),
            "Centroid_Y_vox": float(cy),
            "Centroid_X_vox": float(cx),
            "Centroid_Z_um": float(cz * z_um),
            "Centroid_Y_um": float(cy * xy_um),
            "Centroid_X_um": float(cx * xy_um),
            "Distance_From_Nuclear_Surface_um": distance,
            "Shape_Aspect_Ratio_3D": float(kept_shape_metrics[pid - 1][0]),
            "Shape_Sphericity_PCA_3D": float(kept_shape_metrics[pid - 1][1]),
        })

    del outside, inside, nuclear_mask
    gc.collect()
    tqdm.write(f"    Puncta measurement/assignment: {time.perf_counter() - ts:.1f} s")
    tqdm.write(f"    Final Group 2 puncta/hubs: {len(records):,}")
    tqdm.write(f"    Puncta analysis total: {time.perf_counter() - t0:.1f} s")
    return puncta_labels, records, threshold, rejected_labels



def detect_diffuse_signal(
    group2,
    whole_labels,
    puncta_labels,
    rejected_puncta_labels,
    args,
):
    """Detect broad Group 2 signal while excluding only retained puncta.

    A large-scale Gaussian image estimates local background. The residual
    (raw - background) is robustly thresholded inside cell territories.
    Puncta retained for quantitative reporting are removed before tiny diffuse
    islands are rejected. Candidates rejected from puncta reporting remain
    eligible for diffuse classification if they pass the diffuse criteria.
    """
    t0 = time.perf_counter()
    z_um = float(args.z_spacing_um)
    xy_um = float(args.xy_pixel_size_um)
    sigma_um = float(args.diffuse_background_sigma_um)
    sigma = (sigma_um / z_um, sigma_um / xy_um, sigma_um / xy_um)

    tqdm.write("    Detecting diffuse Group 2 signal...")
    raw = np.asarray(group2, dtype=np.float32)
    background = ndi.gaussian_filter(raw, sigma=sigma, mode="nearest")
    residual = raw - background

    valid = whole_labels > 0
    values = residual[valid]
    if values.size == 0:
        return np.zeros(group2.shape, dtype=np.uint8), np.nan

    step = max(1, values.size // 2_000_000)
    sample = values[::step]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        robust_sigma = float(np.std(sample))
    automatic_threshold = (
        med
        + float(args.diffuse_threshold_std) * robust_sigma
    )


def estimate_plasmid_background(group2, args):
    """Return one raw-intensity background estimate for a Group 2 stack.

    Values are sampled for speed but are never rescaled. The returned value is
    therefore in the same native intensity units as the TIFF and all Group 2
    intensity measurements.
    """
    fixed = args.plasmid_fixed_background_intensity
    if fixed is not None:
        return float(fixed), "Fixed"

    percentile = float(args.plasmid_background_percentile)
    flat = np.asarray(group2).reshape(-1)
    step = max(1, flat.size // 2_000_000)
    sample = np.asarray(flat[::step], dtype=np.float32)
    if sample.size == 0:
        return 0.0, f"Percentile_{percentile:g}"

    background = float(np.percentile(sample, percentile))
    return background, f"Percentile_{percentile:g}"

    # Prevent the adaptive threshold from collapsing toward zero in images
    # containing little or no real Group 2 signal.
    threshold_floor = float(args.diffuse_min_residual_threshold)
    threshold = max(automatic_threshold, threshold_floor)

    if threshold > automatic_threshold:
        tqdm.write(
            f"    Automatic diffuse threshold {automatic_threshold:.6g} "
            f"raised to hard floor {threshold:.6g}"
        )
    else:
        tqdm.write(
            f"    Automatic diffuse threshold used: {threshold:.6g}"
        )

    diffuse = (residual > threshold) & valid
    if float(args.diffuse_min_raw_intensity) > 0:
        diffuse &= raw >= float(args.diffuse_min_raw_intensity)

    # Remove only puncta retained for quantitative reporting. Rejection class 3
    # marks puncta excluded by puncta_export_max_volume_um3. Those objects remain
    # eligible for diffuse classification. Candidates rejected earlier by size
    # or shape are already absent from puncta_labels and also remain eligible.
    retained_puncta = puncta_labels > 0
    if rejected_puncta_labels is not None:
        retained_puncta &= rejected_puncta_labels != 3
    diffuse &= ~retained_puncta

    min_volume = float(args.diffuse_min_volume_um3)
    if min_volume > 0 and np.any(diffuse):
        cc, n = ndi.label(diffuse, structure=ndi.generate_binary_structure(3, 1))
        sizes = np.bincount(cc.ravel(), minlength=n + 1)
        min_vox = max(1, int(np.ceil(min_volume / voxel_volume_um3(args))))
        keep = sizes >= min_vox
        keep[0] = False
        diffuse = keep[cc]
        del cc, sizes, keep

    tqdm.write(
        f"    Diffuse threshold: {threshold:.6g}; "
        f"positive voxels: {np.count_nonzero(diffuse):,}; "
        f"time: {time.perf_counter() - t0:.1f} s"
    )
    return diffuse.astype(np.uint8, copy=False), threshold


def diffuse_stats(diffuse_mask, region_labels, image, n_objects):
    """Per-object diffuse statistics while preserving every original Object_ID.

    ``region_stats`` normally sizes its output arrays from the maximum label that
    remains in ``region_labels``. After masking to diffuse-positive voxels, cells
    with no diffuse signal -- especially high-numbered Object_IDs -- may disappear.
    This function pads every returned statistic to ``n_objects + 1`` so all
    original Cellpose Object_IDs remain safe to index.
    """
    masked_labels = np.where(
        diffuse_mask > 0, region_labels, 0
    ).astype(np.int32, copy=False)

    measured = region_stats(masked_labels, image, False)
    target_size = int(n_objects) + 1

    stats = {}
    for key, values in measured.items():
        values = np.asarray(values)

        if key in {"count", "sum"}:
            fill_value = 0
        else:
            fill_value = np.nan

        padded = np.full(
            target_size,
            fill_value,
            dtype=values.dtype,
        )

        copy_size = min(values.size, target_size)
        padded[:copy_size] = values[:copy_size]
        stats[key] = padded

    return stats

def puncta_summary(records, n_objects):
    out = {}
    for oid in range(1, n_objects + 1):
        out[oid] = {}
        for c in ("Nuclear", "Cytosolic"):
            out[oid][c] = {
                "count": 0, "volume": 0.0, "total": 0.0,
                "means": [], "max": np.nan, "largest": 0.0,
            }

    for r in records:
        if r["Compartment"] not in ("Nuclear", "Cytosolic"):
            continue
        d = out[r["Cell_Object_ID"]][r["Compartment"]]
        d["count"] += 1
        d["volume"] += r["Volume_um3"]
        d["total"] += r["Integrated_Intensity"]
        d["means"].append(r["Mean_Intensity"])
        d["largest"] = max(d["largest"], r["Volume_um3"])
        d["max"] = (
            r["Max_Intensity"] if np.isnan(d["max"])
            else max(d["max"], r["Max_Intensity"])
        )
    return out


def make_compartment_rgb(nucleus_labels, cytosol_labels):
    rgb = np.zeros(nucleus_labels.shape + (3,), dtype=np.uint8)
    rgb[cytosol_labels > 0] = (0, 180, 220)
    rgb[nucleus_labels > 0] = (230, 60, 180)
    return rgb


def make_puncta_multichannel_qc(
    group2, nucleus_labels, cytosol_labels, puncta_labels,
    rejected_labels=None, diffuse_mask=None
):
    """Create a channel-separated QC stack instead of an RGB overlay.

    Returns uint8 data in T,Z,C,Y,X order for a Fiji/ImageJ hyperstack.
    T is a singleton time dimension. Fiji opens this as a Composite hyperstack
    with independent Channel and Z sliders rather than a flat sequential stack.

    Channels:
      1 Group2_Raw_Normalized
      2 Nucleus_Mask
      3 Cytosol_Mask
      4 Diffuse_Signal
      5 Accepted_Puncta
      6 Rejected_Size_Volume
      7 Rejected_3D_Shape
      8 Rejected_Export_Max_Volume
    """
    raw = normalize_group1_for_overlay(group2)
    shape = group2.shape
    channels = np.zeros((1, shape[0], 8, shape[1], shape[2]), dtype=np.uint8)

    channels[0, :, 0] = raw
    channels[0, :, 1] = (nucleus_labels > 0).astype(np.uint8) * 255
    channels[0, :, 2] = (cytosol_labels > 0).astype(np.uint8) * 255

    if diffuse_mask is not None:
        channels[0, :, 3] = (diffuse_mask > 0).astype(np.uint8) * 255

    # A punctum can remain in puncta_labels but be excluded from export by
    # rejection class 3, so keep that class out of the accepted channel.
    accepted = puncta_labels > 0
    if rejected_labels is not None:
        accepted &= rejected_labels != 3
    channels[0, :, 4] = accepted.astype(np.uint8) * 255

    if rejected_labels is not None:
        channels[0, :, 5] = (rejected_labels == 1).astype(np.uint8) * 255
        channels[0, :, 6] = (rejected_labels == 2).astype(np.uint8) * 255
        channels[0, :, 7] = (rejected_labels == 3).astype(np.uint8) * 255

    return channels


PUNCTA_HEADERS = [
    "Image",
    "Group",
    "Cell_Object_ID",
    "Punctum_ID",
    "Compartment",
    "Voxel_Count",
    "Volume_um3",
    "Mean_Intensity",
    "Max_Intensity",
    "Integrated_Intensity",
    "Centroid_Z_vox",
    "Centroid_Y_vox",
    "Centroid_X_vox",
    "Centroid_Z_um",
    "Centroid_Y_um",
    "Centroid_X_um",
    "Distance_From_Nuclear_Surface_um",
    "Shape_Aspect_Ratio_3D",
    "Shape_Sphericity_PCA_3D",
]


def format_sheet_header(sheet):
    """Apply the standard workbook header style."""
    header_fill = PatternFill(
        "solid",
        fgColor="1F4E78",
    )
    header_font = Font(
        color="FFFFFF",
        bold=True,
    )

    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center"
        )

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def excel_safe_sheet_name(name):
    """Return an Excel-safe worksheet-name fragment."""
    safe = re.sub(r'[:\\/?*\[\]]', "_", str(name)).strip().strip("'")
    return safe or "Unmatched"


def create_puncta_sheet(workbook, group_name, part_number):
    """Create a group-specific puncta sheet, with rollover parts if needed."""
    safe_group = excel_safe_sheet_name(group_name)
    base = f"Puncta {safe_group}" if part_number == 1 else f"Puncta {safe_group} {part_number}"
    title = base[:31]

    if title in workbook.sheetnames:
        suffix = f"_{part_number}"
        title = base[:31 - len(suffix)] + suffix

    sheet = workbook.create_sheet(title)
    sheet.append(PUNCTA_HEADERS)
    format_sheet_header(sheet)

    for column_cells in sheet.columns:
        sheet.column_dimensions[column_cells[0].column_letter].width = 18

    sheet.column_dimensions["A"].width = 42
    sheet.column_dimensions["B"].width = 16
    return sheet


def append_puncta_row(
    workbook,
    puncta_sheets_by_group,
    group_name,
    row_values,
    max_rows_per_sheet,
):
    """Append a punctum to its group's sheet and roll over if Excel fills up."""
    key = str(group_name)

    if key not in puncta_sheets_by_group:
        puncta_sheets_by_group[key] = [
            create_puncta_sheet(workbook, key, 1)
        ]

    sheets = puncta_sheets_by_group[key]
    current = sheets[-1]

    if current.max_row >= int(max_rows_per_sheet):
        current = create_puncta_sheet(
            workbook,
            key,
            len(sheets) + 1,
        )
        sheets.append(current)
        tqdm.write(
            f"    Excel puncta rollover for group '{key}': created '{current.title}'"
        )

    current.append(row_values)
    return current


def create_workbook():
    wb = Workbook()

    results = wb.active
    results.title = "Object Results"
    results.append([
        "Image", "Group", "Object_ID",
        "Nucleus_Voxel_Count", "Nucleus_Volume_um3",
        "Cytosol_Voxel_Count", "Cytosol_Volume_um3",
        "Whole_Cell_Voxel_Count", "Whole_Cell_Volume_um3",
        "Group1_Nuclear_Mean",
        "Group2_Nuclear_Mean", "Group2_Nuclear_Median",
        "Group2_Nuclear_StdDev", "Group2_Nuclear_Min",
        "Group2_Nuclear_Max", "Group2_Nuclear_Total",
        "Group2_Cytosol_Mean", "Group2_Cytosol_Median",
        "Group2_Cytosol_StdDev", "Group2_Cytosol_Min",
        "Group2_Cytosol_Max", "Group2_Cytosol_Total",
        "Group2_WholeCell_Mean",
        "Nuclear_to_Cytosol_Mean_Ratio",
        "Cytosol_to_Nuclear_Mean_Ratio",
        "Nuclear_Puncta_Count", "Nuclear_Puncta_Total_Volume_um3",
        "Nuclear_Puncta_Volume_Percent",
        "Nuclear_Puncta_Total_Intensity",
        "Nuclear_Puncta_Mean_Intensity",
        "Nuclear_Puncta_Max_Intensity",
        "Nuclear_Largest_Punctum_um3",
        "Cytosolic_Puncta_Count", "Cytosolic_Puncta_Total_Volume_um3",
        "Cytosolic_Puncta_Volume_Percent",
        "Cytosolic_Puncta_Total_Intensity",
        "Cytosolic_Puncta_Mean_Intensity",
        "Cytosolic_Puncta_Max_Intensity",
        "Cytosolic_Largest_Punctum_um3",
        "Nuclear_Diffuse_Voxel_Count", "Nuclear_Diffuse_Volume_um3",
        "Nuclear_Diffuse_Volume_Percent", "Nuclear_Diffuse_Mean_Intensity",
        "Nuclear_Diffuse_Total_Intensity",
        "Cytosolic_Diffuse_Voxel_Count", "Cytosolic_Diffuse_Volume_um3",
        "Cytosolic_Diffuse_Volume_Percent", "Cytosolic_Diffuse_Mean_Intensity",
        "Cytosolic_Diffuse_Total_Intensity",
        "Whole_Cell_Diffuse_Voxel_Count", "Whole_Cell_Diffuse_Volume_um3",
        "Whole_Cell_Diffuse_Volume_Percent", "Whole_Cell_Diffuse_Mean_Intensity",
        "Whole_Cell_Diffuse_Total_Intensity",
        "Diffuse_Nuclear_to_Cytosol_Mean_Ratio",
        "Z_Min", "Z_Max", "Y_Min", "Y_Max", "X_Min", "X_Max",
    ])

    summary = wb.create_sheet("Image Summary")
    summary.append([
        "Image", "Group", "Group1_File", "Group2_File", "Cellpose_Model", "Device",
        "Object_Count", "Puncta_Count", "Nuclear_Puncta_Count", "Cytosolic_Puncta_Count", "Nuclear_Puncta_Per_Object", "Cytosolic_Puncta_Per_Object", "Puncta_DoG_Threshold",
        "Group2_Zero_Percent", "Group2_Mean", "Group2_Max", "Group2_StdDev",
        "Group2_Signal_QC", "Diffuse_Threshold",
        "Nuclear_Diffuse_Volume_Percent_Mean",
        "Cytosolic_Diffuse_Volume_Percent_Mean",
        "Nuclear_Diffuse_Mean_Intensity",
        "Cytosolic_Diffuse_Mean_Intensity",
        "Nuclear_Diffuse_Total_Volume_um3",
        "Cytosolic_Diffuse_Total_Volume_um3",
        "Nuclear_Diffuse_Total_Intensity",
        "Cytosolic_Diffuse_Total_Intensity",
        "Diffuse_Nuclear_to_Cytosol_Ratio",
        "Nuclear_Diffuse_Total_Volume_Percent",
        "Whole_Cell_Diffuse_Total_Volume_Percent",
        "Cytosol_Expansion_Method", "XY_Pixel_Size_um", "Z_Spacing_um", "Status",
    ])

    plasmid = wb.create_sheet("Plasmid Metrics")
    plasmid.append([
        "Image", "Group", "Object_ID",
        "Background_Intensity", "Background_Method",
        "Nucleus_Volume_um3", "Cytosol_Volume_um3", "Whole_Cell_Volume_um3",
        "Nuclear_Raw_Total_Intensity", "Cytosolic_Raw_Total_Intensity",
        "Whole_Cell_Raw_Total_Intensity",
        "Nuclear_BGCorrected_Total_Intensity",
        "Cytosolic_BGCorrected_Total_Intensity",
        "Whole_Cell_BGCorrected_Total_Intensity",
        "Nuclear_BGCorrected_Mean_Intensity",
        "Cytosolic_BGCorrected_Mean_Intensity",
        "Whole_Cell_BGCorrected_Mean_Intensity",
        "Nuclear_Plasmid_Fraction", "Nuclear_Plasmid_Percent",
        "Nuclear_to_Cytosol_BGCorrected_Mean_Ratio",
        "Nuclear_Puncta_Count", "Cytosolic_Puncta_Count",
        "Nuclear_Puncta_Per_100_um3", "Cytosolic_Puncta_Per_100_um3",
        "Nuclear_Puncta_Total_Volume_um3", "Cytosolic_Puncta_Total_Volume_um3",
        "Nuclear_Puncta_Median_Volume_um3", "Cytosolic_Puncta_Median_Volume_um3",
        "Nuclear_Largest_Punctum_um3", "Cytosolic_Largest_Punctum_um3",
        "Nuclear_Puncta_Median_BGCorrected_Integrated_Intensity",
        "Cytosolic_Puncta_Median_BGCorrected_Integrated_Intensity",
        "Nuclear_Brightest_Punctum_Raw_Max_Intensity",
        "Cytosolic_Brightest_Punctum_Raw_Max_Intensity",
        "Nuclear_Puncta_BGCorrected_Total_Intensity",
        "Cytosolic_Puncta_BGCorrected_Total_Intensity",
        "Nuclear_Punctate_Intensity_Fraction",
        "Cytosolic_Punctate_Intensity_Fraction",
        "Whole_Cell_Punctate_Intensity_Fraction",
        "Nuclear_Diffuse_BGCorrected_Total_Intensity",
        "Cytosolic_Diffuse_BGCorrected_Total_Intensity",
        "Whole_Cell_Diffuse_BGCorrected_Total_Intensity",
        "Nuclear_Diffuse_Fraction_Of_Classified_Intensity",
        "Cytosolic_Diffuse_Fraction_Of_Classified_Intensity",
        "Whole_Cell_Diffuse_Fraction_Of_Classified_Intensity",
        "Nuclear_Diffuse_Volume_Percent",
        "Whole_Cell_Diffuse_Volume_Percent",
        "Median_Punctum_Distance_From_Nuclear_Surface_um",
        "Puncta_Within_Distance_Band_Count",
        "Puncta_Within_Distance_Band_Percent",
        "Perinuclear_Cytosolic_Puncta_Count",
        "Deep_Nuclear_Puncta_Count",
        "Distance_Band_um",
        "Cytosol_Expansion_Method",
    ])

    format_sheet_header(results)
    format_sheet_header(summary)
    format_sheet_header(plasmid)

    for ws in (results, summary, plasmid):
        for column_cells in ws.columns:
            letter = column_cells[0].column_letter
            ws.column_dimensions[letter].width = 18

    results.column_dimensions["A"].width = 42
    results.column_dimensions["B"].width = 16
    summary.column_dimensions["A"].width = 42
    summary.column_dimensions["B"].width = 16
    summary.column_dimensions["C"].width = 48
    summary.column_dimensions["D"].width = 48
    plasmid.column_dimensions["A"].width = 42
    plasmid.column_dimensions["B"].width = 16

    puncta_sheets_by_group = {}

    return wb, results, puncta_sheets_by_group, summary, plasmid


BATCH_ROOT_ENV = "CELLPOSE_SAM_BATCH_EXPERIMENT_ROOT"


def configure_experiment_root(args, root):
    """Apply all root-dependent paths for one experiment run."""
    args.root = Path(root).expanduser()
    args.output_folder = args.root / str(args.output_subfolder_name)
    args.output = args.output_folder / str(args.output_filename)
    return args


def run_batch_controller(args, experiment_roots):
    """Run configured experiment roots sequentially in isolated processes.

    Process isolation ensures that GPU memory and Python allocations are fully
    released between experiments. A failed experiment is recorded and the next
    configured root is still attempted.
    """
    roots = [Path(root).expanduser() for root in experiment_roots]
    if not roots:
        raise ValueError("experiment_roots must contain at least one folder.")

    normalized = [str(root).casefold() for root in roots]
    if len(normalized) != len(set(normalized)):
        raise ValueError("experiment_roots contains duplicate folders.")

    print()
    print("=" * 72)
    print("CELLPOSE-SAM MULTI-EXPERIMENT BATCH")
    print("=" * 72)
    print(f"Experiment folders: {len(roots)}")

    batch_wb = Workbook()
    batch_ws = batch_wb.active
    batch_ws.title = "Batch Summary"
    batch_ws.append([
        "Run_Order",
        "Experiment_Root",
        "Output_Workbook",
        "Status",
        "Return_Code",
        "Elapsed_Minutes",
    ])
    format_sheet_header(batch_ws)
    batch_ws.column_dimensions["A"].width = 12
    batch_ws.column_dimensions["B"].width = 70
    batch_ws.column_dimensions["C"].width = 70
    batch_ws.column_dimensions["D"].width = 18
    batch_ws.column_dimensions["E"].width = 14
    batch_ws.column_dimensions["F"].width = 18

    failed = 0
    script_path = Path(__file__).resolve()

    for run_order, root in enumerate(roots, start=1):
        output_workbook = (
            root
            / str(args.output_subfolder_name)
            / str(args.output_filename)
        )

        print()
        print("=" * 72)
        print(f"BATCH EXPERIMENT {run_order}/{len(roots)}")
        print(f"Root: {root}")
        print("=" * 72)

        environment = os.environ.copy()
        environment[BATCH_ROOT_ENV] = str(root)
        start = time.perf_counter()

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                env=environment,
                check=False,
            )
            return_code = int(completed.returncode)
            status = "OK" if return_code == 0 else "FAILED"
        except Exception as error:
            return_code = -1
            status = f"LAUNCH ERROR: {error}"

        elapsed_minutes = (time.perf_counter() - start) / 60.0
        if return_code != 0:
            failed += 1

        batch_ws.append([
            run_order,
            str(root),
            str(output_workbook),
            status,
            return_code,
            elapsed_minutes,
        ])

        # Save after every experiment so completed-run history survives an
        # interruption later in the batch.
        summary_path = roots[0].parent / str(args.batch_summary_filename)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        batch_wb.save(summary_path)

        print(
            f"Batch experiment {run_order}/{len(roots)} finished: "
            f"{status} ({elapsed_minutes:.2f} min)"
        )

    print()
    print("=" * 72)
    print("MULTI-EXPERIMENT BATCH COMPLETE")
    print("=" * 72)
    print(f"Experiments attempted: {len(roots)}")
    print(f"Experiments failed: {failed}")
    print(f"Batch summary: {summary_path}")

    return 1 if failed else 0


def main():
    args = get_settings()

    selected_batch_root = os.environ.get(BATCH_ROOT_ENV)

    if selected_batch_root:
        # Child process: analyze only the root selected by the controller.
        configure_experiment_root(args, Path(selected_batch_root))
    else:
        experiment_roots = list(args.experiment_roots)

        if not experiment_roots:
            raise ValueError("experiment_roots must contain at least one folder.")

        if len(experiment_roots) > 1:
            return run_batch_controller(args, experiment_roots)

        # A single configured experiment runs directly without spawning a child.
        configure_experiment_root(args, experiment_roots[0])

    anti_sleep_enabled = False

    if args.prevent_sleep:
        anti_sleep_enabled = enable_anti_sleep(
            keep_display_on=bool(args.keep_display_on)
        )

        if anti_sleep_enabled:
            # Restore normal sleep behavior even if the program exits because
            # of an exception or Ctrl+C.
            atexit.register(disable_anti_sleep)

    check_dependencies(args)

    root = args.root.expanduser().resolve()

    group1_dir = root / args.group1_folder
    group2_dir = root / args.group2_folder

    output_path = Path(
        args.output
    ).expanduser().resolve()

    checkpoint_path = (
        output_path.parent
        / args.checkpoint_filename
    )

    if int(args.min_size) < 1:
        raise ValueError(
            "min_size must be >= 1"
        )

    if not args.group_names:
        raise ValueError("group_names must contain at least one group.")

    group_keys = [str(group).casefold() for group in args.group_names]
    if len(group_keys) != len(set(group_keys)):
        raise ValueError(
            "group_names contains duplicate names when compared case-insensitively."
        )

    if not 2 <= int(args.excel_rows_per_puncta_sheet) <= 1_048_576:
        raise ValueError(
            "excel_rows_per_puncta_sheet must be between 2 and 1,048,576."
        )

    if int(args.save_excel_every_n_images) < 0:
        raise ValueError(
            "save_excel_every_n_images must be >= 0."
        )

    if float(args.xy_pixel_size_um) <= 0 or float(args.z_spacing_um) <= 0:
        raise ValueError("Physical voxel sizes must be > 0.")

    expansion_mode = str(args.cytosol_expansion_mode).strip().casefold()
    if expansion_mode not in {"until_touching", "fixed_distance"}:
        raise ValueError(
            'cytosol_expansion_mode must be "until_touching" or "fixed_distance".'
        )
    if expansion_mode == "fixed_distance" and float(args.cytosol_expansion_distance_um) <= 0:
        raise ValueError("cytosol_expansion_distance_um must be > 0.")

    if float(args.puncta_sigma_small_um) <= 0:
        raise ValueError("puncta_sigma_small_um must be > 0.")

    if float(args.diffuse_background_sigma_um) <= 0:
        raise ValueError("diffuse_background_sigma_um must be > 0.")

    if float(args.diffuse_threshold_std) < 0:
        raise ValueError("diffuse_threshold_std must be >= 0.")

    if float(args.diffuse_min_residual_threshold) < 0:
        raise ValueError("diffuse_min_residual_threshold must be >= 0.")

    if float(args.diffuse_min_volume_um3) < 0:
        raise ValueError("diffuse_min_volume_um3 must be >= 0.")

    if args.plasmid_fixed_background_intensity is not None:
        if float(args.plasmid_fixed_background_intensity) < 0:
            raise ValueError("plasmid_fixed_background_intensity must be >= 0.")

    if not 0.0 <= float(args.plasmid_background_percentile) <= 100.0:
        raise ValueError("plasmid_background_percentile must be between 0 and 100.")

    if float(args.puncta_density_reference_um3) <= 0:
        raise ValueError("puncta_density_reference_um3 must be > 0.")

    if float(args.perinuclear_distance_um) < 0:
        raise ValueError("perinuclear_distance_um must be >= 0.")

    if float(args.deep_nuclear_distance_um) < 0:
        raise ValueError("deep_nuclear_distance_um must be >= 0.")

    if float(args.puncta_sigma_large_um) <= float(args.puncta_sigma_small_um):
        raise ValueError(
            "puncta_sigma_large_um must be greater than puncta_sigma_small_um."
        )

    if not 0.0 <= float(args.overlay_alpha) <= 1.0:
        raise ValueError(
            "overlay_alpha must be between 0 and 1"
        )

    pairs = find_pairs(
        group1_dir,
        group2_dir,
        args.group1_suffix,
        args.group2_suffix,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir = (
        output_path.parent
        / "labels"
    )

    if (
        args.save_labels
        or args.save_colored_labels
        or args.save_overlay
    ):
        labels_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print()
    print("=" * 72)
    print("CELLPOSE-SAM 3D ANALYSIS")
    print("=" * 72)
    print(f"Matched image pairs: {len(pairs)}")
    print(f"Model: {args.pretrained_model}")
    print(f"Output: {output_path}")

    # Load model ONCE and reuse it for every image.
    model = create_cellpose_model(args)

    (
        workbook,
        results_ws,
        puncta_sheets_by_group,
        summary_ws,
        plasmid_ws,
    ) = create_workbook()

    total_objects = 0
    failed_pairs = 0

    overall_progress = tqdm(
        pairs,
        desc="Overall image progress",
        unit="image",
        dynamic_ncols=True,
    )

    for pair_index, (
        pair_name,
        group1_path,
        group2_path,
    ) in enumerate(
        overall_progress,
        start=1,
    ):

        overall_progress.set_postfix_str(
            pair_name[:35]
        )

        tqdm.write("")
        tqdm.write("=" * 72)
        tqdm.write(
            f"[{pair_index}/{len(pairs)}] {pair_name}"
        )

        group_name = find_group_name(pair_name, args)
        tqdm.write(f"  Experiment group: {group_name}")

        try:
            group1 = open_stack(
                group1_path
            )

            group2 = open_stack(
                group2_path
            )

            if group1.shape != group2.shape:
                raise ValueError(
                    "Group 1 and Group 2 shapes do not match: "
                    f"{group1.shape} vs {group2.shape}"
                )

            print(
                f"  Shape Z,Y,X: {group1.shape}"
            )

            tqdm.write(
                f"  [{pair_index}/{len(pairs)}] "
                f"Starting Cellpose: {pair_name}"
            )

            labels = run_cellpose_3d(
                model,
                group1,
                args,
            )

            tqdm.write(
                f"  [{pair_index}/{len(pairs)}] "
                f"Cellpose finished: {pair_name}"
            )

            if args.fill_holes_after_cellpose:
                labels = fill_holes_per_object(
                    labels
                )

            labels = relabel_consecutively(
                labels
            )

            object_count = int(
                labels.max()
            )

            print(
                f"    Final objects: {object_count}"
            )

            if object_count == 0:
                summary_ws.append([
                    pair_name, group_name, group1_path.name, group2_path.name,
                    args.pretrained_model, str(model.device), 0, 0, 0, 0,
                    None, None, None,
                    None, None, None, None, "NOT TESTED",
                    None, None, None, None, None, None, None, None, None, None, None, None,
                    cytosol_expansion_description(args),
                    float(args.xy_pixel_size_um),
                    float(args.z_spacing_um),
                    "No objects detected",
                ])

                del group1
                del group2
                del labels

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                continue

            stage_time = time.perf_counter()
            print("    Expanding nuclei into simulated cytosol...")
            whole_labels, cytosol_labels = expand_nuclei(labels, args)
            tqdm.write(f"    Cytosol expansion: {time.perf_counter() - stage_time:.1f} s")

            stats_start = time.perf_counter()
            print("    Measuring nuclear/cytosolic Group 2 signal...")
            nuc_g1 = region_stats(labels, group1, args.calculate_median)
            nuc_g2 = region_stats(labels, group2, args.calculate_median)
            cyto_g2 = region_stats(cytosol_labels, group2, args.calculate_median)
            whole_g2 = region_stats(whole_labels, group2, False)
            tqdm.write(
                f"    Compartment statistics: {time.perf_counter() - stats_start:.1f} s"
            )

            group2_qc = group2_signal_qc(
                group2,
                args,
            )

            if args.detect_group2_puncta:
                puncta_labels, puncta_records, puncta_threshold, rejected_puncta_labels = detect_puncta(
                    group2, whole_labels, labels, args
                )
            else:
                puncta_labels = np.zeros(labels.shape, dtype=np.int32)
                puncta_records = []
                puncta_threshold = np.nan
                rejected_puncta_labels = np.zeros(labels.shape, dtype=np.uint8)

            ps = puncta_summary(puncta_records, object_count)

            puncta_by_object = {
                object_id: []
                for object_id in range(1, object_count + 1)
            }
            for record in puncta_records:
                object_id = int(record["Cell_Object_ID"])
                if object_id in puncta_by_object:
                    puncta_by_object[object_id].append(record)

            plasmid_background, plasmid_background_method = (
                estimate_plasmid_background(group2, args)
            )
            tqdm.write(
                f"    Plasmid background estimate: {plasmid_background:.6g} "
                f"({plasmid_background_method})"
            )

            if args.detect_group2_diffuse:
                diffuse_mask, diffuse_threshold = detect_diffuse_signal(
                    group2,
                    whole_labels,
                    puncta_labels,
                    rejected_puncta_labels,
                    args,
                )
                nuc_diffuse = diffuse_stats(diffuse_mask, labels, group2, object_count)
                cyto_diffuse = diffuse_stats(diffuse_mask, cytosol_labels, group2, object_count)
            else:
                diffuse_mask = np.zeros(labels.shape, dtype=np.uint8)
                diffuse_threshold = np.nan
                nuc_diffuse = diffuse_stats(diffuse_mask, labels, group2, object_count)
                cyto_diffuse = diffuse_stats(diffuse_mask, cytosol_labels, group2, object_count)

            nuclear_puncta_count = sum(
                record["Compartment"] == "Nuclear"
                for record in puncta_records
            )
            cytosolic_puncta_count = sum(
                record["Compartment"] == "Cytosolic"
                for record in puncta_records
            )

            nuclear_puncta_per_object = (
                nuclear_puncta_count / object_count
                if object_count > 0
                else np.nan
            )

            cytosolic_puncta_per_object = (
                cytosolic_puncta_count / object_count
                if object_count > 0
                else np.nan
            )

            vv = voxel_volume_um3(args)
            boxes = ndi.find_objects(labels)

            for object_id in range(1, object_count + 1):
                box = boxes[object_id - 1]
                if box is None:
                    continue

                z_slice, y_slice, x_slice = box
                nc = int(nuc_g2["count"][object_id])
                cc = int(cyto_g2["count"][object_id])
                wc = int(whole_g2["count"][object_id])
                nm = float(nuc_g2["mean"][object_id])
                cm = float(cyto_g2["mean"][object_id])

                ncr = nm / cm if np.isfinite(nm) and np.isfinite(cm) and cm != 0 else np.nan
                cnr = cm / nm if np.isfinite(nm) and np.isfinite(cm) and nm != 0 else np.nan

                npun = ps[object_id]["Nuclear"]
                cpun = ps[object_id]["Cytosolic"]
                nv = nc * vv
                cv = cc * vv
                npct = 100 * npun["volume"] / nv if nv > 0 else np.nan
                cpct = 100 * cpun["volume"] / cv if cv > 0 else np.nan

                # Combine the mutually exclusive nuclear and cytosolic diffuse
                # measurements to obtain a per-Object_ID whole-cell result.
                whole_diffuse_count = int(
                    nuc_diffuse["count"][object_id]
                    + cyto_diffuse["count"][object_id]
                )
                whole_diffuse_total = float(
                    nuc_diffuse["sum"][object_id]
                    + cyto_diffuse["sum"][object_id]
                )
                whole_diffuse_mean = (
                    whole_diffuse_total / whole_diffuse_count
                    if whole_diffuse_count > 0 else np.nan
                )
                whole_diffuse_percent = (
                    100.0 * whole_diffuse_count / wc
                    if wc > 0 else np.nan
                )

                # ---------------------------------------------------
                # PER-OBJECT PLASMID COMPARISON METRICS
                # ---------------------------------------------------
                nuclear_raw_total = float(nuc_g2["sum"][object_id])
                cytosolic_raw_total = float(cyto_g2["sum"][object_id])
                whole_raw_total = nuclear_raw_total + cytosolic_raw_total

                nuclear_corrected_total = max(
                    0.0,
                    nuclear_raw_total - plasmid_background * nc,
                )
                cytosolic_corrected_total = max(
                    0.0,
                    cytosolic_raw_total - plasmid_background * cc,
                )
                whole_corrected_total = (
                    nuclear_corrected_total + cytosolic_corrected_total
                )

                nuclear_corrected_mean = (
                    nuclear_corrected_total / nc if nc > 0 else np.nan
                )
                cytosolic_corrected_mean = (
                    cytosolic_corrected_total / cc if cc > 0 else np.nan
                )
                whole_corrected_mean = (
                    whole_corrected_total / wc if wc > 0 else np.nan
                )

                nuclear_plasmid_fraction = (
                    nuclear_corrected_total / whole_corrected_total
                    if whole_corrected_total > 0 else np.nan
                )
                corrected_nc_ratio = (
                    nuclear_corrected_mean / cytosolic_corrected_mean
                    if np.isfinite(nuclear_corrected_mean)
                    and np.isfinite(cytosolic_corrected_mean)
                    and cytosolic_corrected_mean > 0 else np.nan
                )

                object_puncta = puncta_by_object[object_id]
                nuclear_records = [
                    record for record in object_puncta
                    if record["Compartment"] == "Nuclear"
                ]
                cytosolic_records = [
                    record for record in object_puncta
                    if record["Compartment"] == "Cytosolic"
                ]

                def puncta_metrics(records):
                    volumes = np.asarray(
                        [record["Volume_um3"] for record in records],
                        dtype=np.float64,
                    )
                    corrected_integrated = np.asarray([
                        max(
                            0.0,
                            float(record["Integrated_Intensity"])
                            - plasmid_background * int(record["Voxel_Count"]),
                        )
                        for record in records
                    ], dtype=np.float64)
                    maxima = np.asarray(
                        [record["Max_Intensity"] for record in records],
                        dtype=np.float64,
                    )
                    return {
                        "median_volume": (
                            float(np.median(volumes)) if volumes.size else np.nan
                        ),
                        "largest_volume": (
                            float(np.max(volumes)) if volumes.size else 0.0
                        ),
                        "median_corrected_integrated": (
                            float(np.median(corrected_integrated))
                            if corrected_integrated.size else np.nan
                        ),
                        "corrected_total": float(np.sum(corrected_integrated)),
                        "brightest_raw_max": (
                            float(np.max(maxima)) if maxima.size else np.nan
                        ),
                    }

                nuclear_puncta_metrics = puncta_metrics(nuclear_records)
                cytosolic_puncta_metrics = puncta_metrics(cytosolic_records)
                nuclear_puncta_corrected = nuclear_puncta_metrics["corrected_total"]
                cytosolic_puncta_corrected = cytosolic_puncta_metrics["corrected_total"]
                whole_puncta_corrected = (
                    nuclear_puncta_corrected + cytosolic_puncta_corrected
                )

                nuclear_diffuse_corrected = max(
                    0.0,
                    float(nuc_diffuse["sum"][object_id])
                    - plasmid_background * int(nuc_diffuse["count"][object_id]),
                )
                cytosolic_diffuse_corrected = max(
                    0.0,
                    float(cyto_diffuse["sum"][object_id])
                    - plasmid_background * int(cyto_diffuse["count"][object_id]),
                )
                whole_diffuse_corrected = (
                    nuclear_diffuse_corrected + cytosolic_diffuse_corrected
                )

                nuclear_punctate_fraction = (
                    nuclear_puncta_corrected / nuclear_corrected_total
                    if nuclear_corrected_total > 0 else np.nan
                )
                cytosolic_punctate_fraction = (
                    cytosolic_puncta_corrected / cytosolic_corrected_total
                    if cytosolic_corrected_total > 0 else np.nan
                )
                whole_punctate_fraction = (
                    whole_puncta_corrected / whole_corrected_total
                    if whole_corrected_total > 0 else np.nan
                )

                def diffuse_classified_fraction(diffuse_total, puncta_total):
                    classified_total = diffuse_total + puncta_total
                    return (
                        diffuse_total / classified_total
                        if classified_total > 0 else np.nan
                    )

                nuclear_diffuse_classified_fraction = diffuse_classified_fraction(
                    nuclear_diffuse_corrected, nuclear_puncta_corrected
                )
                cytosolic_diffuse_classified_fraction = diffuse_classified_fraction(
                    cytosolic_diffuse_corrected, cytosolic_puncta_corrected
                )
                whole_diffuse_classified_fraction = diffuse_classified_fraction(
                    whole_diffuse_corrected, whole_puncta_corrected
                )

                distances = np.asarray([
                    record["Distance_From_Nuclear_Surface_um"]
                    for record in object_puncta
                ], dtype=np.float64)
                distance_band = float(args.perinuclear_distance_um)
                finite_distances = distances[np.isfinite(distances)]
                within_band_count = int(
                    np.count_nonzero(np.abs(finite_distances) <= distance_band)
                )
                within_band_percent = (
                    100.0 * within_band_count / finite_distances.size
                    if finite_distances.size else np.nan
                )
                perinuclear_cytosolic_count = int(np.count_nonzero(
                    (finite_distances > 0)
                    & (finite_distances <= distance_band)
                ))
                deep_nuclear_count = int(np.count_nonzero(
                    finite_distances <= -float(args.deep_nuclear_distance_um)
                ))
                median_punctum_distance = (
                    float(np.median(finite_distances))
                    if finite_distances.size else np.nan
                )

                results_ws.append([
                    pair_name, group_name, object_id,
                    nc, nv, cc, cv, wc, wc * vv,
                    float(nuc_g1["mean"][object_id]),
                    nm, float(nuc_g2["median"][object_id]),
                    float(nuc_g2["std"][object_id]),
                    float(nuc_g2["min"][object_id]),
                    float(nuc_g2["max"][object_id]),
                    float(nuc_g2["sum"][object_id]),
                    cm, float(cyto_g2["median"][object_id]),
                    float(cyto_g2["std"][object_id]),
                    float(cyto_g2["min"][object_id]),
                    float(cyto_g2["max"][object_id]),
                    float(cyto_g2["sum"][object_id]),
                    float(whole_g2["mean"][object_id]),
                    float(ncr), float(cnr),
                    npun["count"], npun["volume"], npct, npun["total"],
                    float(np.mean(npun["means"])) if npun["means"] else np.nan,
                    npun["max"], npun["largest"],
                    cpun["count"], cpun["volume"], cpct, cpun["total"],
                    float(np.mean(cpun["means"])) if cpun["means"] else np.nan,
                    cpun["max"], cpun["largest"],
                    int(nuc_diffuse["count"][object_id]),
                    float(nuc_diffuse["count"][object_id] * vv),
                    100.0 * float(nuc_diffuse["count"][object_id]) / nc if nc > 0 else np.nan,
                    float(nuc_diffuse["mean"][object_id]),
                    float(nuc_diffuse["sum"][object_id]),
                    int(cyto_diffuse["count"][object_id]),
                    float(cyto_diffuse["count"][object_id] * vv),
                    100.0 * float(cyto_diffuse["count"][object_id]) / cc if cc > 0 else np.nan,
                    float(cyto_diffuse["mean"][object_id]),
                    float(cyto_diffuse["sum"][object_id]),
                    whole_diffuse_count,
                    float(whole_diffuse_count * vv),
                    float(whole_diffuse_percent),
                    float(whole_diffuse_mean),
                    whole_diffuse_total,
                    (float(nuc_diffuse["mean"][object_id]) / float(cyto_diffuse["mean"][object_id])
                     if np.isfinite(nuc_diffuse["mean"][object_id])
                     and np.isfinite(cyto_diffuse["mean"][object_id])
                     and float(cyto_diffuse["mean"][object_id]) != 0 else np.nan),
                    z_slice.start, z_slice.stop - 1,
                    y_slice.start, y_slice.stop - 1,
                    x_slice.start, x_slice.stop - 1,
                ])

                density_reference = float(args.puncta_density_reference_um3)
                plasmid_ws.append([
                    pair_name,
                    group_name,
                    object_id,
                    float(plasmid_background),
                    plasmid_background_method,
                    float(nv),
                    float(cv),
                    float(wc * vv),
                    nuclear_raw_total,
                    cytosolic_raw_total,
                    whole_raw_total,
                    nuclear_corrected_total,
                    cytosolic_corrected_total,
                    whole_corrected_total,
                    float(nuclear_corrected_mean),
                    float(cytosolic_corrected_mean),
                    float(whole_corrected_mean),
                    float(nuclear_plasmid_fraction),
                    (100.0 * float(nuclear_plasmid_fraction)
                     if np.isfinite(nuclear_plasmid_fraction) else np.nan),
                    float(corrected_nc_ratio),
                    len(nuclear_records),
                    len(cytosolic_records),
                    (density_reference * len(nuclear_records) / nv
                     if nv > 0 else np.nan),
                    (density_reference * len(cytosolic_records) / cv
                     if cv > 0 else np.nan),
                    float(npun["volume"]),
                    float(cpun["volume"]),
                    nuclear_puncta_metrics["median_volume"],
                    cytosolic_puncta_metrics["median_volume"],
                    nuclear_puncta_metrics["largest_volume"],
                    cytosolic_puncta_metrics["largest_volume"],
                    nuclear_puncta_metrics["median_corrected_integrated"],
                    cytosolic_puncta_metrics["median_corrected_integrated"],
                    nuclear_puncta_metrics["brightest_raw_max"],
                    cytosolic_puncta_metrics["brightest_raw_max"],
                    nuclear_puncta_corrected,
                    cytosolic_puncta_corrected,
                    float(nuclear_punctate_fraction),
                    float(cytosolic_punctate_fraction),
                    float(whole_punctate_fraction),
                    nuclear_diffuse_corrected,
                    cytosolic_diffuse_corrected,
                    whole_diffuse_corrected,
                    float(nuclear_diffuse_classified_fraction),
                    float(cytosolic_diffuse_classified_fraction),
                    float(whole_diffuse_classified_fraction),
                    (100.0 * float(nuc_diffuse["count"][object_id]) / nc
                     if nc > 0 else np.nan),
                    float(whole_diffuse_percent),
                    float(median_punctum_distance),
                    within_band_count,
                    float(within_band_percent),
                    perinuclear_cytosolic_count,
                    deep_nuclear_count,
                    distance_band,
                    cytosol_expansion_description(args),
                ])

            for r in puncta_records:
                append_puncta_row(
                    workbook,
                    puncta_sheets_by_group,
                    group_name,
                    [
                        pair_name,
                        group_name,
                        r["Cell_Object_ID"],
                        r["Punctum_ID"],
                        r["Compartment"],
                        r["Voxel_Count"],
                        r["Volume_um3"],
                        r["Mean_Intensity"],
                        r["Max_Intensity"],
                        r["Integrated_Intensity"],
                        r["Centroid_Z_vox"],
                        r["Centroid_Y_vox"],
                        r["Centroid_X_vox"],
                        r["Centroid_Z_um"],
                        r["Centroid_Y_um"],
                        r["Centroid_X_um"],
                        r.get(
                            "Distance_From_Nuclear_Surface_um",
                            np.nan,
                        ),
                        r.get("Shape_Aspect_Ratio_3D", np.nan),
                        r.get("Shape_Sphericity_PCA_3D", np.nan),
                    ],
                    args.excel_rows_per_puncta_sheet,
                )

            # -------------------------------------------------------
            # SAVE RAW LABELS
            # -------------------------------------------------------

            if args.save_labels:
                label_path = (
                    labels_dir
                    / f"{pair_name}_labels.tif"
                )

                print(
                    f"    Saving: {label_path.name}"
                )

                tifffile.imwrite(
                    label_path,
                    labels.astype(
                        np.int32,
                        copy=False,
                    ),
                    photometric="minisblack",
                    metadata={
                        "axes": "ZYX"
                    },
                    compression=("zlib" if args.compress_output_tiffs else None),
                )

            # -------------------------------------------------------
            # SAVE COMPARTMENT / PUNCTA QC
            # -------------------------------------------------------
            if args.save_compartment_labels:
                comp_rgb = make_compartment_rgb(labels, cytosol_labels)
                tifffile.imwrite(
                    labels_dir / f"{pair_name}_NUCLEUS_CYTOSOL.tif",
                    comp_rgb, photometric="rgb",
                    metadata={"axes": "ZYXC"}, compression=("zlib" if args.compress_output_tiffs else None),
                )
                del comp_rgb

            if args.save_puncta_labels:
                tifffile.imwrite(
                    labels_dir / f"{pair_name}_GROUP2_PUNCTA.tif",
                    puncta_labels.astype(np.int32, copy=False),
                    photometric="minisblack",
                    metadata={"axes": "ZYX"}, compression=("zlib" if args.compress_output_tiffs else None),
                )

            if args.save_puncta_overlay:
                po = make_puncta_multichannel_qc(
                    group2, labels, cytosol_labels, puncta_labels,
                    rejected_puncta_labels, diffuse_mask
                )
                channel_names = [
                    "Group2_Raw_Normalized",
                    "Nucleus_Mask",
                    "Cytosol_Mask",
                    "Diffuse_Signal",
                    "Accepted_Puncta",
                    "Rejected_Size_Volume",
                    "Rejected_3D_Shape",
                    "Rejected_Export_Max_Volume",
                ]
                # Save as a Fiji/ImageJ hyperstack in Composite mode.
                # TZCYX with singleton T is the layout tifffile expects for
                # an ImageJ hyperstack with separate channel and Z dimensions.
                # ImageJ metadata controls the display mode; OME metadata alone
                # does not force Fiji to open an image as a Composite.
                qc_path = (
                    labels_dir
                    / f"{pair_name}_GROUP2_PUNCTA_COMPOSITE.tif"
                )
                tifffile.imwrite(
                    qc_path,
                    po,
                    imagej=True,
                    metadata={
                        "axes": "TZCYX",
                        "mode": "composite",
                        "unit": "um",
                        "spacing": float(args.z_spacing_um),
                        "finterval": 0.0,
                        "Labels": channel_names,
                    },
                    resolution=(
                        1.0 / float(args.xy_pixel_size_um),
                        1.0 / float(args.xy_pixel_size_um),
                    ),
                    compression=("zlib" if args.compress_output_tiffs else None),
                )
                del po

            # -------------------------------------------------------
            # SAVE COLOR / OVERLAY
            # -------------------------------------------------------

            if (
                args.save_colored_labels
                or args.save_overlay
            ):
                colored_labels = make_label_colors(
                    labels
                )

                if args.save_colored_labels:
                    color_path = (
                        labels_dir
                        / f"{pair_name}_labels_COLOR.tif"
                    )

                    print(
                        f"    Saving: {color_path.name}"
                    )

                    tifffile.imwrite(
                        color_path,
                        colored_labels,
                        photometric="rgb",
                        metadata={
                            "axes": "ZYXC"
                        },
                        compression=("zlib" if args.compress_output_tiffs else None),
                    )

                if args.save_overlay:
                    overlay = make_overlay(
                        group1,
                        colored_labels,
                        labels,
                        float(args.overlay_alpha),
                    )

                    overlay_path = (
                        labels_dir
                        / f"{pair_name}_labels_OVERLAY.tif"
                    )

                    print(
                        f"    Saving: {overlay_path.name}"
                    )

                    tifffile.imwrite(
                        overlay_path,
                        overlay,
                        photometric="rgb",
                        metadata={
                            "axes": "ZYXC"
                        },
                        compression=("zlib" if args.compress_output_tiffs else None),
                    )

                    del overlay

                del colored_labels

            # Image-level diffuse summaries. Volume-percent values are the
            # mean of per-cell percentages; intensity values are pooled over
            # all diffuse-positive voxels in the corresponding compartment.
            object_ids = np.arange(1, object_count + 1)
            nuc_counts_img = nuc_diffuse["count"][1:object_count + 1].astype(np.float64)
            cyto_counts_img = cyto_diffuse["count"][1:object_count + 1].astype(np.float64)
            nuc_region_counts = nuc_g2["count"][1:object_count + 1].astype(np.float64)
            cyto_region_counts = cyto_g2["count"][1:object_count + 1].astype(np.float64)

            nuc_diffuse_pct_cells = np.divide(
                100.0 * nuc_counts_img, nuc_region_counts,
                out=np.full(object_count, np.nan), where=nuc_region_counts > 0,
            )
            cyto_diffuse_pct_cells = np.divide(
                100.0 * cyto_counts_img, cyto_region_counts,
                out=np.full(object_count, np.nan), where=cyto_region_counts > 0,
            )

            nuc_diffuse_total_count = float(np.sum(nuc_counts_img))
            cyto_diffuse_total_count = float(np.sum(cyto_counts_img))
            nuc_diffuse_total_intensity = float(np.sum(nuc_diffuse["sum"][1:object_count + 1]))
            cyto_diffuse_total_intensity = float(np.sum(cyto_diffuse["sum"][1:object_count + 1]))
            nuc_diffuse_image_mean = (
                nuc_diffuse_total_intensity / nuc_diffuse_total_count
                if nuc_diffuse_total_count > 0 else np.nan
            )
            cyto_diffuse_image_mean = (
                cyto_diffuse_total_intensity / cyto_diffuse_total_count
                if cyto_diffuse_total_count > 0 else np.nan
            )
            diffuse_nc_ratio = (
                nuc_diffuse_image_mean / cyto_diffuse_image_mean
                if np.isfinite(nuc_diffuse_image_mean)
                and np.isfinite(cyto_diffuse_image_mean)
                and cyto_diffuse_image_mean != 0 else np.nan
            )
            # Image-level pooled volume fractions. Unlike the previous
            # Diffuse_Positive_Cell_Percent measurement, these do not classify
            # a cell as positive based on a single voxel. They calculate how
            # much of the total nuclear or whole-cell volume is occupied by
            # diffuse-positive voxels across the image.
            total_nuclear_voxels = float(np.sum(nuc_region_counts))
            total_whole_cell_voxels = float(
                np.sum(nuc_region_counts + cyto_region_counts)
            )
            total_diffuse_voxels = float(
                nuc_diffuse_total_count + cyto_diffuse_total_count
            )

            nuclear_diffuse_total_volume_percent = (
                100.0 * nuc_diffuse_total_count / total_nuclear_voxels
                if total_nuclear_voxels > 0 else np.nan
            )
            whole_cell_diffuse_total_volume_percent = (
                100.0 * total_diffuse_voxels / total_whole_cell_voxels
                if total_whole_cell_voxels > 0 else np.nan
            )

            summary_ws.append([
                pair_name, group_name, group1_path.name, group2_path.name,
                args.pretrained_model, str(model.device), object_count,
                len(puncta_records),
                nuclear_puncta_count,
                cytosolic_puncta_count,
                float(nuclear_puncta_per_object),
                float(cytosolic_puncta_per_object),
                float(puncta_threshold) if np.isfinite(puncta_threshold) else None,
                100.0 * float(group2_qc["zero_fraction"]),
                float(group2_qc["mean"]),
                float(group2_qc["max"]),
                float(group2_qc["std"]),
                "LOW SIGNAL" if group2_qc["low_signal"] else "OK",
                float(diffuse_threshold) if np.isfinite(diffuse_threshold) else None,
                float(np.nanmean(nuc_diffuse_pct_cells)) if np.any(np.isfinite(nuc_diffuse_pct_cells)) else None,
                float(np.nanmean(cyto_diffuse_pct_cells)) if np.any(np.isfinite(cyto_diffuse_pct_cells)) else None,
                float(nuc_diffuse_image_mean) if np.isfinite(nuc_diffuse_image_mean) else None,
                float(cyto_diffuse_image_mean) if np.isfinite(cyto_diffuse_image_mean) else None,
                float(nuc_diffuse_total_count * vv),
                float(cyto_diffuse_total_count * vv),
                float(nuc_diffuse_total_intensity),
                float(cyto_diffuse_total_intensity),
                float(diffuse_nc_ratio) if np.isfinite(diffuse_nc_ratio) else None,
                (float(nuclear_diffuse_total_volume_percent)
                 if np.isfinite(nuclear_diffuse_total_volume_percent) else None),
                (float(whole_cell_diffuse_total_volume_percent)
                 if np.isfinite(whole_cell_diffuse_total_volume_percent) else None),
                cytosol_expansion_description(args),
                float(args.xy_pixel_size_um),
                float(args.z_spacing_um),
                "OK",
            ])

            total_objects += object_count

            if (
                int(args.save_excel_every_n_images) > 0
                and pair_index % int(args.save_excel_every_n_images) == 0
            ):
                tqdm.write(
                    f"  Saving Excel checkpoint after image "
                    f"{pair_index}/{len(pairs)}..."
                )

                checkpoint_start = time.perf_counter()

                workbook.save(
                    checkpoint_path
                )

                tqdm.write(
                    f"  Checkpoint saved in "
                    f"{time.perf_counter() - checkpoint_start:.1f} s: "
                    f"{checkpoint_path.name}"
                )

            print(
                f"  Completed: "
                f"{object_count} object(s)"
            )

            del group1, group2, labels, whole_labels, cytosol_labels
            del puncta_labels, puncta_records, rejected_puncta_labels, ps, group2_qc
            del puncta_by_object
            del diffuse_mask, nuc_diffuse, cyto_diffuse
            del nuc_g1, nuc_g2, cyto_g2, whole_g2, boxes

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as error:
            failed_pairs += 1

            print(
                f"  ERROR: {error}",
                file=sys.stderr,
            )

            traceback.print_exc()

            summary_ws.append([
                pair_name, group_name, group1_path.name, group2_path.name,
                args.pretrained_model, str(model.device),
                None, None, None, None, None, None, None,
                None, None, None, None, "ERROR",
                None, None, None, None, None, None, None, None, None, None, None, None,
                cytosol_expansion_description(args),
                float(args.xy_pixel_size_um),
                float(args.z_spacing_um),
                f"ERROR: {error}",
            ])

            gc.collect()

            if (
                torch is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # EXCEL NUMBER FORMATTING
    # ---------------------------------------------------------------

    all_puncta_sheets = [
        sheet
        for group_sheets in puncta_sheets_by_group.values()
        for sheet in group_sheets
    ]

    all_output_sheets = [
        results_ws,
        plasmid_ws,
        *all_puncta_sheets,
        summary_ws,
    ]

    for ws in all_output_sheets:
        for row in ws.iter_rows(
            min_row=2,
            max_row=ws.max_row,
        ):
            for cell in row:
                if isinstance(
                    cell.value,
                    float,
                ):
                    cell.number_format = "0.0000"

    print()
    print("Saving Excel workbook...")

    workbook.save(
        output_path
    )

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    total_puncta_sheets = sum(
        len(group_sheets)
        for group_sheets in puncta_sheets_by_group.values()
    )

    print(
        f"Puncta data written across "
        f"{total_puncta_sheets} group-specific Excel worksheet(s)."
    )

    for group_name, group_sheets in puncta_sheets_by_group.items():
        print(
            f"  {group_name}: "
            + ", ".join(sheet.title for sheet in group_sheets)
        )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(
        f"Matched pairs: {len(pairs)}"
    )
    print(
        f"Objects measured: {total_objects}"
    )
    print(
        f"Failed pairs: {failed_pairs}"
    )
    print(
        f"Excel file: {output_path}"
    )

    if (
        args.save_labels
        or args.save_colored_labels
        or args.save_overlay
    ):
        print(
            f"Label images: {labels_dir}"
        )

    return 1 if failed_pairs else 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nCancelled by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"\nFATAL ERROR: {error}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1)
