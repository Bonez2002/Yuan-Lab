# ============================================================
# 3D U-Net Sliding Window Inference Script
# Low RAM / GPU memory version
# ============================================================

import os
import sys
import subprocess
import atexit
import gc
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from tqdm import tqdm


# ============================================================
# CONFIG — EDIT THESE VARIABLES
# ============================================================

CONFIG = {

    "input_images_dir": r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_03_13/Raw/Split/group_1",

    "output_masks_dir": r"S:/Lab Data/Stellaris 8/Plasmid Small Molecule/2026_03_13/Raw/Split/group_1/Segment",

    "model_path": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Models/LowSNR_10x_Good/best_model.pth",

    "base_filters": 16,
    "in_channels": "auto",
    "out_channels": 1,

    "threshold": 0.25,

    # Sliding window parameters
    "patch_size": 96,
    "stride": 64,

    # Half precision reduces GPU memory ~50%
    "use_fp16": True,

    "device": None
}


# ============================================================
# DEVICE
# ============================================================

if CONFIG["device"] is None:
    CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(CONFIG["device"])

print("Using device:", device)


# ============================================================
# ANTI-SLEEP
# ============================================================

class AntiSleep:

    def __init__(self):
        self.proc = None
        self.platform = sys.platform
        atexit.register(self.stop)

    def start(self):

        try:

            if self.platform.startswith("win"):

                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    0x80000000 | 0x00000001
                )
                print("Anti-sleep enabled (Windows)")

            elif self.platform == "darwin":

                self.proc = subprocess.Popen(["caffeinate"])
                print("Anti-sleep enabled (macOS)")

            else:

                self.proc = subprocess.Popen(
                    ["systemd-inhibit", "sleep", "999999"]
                )
                print("Anti-sleep enabled (Linux)")

        except:
            print("Warning: Anti-sleep failed")

    def stop(self):

        try:
            if self.proc:
                self.proc.terminate()
        except:
            pass


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.seq = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            (
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ),
        )

        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):

    def __init__(self, in_channels, base_filters, out_channels):

        super().__init__()

        f = base_filters

        self.inc = DoubleConv(in_channels, f)

        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 8)

        self.up1 = Up(f * 16, f * 4)
        self.up2 = Up(f * 8, f * 2)
        self.up3 = Up(f * 4, f)
        self.up4 = Up(f * 2, f)

        self.outc = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return torch.sigmoid(self.outc(x))


# ============================================================
# TIFF LOADING
# ============================================================

def load_tiff(path):

    vol = tifffile.imread(path)

    vol = vol.astype(np.float32)

    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]

    elif vol.ndim == 4:
        vol = np.moveaxis(vol, -1, 0)

    else:
        raise ValueError(f"Unsupported TIFF shape: {vol.shape}")

    return vol


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(vol):

    out = np.zeros_like(vol)

    for c in range(vol.shape[0]):

        p1, p99 = np.percentile(vol[c], (1, 99))

        out[c] = np.clip(
            (vol[c] - p1) / (p99 - p1 + 1e-6),
            0,
            1
        )

    return out


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

def sliding_window_inference(vol, model):

    patch = CONFIG["patch_size"]
    stride = CONFIG["stride"]

    C, Z, Y, X = vol.shape

    output = np.zeros((Z, Y, X), dtype=np.float32)
    count = np.zeros((Z, Y, X), dtype=np.float32)

    for z in range(0, Z - patch + 1, stride):
        for y in range(0, Y - patch + 1, stride):
            for x in range(0, X - patch + 1, stride):

                cube = vol[:, z:z+patch, y:y+patch, x:x+patch]

                tensor = torch.from_numpy(cube)[None].to(device)

                if CONFIG["use_fp16"] and device.type == "cuda":
                    tensor = tensor.half()

                pred = model(tensor).cpu().numpy()[0, 0]

                output[z:z+patch, y:y+patch, x:x+patch] += pred
                count[z:z+patch, y:y+patch, x:x+patch] += 1

    output /= np.maximum(count, 1)

    return output


# ============================================================
# MAIN
# ============================================================

def segment_folder():

    os.makedirs(CONFIG["output_masks_dir"], exist_ok=True)

    image_paths = sorted(
        glob(os.path.join(CONFIG["input_images_dir"], "*.tif"))
    )

    print("Images found:", len(image_paths))

    if CONFIG["in_channels"] == "auto":

        sample = tifffile.imread(image_paths[0])

        if sample.ndim == 3:
            in_ch = 1
        else:
            in_ch = sample.shape[-1]

    else:
        in_ch = CONFIG["in_channels"]

    model = UNet3D(
        in_ch,
        CONFIG["base_filters"],
        CONFIG["out_channels"]
    ).to(device)

    model.load_state_dict(
        torch.load(CONFIG["model_path"], map_location=device)
    )

    if CONFIG["use_fp16"] and device.type == "cuda":
        model = model.half()

    model.eval()

    with torch.no_grad():

        for path in tqdm(image_paths):

            vol = normalize(load_tiff(path))

            pred = sliding_window_inference(vol, model)

            mask = (pred > CONFIG["threshold"]).astype(np.uint8)

            name = os.path.basename(path)

            out_path = os.path.join(CONFIG["output_masks_dir"], name)

            tifffile.imwrite(out_path, mask)

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("Segmentation complete!")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    anti = AntiSleep()
    anti.start()

    try:
        segment_folder()

    finally:
        anti.stop()