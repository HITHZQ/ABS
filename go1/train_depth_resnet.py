# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Ray-prediction network training: depth image -> lidar (ray distances).
Data format from go1/camrec.py: each subfolder has .npy (depth HxW) + label.pkl (ray2d 11-d).

Run: python go1/train_depth_resnet.py --data_folder path/to/logs/rec_cam [--exp_name my_run]

Interface with camrec.py:
  - data_folder: same as camrec --log_root (default: ABS_ROOT/logs/rec_cam)
  - Each subfolder (test10, test11, ...): robot_*_step*.npy + label.pkl
  - depth: float32 (H, W), range [0.1, 6.0] per CAMREC_CAMERA clipping_range
  - label: float32 (11,) ray distances, matches agile lidar (11 rays)
"""

from __future__ import annotations

import copy
import os
from datetime import datetime

import numpy as np
import pickle
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

ABS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Interface constants: MUST match go1/camrec.py and agile lidar
# -----------------------------------------------------------------------------
LIDAR_NUM_RAYS = 11  # agile lidar: horizontal_fov pi/2, res pi/20
DEPTH_MIN = 0.1
DEPTH_MAX = 6.0  # CAMREC_CAMERA clipping_range
LOG_EPS = 0.01  # avoid log(0) = -inf


class RayPredictionDataset(Dataset):
    """Dataset for (depth_image, ray_label) from camrec.py output."""

    def __init__(self, all_dataset, log_eps: float = LOG_EPS):
        self.data = list(all_dataset.values())
        self.log_eps = log_eps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, lidar_data = self.data[idx]
        # Data already log2-transformed in _load_camrec_data; do not clip (would corrupt log values)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).float()
        lidar_tensor = torch.from_numpy(np.asarray(lidar_data, dtype=np.float32)).float()
        return image_tensor, lidar_tensor


class ResNetModel(torch.nn.Module):
    def __init__(self, resnet_type: str, num_rays: int = LIDAR_NUM_RAYS):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        elif resnet_type == "resnet34":
            self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError(f"resnet_type {resnet_type}")
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(self.num_ftrs, num_rays)

    def forward(self, x):
        return self.resnet(x)


def _load_camrec_data(data_folder: str, log_eps: float = LOG_EPS):
    """
    Load data from camrec.py output structure.
    data_folder contains subdirs (test10, test11, ...), each with:
      - robot_*_step*.npy: depth (H, W) float32
      - label.pkl: dict[save_name] -> ray (11,) float32
    """
    all_dataset = {}
    subdirs = [
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ]
    subdirs.sort()

    for folder_name in subdirs:
        folder_path = os.path.join(data_folder, folder_name)
        label_path = os.path.join(folder_path, "label.pkl")
        if not os.path.isfile(label_path):
            print(f"Skip {folder_name}: no label.pkl")
            continue
        with open(label_path, "rb") as f:
            _label = pickle.load(f)
        npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        for fname in npy_files:
            image_idx = fname[:-4]
            if image_idx not in _label:
                continue
            image_path = os.path.join(folder_path, fname)
            image = np.load(image_path, allow_pickle=True)
            if not isinstance(image, np.ndarray):
                continue
            ray_label = _label[image_idx]
            ray_label = np.asarray(ray_label, dtype=np.float32).flatten()
            if ray_label.size != LIDAR_NUM_RAYS:
                continue
            image_log = np.log2(np.clip(image.astype(np.float32), log_eps, None))
            label_log = np.log2(np.clip(ray_label, log_eps, None))
            key = f"{folder_name}_{image_idx}"
            all_dataset[key] = [image_log, label_log]
    return all_dataset


def train(args):
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(
        ABS_ROOT, "logs", "depth_logs",
        f"{datetime_now}-{args.resnet_type}-{args.exp_name}",
    )
    os.makedirs(log_folder, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=log_folder)

    print("============================== loading data ==============================")
    data_folder = args.data_folder or os.path.join(ABS_ROOT, "logs", "rec_cam")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(
            f"data_folder not found: {data_folder}\n"
            "Run camrec.py first, or pass --data_folder path/to/rec_cam"
        )
    all_dataset = _load_camrec_data(data_folder)
    if len(all_dataset) == 0:
        raise ValueError(f"No valid samples in {data_folder}")

    print(f"Total samples: {len(all_dataset)}")
    dataset = RayPredictionDataset(all_dataset)

    test_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResNetModel(args.resnet_type).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.MSELoss()

    log2_min = np.log2(LOG_EPS)
    log2_max = np.log2(DEPTH_MAX)

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if (not args.no_leftright) and np.random.uniform() < 0.5:
                inputs = torch.flip(inputs, dims=[-1])
                targets = torch.flip(targets, dims=[-1])

            if args.gaussian_blur_augmentation and np.random.uniform() < args.gaussian_blur_augmentation_prob:
                blur = torchvision.transforms.GaussianBlur(
                    kernel_size=args.gaussian_blur_augmentation_kernel_size,
                    sigma=(0.1, 2.0),
                )
                inputs = blur(inputs)

            if args.noise_augmentation and np.random.uniform() < args.noise_augmentation_prob:
                noise_std = np.random.uniform() * args.noise_augmentation_std_max
                inputs = inputs + torch.randn_like(inputs) * noise_std + args.noise_augmentation_mean

            if args.random_erasing_augmentation and np.random.uniform() < args.random_erasing_augmentation_prob:
                H, W = inputs.size(1), inputs.size(2)
                h = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max + 1)
                w = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max + 1)
                if H > h and W > w:
                    x = np.random.randint(0, W - w)
                    y = np.random.randint(0, H - h)
                    v = np.random.choice([log2_min, log2_max])
                    inputs = torchvision.transforms.functional.erase(inputs, x, y, h, w, v=v)

            inputs = inputs.to(device).unsqueeze(1).repeat(1, 3, 1, 1)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch}/{args.num_epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
                tensorboard_writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device).unsqueeze(1).repeat(1, 3, 1, 1)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.6f}")
        tensorboard_writer.add_scalar("Test/Loss", test_loss, epoch)

        if epoch % args.save_interval == 0:
            model_path = os.path.join(log_folder, f"depth_lidar_model_{datetime_now}_{epoch}.pt")
            model_cpu = copy.deepcopy(model).cpu()
            try:
                torch.jit.script(model_cpu).save(model_path)
            except Exception:
                torch.save(model_cpu.state_dict(), model_path.replace(".pt", "_state.pt"))
            print(f"Model saved: {model_path}")

    tensorboard_writer.close()


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ray-prediction: train depth->lidar ResNet. Data from go1/camrec.py."
    )
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Path to rec_cam (camrec --log_root). Default: ABS_ROOT/logs/rec_cam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=302)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--resnet_type", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--no_leftright", action="store_true", help="Disable left-right augmentation")
    parser.add_argument("--noise_augmentation", action="store_true", default=True)
    parser.add_argument("--noise_augmentation_std_max", type=float, default=0.3)
    parser.add_argument("--noise_augmentation_mean", type=float, default=0.0)
    parser.add_argument("--noise_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--gaussian_blur_augmentation", action="store_true", default=True)
    parser.add_argument("--gaussian_blur_augmentation_kernel_size", type=int, default=5)
    parser.add_argument("--gaussian_blur_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--random_erasing_augmentation", action="store_true", default=True)
    parser.add_argument("--random_erasing_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--random_erasing_area_min", type=int, default=5)
    parser.add_argument("--random_erasing_area_max", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
