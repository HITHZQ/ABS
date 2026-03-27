"""
Go2 ray-prediction training: depth image -> lidar ray distances.

Expected data format from go2/camrec.py:
  <data_folder>/testXX/robot_<env_id>_step<step>.npy
  <data_folder>/testXX/label_raw.pkl (recommended for stable supervision)
  <data_folder>/testXX/label_obs.pkl (matches policy observation domain)
"""

from __future__ import annotations

import copy
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

ABS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPTH_MIN = 0.1
DEPTH_MAX = 6.0
LOG_EPS = 0.01

class RayPredictionDataset(Dataset):
    def __init__(self, all_dataset):
        self.data = list(all_dataset.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, lidar_data = self.data[idx]
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).float()
        lidar_tensor = torch.from_numpy(np.asarray(lidar_data, dtype=np.float32)).float()
        return image_tensor, lidar_tensor


class ResNetModel(torch.nn.Module):
    def __init__(self, resnet_type: str, num_rays: int):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        elif resnet_type == "resnet34":
            self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError(f"Unsupported resnet_type: {resnet_type}")
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, num_rays)

    def forward(self, x):
        return self.resnet(x)


def _load_camrec_data(
    data_folder: str,
    num_rays: int,
    label_file: str,
    image_log: bool = True,
    labels_are_log: bool = True,
    log_eps: float = LOG_EPS,
):
    all_dataset = {}
    subdirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    subdirs.sort()

    for folder_name in subdirs:
        folder_path = os.path.join(data_folder, folder_name)
        label_path = os.path.join(folder_path, label_file)
        if not os.path.isfile(label_path):
            print(f"Skip {folder_name}: no {label_file}")
            continue
        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        for fname in os.listdir(folder_path):
            if not fname.endswith(".npy"):
                continue
            image_idx = fname[:-4]
            if image_idx not in labels:
                continue
            image = np.load(os.path.join(folder_path, fname), allow_pickle=True)
            if not isinstance(image, np.ndarray):
                continue
            ray_label = np.asarray(labels[image_idx], dtype=np.float32).flatten()
            if ray_label.size != num_rays:
                continue
            if image_log:
                img = np.asarray(image, dtype=np.float32)
                # torch camera depth can contain inf; clip before log to avoid nan.
                img = np.nan_to_num(img, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=DEPTH_MIN)
                img = np.clip(img, DEPTH_MIN, DEPTH_MAX)
                image_data = np.log(np.clip(img, log_eps, None))
            else:
                img = np.asarray(image, dtype=np.float32)
                img = np.nan_to_num(img, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=DEPTH_MIN)
                image_data = np.clip(img, DEPTH_MIN, DEPTH_MAX)
            if labels_are_log:
                # label is already in log-space; sanitize for safety.
                label_data = np.nan_to_num(
                    ray_label,
                    nan=np.log(log_eps),
                    posinf=np.log(DEPTH_MAX),
                    neginf=np.log(log_eps),
                )
                label_data = np.clip(label_data, np.log(log_eps), np.log(DEPTH_MAX))
            else:
                lab = np.asarray(ray_label, dtype=np.float32)
                lab = np.nan_to_num(lab, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=DEPTH_MIN)
                lab = np.clip(lab, log_eps, DEPTH_MAX)
                label_data = np.log(lab)
            key = f"{folder_name}_{image_idx}"
            all_dataset[key] = [image_data, label_data]
    return all_dataset


def train(args):
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(
        ABS_ROOT,
        "logs",
        "depth_logs_go2",
        f"{run_ts}-{args.resnet_type}-{args.exp_name}",
    )
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    data_folder = args.data_folder or os.path.join(ABS_ROOT, "logs", "rec_cam_go2")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"data_folder not found: {data_folder}")

    all_dataset = _load_camrec_data(
        data_folder,
        num_rays=args.num_rays,
        label_file=args.label_file,
        image_log=args.image_log,
        labels_are_log=args.labels_are_log,
    )
    if len(all_dataset) == 0:
        raise ValueError(f"No valid samples in {data_folder}")

    dataset = RayPredictionDataset(all_dataset)
    test_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - test_size
    if train_size <= 0:
        raise ValueError("Not enough data to split train/test. Collect more samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Train/Test samples:", len(train_dataset), len(test_dataset))

    model = ResNetModel(args.resnet_type, num_rays=args.num_rays).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.MSELoss()

    log_min = np.log(LOG_EPS)
    log_max = np.log(6.0)

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
                height, width = inputs.size(1), inputs.size(2)
                erase_h = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max + 1)
                erase_w = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max + 1)
                if height > erase_h and width > erase_w:
                    x = np.random.randint(0, width - erase_w)
                    y = np.random.randint(0, height - erase_h)
                    v = np.random.choice([log_min, log_max])
                    inputs = torchvision.transforms.functional.erase(inputs, x, y, erase_h, erase_w, v=v)

            inputs = inputs.to(device).unsqueeze(1).repeat(1, 3, 1, 1)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                print(
                    f"Epoch {epoch}/{args.num_epochs} "
                    f"[{batch_idx * len(inputs)}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.6f}"
                )
                writer.add_scalar("Train/Loss", loss.item(), global_step)

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
        writer.add_scalar("Test/Loss", test_loss, epoch)

        if epoch % args.save_interval == 0:
            model_path = os.path.join(log_folder, f"depth_lidar_model_{run_ts}_{epoch}.pt")
            model_cpu = copy.deepcopy(model).cpu()
            try:
                torch.jit.script(model_cpu).save(model_path)
            except Exception:
                state_path = model_path.replace(".pt", "_state.pt")
                torch.save(model_cpu.state_dict(), state_path)
                print("TorchScript export failed, saved state_dict:", state_path)
            print("Model saved:", model_path)

    writer.close()


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Go2 depth->lidar ray prediction training.")
    parser.add_argument("--data_folder", type=str, default=None, help="Path to rec_cam_go2 folder")
    parser.add_argument("--num_rays", type=int, default=11, help="Number of lidar rays (Go2 default: 11)")
    parser.add_argument(
        "--label_file",
        type=str,
        default="label_raw.pkl",
        choices=["label_raw.pkl", "label_obs.pkl"],
        help="Which label file to use from each rollout folder",
    )
    parser.add_argument(
        "--image_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use natural-log transform for depth images before training",
    )
    parser.add_argument(
        "--labels_are_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set true for labels generated by go2/camrec.py (already log-space)",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=302)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--resnet_type", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--no_leftright", action="store_true", help="Disable left-right augmentation")
    parser.add_argument("--noise_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise_augmentation_std_max", type=float, default=0.3)
    parser.add_argument("--noise_augmentation_mean", type=float, default=0.0)
    parser.add_argument("--noise_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--gaussian_blur_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian_blur_augmentation_kernel_size", type=int, default=5)
    parser.add_argument("--gaussian_blur_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--random_erasing_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_erasing_augmentation_prob", type=float, default=0.5)
    parser.add_argument("--random_erasing_area_min", type=int, default=5)
    parser.add_argument("--random_erasing_area_max", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
