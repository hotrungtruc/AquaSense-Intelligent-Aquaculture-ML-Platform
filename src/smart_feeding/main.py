# main.py

import os
import sys
import time
import logging as log_config
import torch
import torch.optim as optim
import numpy as np

from .models import Cnn6
from .losses import get_loss_func
from .fish_voice_dataset import data_generator, Fish_Voice_Dataset
from .ultils import make_loaders_from_full_dataset
from .train import train_loop
from .config import CONFIG   

def main():
    # --- 1. Load config ---
    exp_name = CONFIG["exp_name"]
    model_type = CONFIG["model_type"]

    batch_size = CONFIG["batch_size"]
    num_epochs = CONFIG["num_epochs"]
    learning_rate = CONFIG["learning_rate"]

    sample_rate = CONFIG["sample_rate"]
    window_size = CONFIG["window_size"]
    hop_size = CONFIG["hop_size"]
    mel_bins = CONFIG["mel_bins"]
    fmin = CONFIG["fmin"]
    fmax = CONFIG["fmax"]
    classes_num = CONFIG["classes_num"]

    workspace_dir = CONFIG["workspace_dir"]

    # --- 2. Folder & Logging ---
    ckpt_dir = os.path.join(workspace_dir, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_name = os.path.join(ckpt_dir, 'best.pt')

    log_dir = os.path.join(workspace_dir, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir, f'{exp_name}-{int(time.time())}.log')),
            log_config.StreamHandler(sys.stdout)
        ]
    )
    logger = log_config.getLogger()

    # --- 3. Initialize model ---
    model_params = {
        "sample_rate": sample_rate,
        "window_size": window_size,
        "hop_size": hop_size,
        "mel_bins": mel_bins,
        "fmin": fmin,
        "fmax": fmax,
        "classes_num": classes_num
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Thí nghiệm đang chạy trên: {device}")

    Model = eval(model_type)
    model = Model(**model_params).to(device)
    logger.info(f"Đã tải mô hình {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    loss_func = get_loss_func('clip_ce')

    # --- 4. Load original dataset ---
    logger.info("Loading original dataset...")
    train_list, test_list = data_generator(seed=20, train_ratio=CONFIG["train_ratio"])
    all_data_list = train_list + test_list
    dataset_full = Fish_Voice_Dataset(data_list=all_data_list, sample_rate=sample_rate)

    logger.info(f"Loaded full dataset with {len(dataset_full)} samples.")

    # stratify labels
    stratify_labels = [int(item[1]) for item in all_data_list]

    # --- 5. Split into train / valid / test ---
    logger.info("Creating Dataloaders...")

    train_loader, train_eval_loader, valid_loader, test_loader = make_loaders_from_full_dataset(
        dataset_full,
        batch_size=batch_size,
        ratios=CONFIG["split_ratios"],
        seed=CONFIG["seed"],
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        stratify_labels=stratify_labels
    )

    if len(train_loader.dataset) == 0:
        logger.error("Train set is empty - please check the dataset.")
        return

    # --- 6. Train ---
    train_loop(
        model, train_loader, train_eval_loader, valid_loader, test_loader,
        num_epochs, device, optimizer, loss_func, best_ckpt_name, logger
    )

if __name__ == '__main__':
    main()
