import time
import os
import sys
import logging as log_config
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from .evaluate import Evaluator
from .models import Cnn6
from .losses import get_loss_func
from .fish_voice_dataset import data_generator, Fish_Voice_Dataset
from .ultils import make_loaders_from_full_dataset


def save_model_checkpoint(path, model, optimizer, ave_precision, epoch):
    """Save a training checkpoint dict to `path`.

    Stores epoch, model_state_dict, optimizer_state_dict and ave_precision.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ave_precision': ave_precision,
    }, path)
    print(f"Đã lưu checkpoint tốt nhất tại epoch {epoch} vào {path}")

# --- train_loop ---
def train_loop(model, train_loader, train_eval_loader, valid_loader, test_loader,
               num_epochs, device, optimizer, loss_func, best_ckpt_name, logger):
    logger.info("Bắt đầu quá trình huấn luyện mới")

    evaluator = Evaluator(model=model) 
    best_mAP = 0.0
    best_message = ''
    history = {
        'train_mAP': [],
        'valid_mAP': [],
        'train_acc': [],
        'valid_acc': [],
        'train_loss': []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        mean_loss = 0.0

        for data_dict in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]"):
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)

            output_dict = model(data_dict['waveform'], mixup_lambda=None)
            target_dict = {'target': data_dict['target']}
            loss = loss_func(output_dict, target_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()

        epoch_loss = mean_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - Training loss: {epoch_loss:.4f}")
        history['train_loss'].append(epoch_loss)

        # ===== Evaluate on train_eval =====
        model.eval()
        train_stats = evaluator.evaluate(train_eval_loader)
        train_mAP = float(np.mean(train_stats['average_precision']))
        train_acc = float(np.mean(train_stats['accuracy']))
        logger.info(f"Epoch {epoch} - Train mAP: {train_mAP:.4f}, Train Acc: {train_acc:.4f}")

        # ===== Evaluate on valid =====
        valid_stats = evaluator.evaluate(valid_loader)
        valid_mAP = float(np.mean(valid_stats['average_precision']))
        valid_acc = float(np.mean(valid_stats['accuracy']))
        valid_msg = valid_stats.get('message', '')
        logger.info(f"Epoch {epoch} - Valid mAP: {valid_mAP:.4f}, Valid Acc: {valid_acc:.4f}")
        logger.info(f"Epoch {epoch} - Valid details: {valid_msg}")

        # Save history
        history['train_mAP'].append(train_mAP)
        history['valid_mAP'].append(valid_mAP)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        # Save best checkpoint according to valid mAP
        if valid_mAP > best_mAP:
            best_mAP = valid_mAP
            best_message = valid_msg
            save_model_checkpoint(best_ckpt_name, model, optimizer, valid_mAP, epoch)
            logger.info(f"Saved new best checkpoint (valid mAP {valid_mAP:.4f}) at epoch {epoch}")

    # End of training
    logger.info("Training completed.")
    logger.info(f'Best valid mAP: {best_mAP:.4f}')
    logger.info(f'Best detailed report (valid): {best_message}')

    # Save history to multi-column CSV
    try:
        df = pd.DataFrame(history)
        # derive experiment directory from best_ckpt_name: ../<exp_name>/save_models/best.pt
        exp_dir = os.path.dirname(os.path.dirname(best_ckpt_name))
        history_path = os.path.join(exp_dir, 'training_history.csv')
        df.to_csv(history_path, index=False)
        logger.info(f"Saved history to '{history_path}'")
    except Exception as e:
        logger.error(f"Could not save history: {e}")

    # (Optional) Final evaluation on test set and print results
    if test_loader is not None:
        test_stats = evaluator.evaluate(test_loader)
        test_mAP = float(np.mean(test_stats['average_precision']))
        test_acc = float(np.mean(test_stats['accuracy']))
        logger.info(f'Final TEST mAP: {test_mAP:.4f}, TEST Acc: {test_acc:.4f}')


# --- MAIN FUNCTION TO RUN EVERYTHING ---

def main():
    # --- 1. Set Parameters ---
    exp_name = 'Cnn6_CRun'
    model_type = 'Cnn6' 
    batch_size = 128   
    learning_rate = 1e-3
    sample_rate = 32000
    window_size = 2048
    num_epochs = 50
    hop_size = 1024
    mel_bins = 64

    # --- 2. Set up Directories and Logging ---
    workspace_dir = 'results/smart_feeding'
    ckpt_dir = os.path.join(workspace_dir, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_name = os.path.join(ckpt_dir, 'best.pt')

    log_dir = os.path.join(workspace_dir, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Set up logger
    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir, f'{exp_name}-{int(time.time())}.log')),
            log_config.StreamHandler(sys.stdout)
        ]
    )
    logger = log_config.getLogger()

    # --- 3. Initialize Model and Data ---
    model_params = {'sample_rate': sample_rate,
                    'window_size': window_size,
                    'hop_size': hop_size,
                    'mel_bins': mel_bins,
                    'fmin': 50,
                    'fmax': None, # fmax=None will use sr/2
                    'classes_num': 4} # 4 classes (None, Strong, Medium, Weak)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Experiment running on: {device}")
    Model = eval(model_type)
    model = Model(**model_params)
    model = model.to(device)
    logger.info(f"Đã tải mô hình {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    loss_func = get_loss_func('clip_ce') # Use Cross Entropy for multi-class

    # START: Build full data list and create loaders
    logger.info("Loading raw data...")
    train_list, test_list = data_generator(seed=20, train_ratio=0.8)
    all_data_list = train_list + test_list
    dataset_full = Fish_Voice_Dataset(data_list=all_data_list, sample_rate=sample_rate)
    logger.info(f"Loaded full dataset with {len(dataset_full)} samples.")
    # extract labels for stratified split
    stratify_labels = [int(item[1]) for item in all_data_list]

    logger.info("Creating Dataloaders...")
    train_loader, train_eval_loader, valid_loader, test_loader = make_loaders_from_full_dataset(
        dataset_full,
        batch_size=batch_size,
        ratios=(0.8,0.1,0.1),
        seed=42,
        num_workers=4,
        pin_memory=True,
        stratify_labels=stratify_labels
    )
 

    # Check if data exists
    if len(train_loader.dataset) == 0:
        logger.error("Train set is empty - check manifest / paths.")
        return

    # Start training: note the parameter order according to the definition of train_loop above
    train_loop(model, train_loader, train_eval_loader, valid_loader, test_loader,
               num_epochs, device, optimizer, loss_func, best_ckpt_name, logger)
# Run main function
if __name__ == '__main__':
    main()
