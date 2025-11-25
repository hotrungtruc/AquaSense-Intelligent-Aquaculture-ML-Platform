import os
import argparse
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from .models import Cnn6
from .fish_voice_dataset import get_dataloader


LABELS = ['None', 'Strong', 'Medium', 'Weak']


def load_checkpoint_to_model(ckpt_path: str, model_class, model_kwargs: dict, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = model_class(**model_kwargs)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_and_collect(model, dataloader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run the model on `dataloader` and return (clipwise_output, target, audio_names).

    clipwise_output: numpy array shape (N, C)
    target: numpy array shape (N, C) (one-hot)
    audio_names: list of strings
    """
    all_preds = []
    all_targets = []
    all_names = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            names = batch.get('audio_name', [])
            wav = batch['waveform'].to(device)
            target = batch.get('target', None)

            out = model(wav, mixup_lambda=None)
            preds = out['clipwise_output']
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            else:
                preds = np.array(preds)

            if isinstance(target, torch.Tensor):
                target_np = target.detach().cpu().numpy()
            else:
                target_np = np.array(target)

            all_preds.append(preds)
            all_targets.append(target_np)
            all_names.extend(names)

    clipwise_output = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.zeros((clipwise_output.shape[0], len(LABELS)))
    return clipwise_output, target, all_names


def compute_metrics_and_plot(clipwise_output: np.ndarray, target: np.ndarray, labels: List[str] = LABELS, title_prefix: str = 'Test', save_dir: str = None):
    """
    Compute per-class mAP/AUC and overall accuracy and plot results.
    Returns a dict with computed metrics.
    """
    if clipwise_output.shape[0] != target.shape[0]:
        raise ValueError('Predictions and targets have different numbers of samples')

    # mAP & AUC per class (safe fallback to zeros)
    try:
        ap = metrics.average_precision_score(target, clipwise_output, average=None)
    except Exception:
        ap = np.zeros(target.shape[1])

    try:
        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
    except Exception:
        auc = np.zeros(target.shape[1])

    y_true = np.argmax(target, axis=1)
    y_pred = np.argmax(clipwise_output, axis=1)
    acc = accuracy_score(y_true, y_pred)

    print(f"{title_prefix} Accuracy: {acc:.4f}")
    print(f"{title_prefix} mAP per class: {np.round(ap, 4)}")
    print(f"{title_prefix} AUC per class: {np.round(auc, 4)}")

    # Bar plot: mAP & AUC
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, ap, width, label='mAP')
    ax.bar(x + width/2, auc, width, label='AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'{title_prefix} - mAP & AUC per class')
    ax.legend()
    for i, (a, b) in enumerate(zip(ap, auc)):
        ax.text(i - width/2, a + 0.02, f"{a:.2f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, b + 0.02, f"{b:.2f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        bar_path = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_map_auc.png")
        fig.savefig(bar_path)
        print(f"Saved mAP/AUC bar chart to {bar_path}")
    else:
        plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label',
           xlabel='Predicted label',
           title=f'{title_prefix} Confusion Matrix (N={len(y_true)})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2. if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    if save_dir is not None:
        cm_path = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_confusion_matrix.png")
        fig.savefig(cm_path)
        print(f"Saved confusion matrix to {cm_path}")
    else:
        plt.show()

    # Save metrics CSV
    if save_dir is not None:
        metrics_df = pd.DataFrame({
            'label': labels,
            'mAP': np.round(ap, 6),
            'AUC': np.round(auc, 6)
        })
        metrics_csv = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_per_class_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved per-class metrics to {metrics_csv}")

    return {'accuracy': acc, 'mAP': ap, 'AUC': auc, 'confusion_matrix': cm}


def plot_map_history(csv_path: str):
    """Plot mAP history stored in a CSV file with a 'mAP' column (or fallback to first column)."""
    if not os.path.exists(csv_path):
        print(f"mAP history file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if 'mAP' in df.columns:
        y = df['mAP'].values
    else:
        y = df.iloc[:, 0].values
    epochs = np.arange(1, len(y) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, y, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP history')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate & visualize Cnn6 model results')
    p.add_argument('--ckpt', default=os.path.join('models', 'smart_feeding', 'best.pt'), help='Path to checkpoint')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--sample-rate', type=int, default=32000)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--map-csv', default=None, help='Path to training history CSV (optional)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_params = {
        'sample_rate': args.sample_rate,
        'window_size': 2048,
        'hop_size': 1024,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': None,
        'classes_num': len(LABELS)
    }

    try:
        model = load_checkpoint_to_model(args.ckpt, Cnn6, model_params, device)
    except Exception as e:
        print('Error loading checkpoint:', e)
        return

    try:
        test_loader = get_dataloader(split='test', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=False, drop_last=False, num_workers=args.num_workers)
    except Exception as e:
        print('Error creating test DataLoader:', e)
        return

    clipwise_output, target, audio_names = evaluate_and_collect(model, test_loader, device)

    # derive experiment directory from checkpoint: ../<exp_name>/save_models/best.pt -> exp_dir
    exp_dir = os.path.dirname(os.path.dirname(args.ckpt))
    plots_dir = os.path.join(exp_dir, 'plots')
    compute_metrics_and_plot(clipwise_output, target, labels=LABELS, title_prefix='Best model on TEST', save_dir=plots_dir)

    # training history CSV: use provided or derive from exp_dir
    map_csv = args.map_csv if args.map_csv is not None else os.path.join(exp_dir, 'training_history.csv')
    if os.path.exists(map_csv):
        plot_map_history(map_csv)
    else:
        print(f"Training history CSV not found at {map_csv}")


if __name__ == '__main__':
    main()