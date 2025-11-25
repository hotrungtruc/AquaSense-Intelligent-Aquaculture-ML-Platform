CONFIG = {
    # ===== EXPERIMENT =====
    "exp_name": "Cnn6_Colab_Run",
    "model_type": "Cnn6",

    # ===== TRAINING HYPERPARAMS =====
    "batch_size": 128,
    "num_epochs": 100,
    "learning_rate": 1e-3,

    # ===== AUDIO PARAMS =====
    "sample_rate": 32000,
    "window_size": 2048,
    "hop_size": 1024,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": None,
    "classes_num": 4,

    # ===== DATALOADER =====
    "train_ratio": 0.8,
    "split_ratios": (0.8, 0.1, 0.1),
    "seed": 42,
    "num_workers": 4,
    "pin_memory": True,

    # ===== WORKSPACE =====
    "workspace_dir": "./Fish_workspace",
}
