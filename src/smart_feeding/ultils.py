import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def move_data_to_device(x, device):
    """
    Safely move different data types to `device`.

    Supports:
      - torch.Tensor -> .to(device)
      - numpy.ndarray -> converted to torch tensor with appropriate dtype
      - python lists -> converted to numpy then torch tensor when possible

    Returns the original object if it cannot be converted.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    # numpy
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            return torch.from_numpy(x.astype(np.float32)).to(device)
        elif np.issubdtype(x.dtype, np.integer):
            return torch.from_numpy(x.astype(np.int64)).to(device)
    # list -> try convert to ndarray
    try:
        arr = np.array(x)
        if arr.size == 0:
            return torch.tensor([]).to(device)
        if np.issubdtype(arr.dtype, np.floating):
            return torch.from_numpy(arr.astype(np.float32)).to(device)
        elif np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64)).to(device)
    except Exception:
        pass
    return x

def do_mixup(x, mixup_lambda):
    """
    Perform mixup on a batch of spectrograms.

    x: Tensor of shape (B, C, T, F) or (B, T, F)
    mixup_lambda: 1D array/tensor with length B (lambda for each sample)

    The implementation pairs even/odd samples: (0 with 1), (2 with 3), ...
    and returns a mixed batch of shape (B/2, ...).
    """
    if isinstance(mixup_lambda, (np.ndarray, list)):
        mixup_lambda = torch.tensor(mixup_lambda, dtype=torch.float32, device=x.device)
    else:
        mixup_lambda = mixup_lambda.to(x.device).float()

    # reshape lambda để broadcast theo x dims
    # lambda shape (B,) -> (B, 1, 1, 1) if x has 4 dims, else (B,1,1)
    if x.dim() == 4:
        lam = mixup_lambda.view(-1, 1, 1, 1)
    elif x.dim() == 3:
        lam = mixup_lambda.view(-1, 1, 1)
    else:
        lam = mixup_lambda.view(-1, 1)

    x_even = x[0::2]
    x_odd = x[1::2]
    lam_even = lam[0::2]

    out = x_even * lam_even + x_odd * (1.0 - lam_even)
    return out

def append_to_dict(res_dict, key, value):
    """
    Helper to append or extend `value` into a list stored at `res_dict[key]`.

    If `value` is a list it will be extended, otherwise appended as a single element.
    """
    if key not in res_dict:
        res_dict[key] = []

    if isinstance(value, list):
        res_dict[key].extend(value)
    else:
        res_dict[key].append(value)

def forward(model, generator, return_target=False):
    """
    Run inference on a data generator and collect outputs.

    Returns a dict with keys:
      - 'audio_name': list of audio names
      - 'clipwise_output': numpy array (N, C)
      - 'target' (optional): numpy array (N, C)
    """
    output_dict = {}
    device = next(model.parameters()).device

    for batch_data_dict in tqdm(generator, desc="Evaluating"):
        # batch_waveform đã là torch.FloatTensor từ collate_fn
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform, None)

        # audio_name: list of strings
        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        # clipwise_output: tensor (B, C)
        clip = batch_output['clipwise_output']
        if isinstance(clip, torch.Tensor):
            clip_np = clip.detach().cpu().numpy()
        else:
            clip_np = np.array(clip)
        append_to_dict(output_dict, 'clipwise_output', clip_np)

        if return_target:
            if 'target' in batch_data_dict:
                t = batch_data_dict['target']
                if isinstance(t, torch.Tensor):
                    t = t.detach().cpu().numpy()
                append_to_dict(output_dict, 'target', t)


    if 'audio_name' in output_dict:
        # ensure it's a flat list already
        audio_names = output_dict['audio_name']
        output_dict['audio_name'] = list(audio_names)

    # các key còn lại là list of numpy arrays -> concatenate
    for key in ['clipwise_output', 'target']:
        if key in output_dict:
            try:
                output_dict[key] = np.concatenate(output_dict[key], axis=0)
            except Exception as e:
                # nếu không concat được, giữ nguyên list để debug
                print(f"Không concat được {key}: {e}")
                output_dict[key] = np.array(output_dict[key], dtype=object)

    return output_dict

def split_indices(n_samples, ratios=(0.8, 0.1, 0.1), seed=42, stratify_labels=None):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = np.random.RandomState(seed)

    if stratify_labels is None:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_train = int(n_samples * ratios[0])
        n_val = int(n_samples * ratios[1])
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()
    else:
        # dùng sklearn để stratify
        idx = np.arange(n_samples)
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            idx, stratify_labels, train_size=ratios[0], random_state=seed, shuffle=True, stratify=stratify_labels)
        # chia temp thành val/test theo tỉ lệ ratios[1]/(ratios[1]+ratios[2])
        if ratios[1] + ratios[2] <= 0:
            raise ValueError("Validation+Test ratios must be > 0")
        val_ratio_rel = ratios[1] / (ratios[1] + ratios[2])
        idx_val, idx_test = train_test_split(
            idx_temp, train_size=val_ratio_rel, random_state=seed, shuffle=True, stratify=y_temp)
        train_idx = idx_train.tolist()
        val_idx = idx_val.tolist()
        test_idx = idx_test.tolist()

    return train_idx, val_idx, test_idx

def make_loaders_from_full_dataset(dataset_full, batch_size=64, ratios=(0.8,0.1,0.1),
                                   seed=42, num_workers=4, pin_memory=True, drop_last=False,
                                   stratify_labels=None):
    """
    Split a full dataset into train/val/test subsets and return DataLoaders.

    Parameters mirror common DataLoader options. If `stratify_labels` is provided
    it will perform stratified splitting using sklearn's `train_test_split`.
    """
    n = len(dataset_full)
    if n == 0:
        raise ValueError("dataset_full rỗng.")
    train_idx, val_idx, test_idx = split_indices(n, ratios=ratios, seed=seed, stratify_labels=stratify_labels)
    train_subset = Subset(dataset_full, train_idx)
    val_subset = Subset(dataset_full, val_idx)
    test_subset = Subset(dataset_full, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    # train_eval: same samples but no shuffle (for evaluating training performance)
    train_eval_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    return train_loader, train_eval_loader, val_loader, test_loader