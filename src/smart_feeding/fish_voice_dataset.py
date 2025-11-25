import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

# use os.path.join so paths work cross-platform
DATA_PATHS = {
    "weak": os.path.join('data', 'Fish_feeding_sounds', 'weak'),
    "strong": os.path.join('data', 'Fish_feeding_sounds', 'strong'),
    "none": os.path.join('data', 'Fish_feeding_sounds', 'None'),
    "middle": os.path.join('data', 'Fish_feeding_sounds', 'middle')
}

CLASS_MAP = {
    "none": 0,
    "strong": 1,
    "middle": 2,
    "weak": 3
}


def load_audio(path, sr=None):
    """
    Load an audio file and resample/pad/truncate to a fixed length when `sr` provided.

    Returns a float32 numpy array (mono). When `sr` is provided, the function
    ensures returned length equals 2 seconds (sr * 2) by truncating or zero-padding.
    """
    # Load audio with original sample rate
    y, orig_sr = librosa.load(path, sr=None)

    if sr is not None:
        target_len = int(sr * 2) # 2 seconds
        if y is None or len(y) == 0:
            print(f"Warning: File {path} is empty, returning zeros.")
            return np.zeros(target_len, dtype=np.float32)
        try:
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        except Exception as e:
            print(f"Error resampling {path}: {e}. Returning zeros.")
            return np.zeros(target_len, dtype=np.float32)
        # ensure length
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        return y.astype(np.float32)
    else:
        return (y.astype(np.float32) if y is not None else np.array([], dtype=np.float32))


def data_generator(seed=20, train_ratio=0.8):
    """
    Create lists of (wav_path, class_id) for train and test splits.

    Uses `DATA_PATHS` and `CLASS_MAP` to discover files under each class directory.
    Returns (train_list, test_list).
    """
    print("Creating data file lists...")
    random_state = np.random.RandomState(seed)

    train_dict_list = []
    test_dict_list = []
    total_files = 0

    for label_name, dir_path in DATA_PATHS.items():
        class_id = CLASS_MAP[label_name]

        # Find all WAV files
        wav_files = glob.glob(os.path.join(dir_path, "**", "*.wav"), recursive=True)

        if not wav_files:
            print(f"WARNING: No .wav files found in {dir_path}")
            continue

        print(f"Found {len(wav_files)} files for class '{label_name}'")
        total_files += len(wav_files)

        # Shuffle
        random_state.shuffle(wav_files)

        # Split according to ratio
        n_train = int(len(wav_files) * train_ratio)
        train_files = wav_files[:n_train]
        test_files  = wav_files[n_train:]

        # Save
        for w in train_files:
            train_dict_list.append([w, class_id])
        for w in test_files:
            test_dict_list.append([w, class_id])

    if total_files == 0:
        print("LỖI: không tìm thấy dữ liệu!")
        return [], []

    # Shuffle train set
    random_state.shuffle(train_dict_list)

    print("— Hoàn tất —")
    print(f"Tổng train: {len(train_dict_list)}")
    print(f"Tổng test : {len(test_dict_list)}")

    return train_dict_list, test_dict_list

class Fish_Voice_Dataset(Dataset):
    def __init__(self, data_list=None, split='train', sample_rate=None):
        """
        A lightweight Dataset for fish feeding audio.

        If `data_list` is provided it should be a list of [path, class_id].
        Otherwise the dataset will build lists using `data_generator`.
        """
        # allow passing a pre-built data_list (list of [path, class_id])
        if data_list is not None:
            self.data_dict = data_list
        else:
            train_dict, test_dict = data_generator(seed=20, train_ratio=0.8)
            if split == 'train':
                self.data_dict = train_dict
            elif split == 'test':
                self.data_dict = test_dict
            else:
                # default to all
                self.data_dict = train_dict + test_dict
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio(wav_name, sr=self.sample_rate)

        # One-hot encode (4 classes)
        target_one_hot = np.eye(4)[target]

        data_dict = {'audio_name': wav_name, 'waveform': wav, 'target': target_one_hot}

        return data_dict

def collate_fn(batch):
    """Collate function to batch items from Fish_Voice_Dataset.

    Converts lists of numpy arrays into torch.FloatTensors and preserves
    a list of audio file names under 'audio_name'.
    """
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}

def get_dataloader(split,
                   batch_size,
                   sample_rate,
                   shuffle=False,
                   drop_last=False,
                   num_workers=1): # SETTING 
    """Convenience wrapper to create a DataLoader for a given split."""

    dataset = Fish_Voice_Dataset(split=split, sample_rate=sample_rate)

    # Enable shuffle for train_loader
    if split == 'train':
        shuffle = True

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)