import argparse
import random
from pathlib import Path

import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio.transforms as transforms

from models import CRNNWithAttn


class AudioUtil:

    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = sig.repeat(new_channel, 1)
        return resig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        resig = transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo], dim=0)
        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), dim=1)
        return sig, sr

    @staticmethod
    def time_shift(aud, shift_pct):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_pct * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    @staticmethod
    def add_noise(aud, noise_level=0.005):
        sig, sr = aud
        noise = torch.randn_like(sig) * noise_level
        return sig + noise, sr

    @staticmethod
    def change_gain(aud, gain_db_range=(-6, 6)):
        sig, sr = aud
        gain_db = random.uniform(*gain_db_range)
        gain = 10 ** (gain_db / 20)
        return sig * gain, sr

    @staticmethod
    def pitch_shift(aud, n_steps=2):
        sig, sr = aud
        return torchaudio.functional.pitch_shift(sig, sr, n_steps), sr

class SoundDS(Dataset):
    def __init__(self, data_path, label, mode="original"):
        self.label = label
        self.data_path = [str(p) for p in data_path]
        self.duration = 4000
        self.sr = 16000
        self.channel = 2
        self.shift_pct = 0.4
        self.mode = mode

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        audio_file = self.data_path[idx]
        class_id = self.label[idx]
        
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, self.channel)
        aud = AudioUtil.pad_trunc(aud, self.duration)

        if self.mode == "time_shift":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
        elif self.mode == "add_noise":
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))
        elif self.mode == "pitch_shift":
            aud = AudioUtil.pitch_shift(aud, n_steps=random.randint(1,3))
        elif self.mode == "combined":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))

        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=780, hop_len=195)

        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=random.randint(1,3), n_time_masks=random.randint(1,3))

        return sgram, class_id


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_paths(data_dir):
    """Load audio files from data/fake and data/real.

    The model uses 0 for fake and 1 for real, matching the prediction
    threshold in main.py.
    """
    data_dir = Path(data_dir)
    paths = []
    labels = []

    for folder_name, label in (("fake", 0), ("real", 1)):
        folder = data_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing data folder: {folder}")

        files = sorted(
            file for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No audio files found in {folder}")
        paths.extend(files)
        labels.extend([label] * len(files))

    return paths, labels


def split_indices(labels, validation_split=0.2, seed=42):
    """Create a reproducible, class-balanced train/validation split."""
    rng = random.Random(seed)
    train_indices = []
    validation_indices = []

    for label in sorted(set(labels)):
        class_indices = [index for index, value in enumerate(labels) if value == label]
        if len(class_indices) < 2:
            raise ValueError("Each class needs at least two audio files for training")
        rng.shuffle(class_indices)
        validation_count = max(1, round(len(class_indices) * validation_split))
        if validation_count >= len(class_indices):
            validation_count = len(class_indices) - 1
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def train(
    data_dir="data",
    model_path="models/best_model10.pth",
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    validation_split=0.2,
    seed=42,
):
    """Fine-tune CRNNWithAttn using files in data/fake and data/real."""
    torch.manual_seed(seed)
    paths, labels = load_audio_paths(data_dir)
    dataset = SoundDS(paths, labels, mode="combined")
    train_indices, validation_indices = split_indices(labels, validation_split, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(model_path)
    model = CRNNWithAttn(pretrained=not checkpoint.exists()).to(device)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"Loaded existing model: {checkpoint}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = -1.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, target in train_loader:
            features = features.to(device)
            target = target.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, target in validation_loader:
                predictions = torch.sigmoid(model(features.to(device))) >= 0.5
                correct += (predictions.cpu().squeeze(1) == target.bool()).sum().item()
                total += target.numel()

        validation_accuracy = correct / total if total else 0.0
        average_loss = train_loss / len(train_indices)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {average_loss:.4f} - validation accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved best model to {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fake/real audio classifier")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/best_model10.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args()
    train(**vars(args))