import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from models import CRNNWithAttn


class AudioUtil:

    @staticmethod
    def open(audio_file):
        """Open audio file with multiple fallback backends (soundfile, librosa, ffmpeg).

        Solves libsndfile errors with .m4a, .aac, .mp3 and other compressed formats.
        """
        try:
            sig, sr = torchaudio.load(audio_file)
            return sig, sr
        except Exception:
            pass

        # Fallback 1: Try ffmpeg backend in torchaudio
        try:
            sig, sr = torchaudio.load(audio_file, backend="ffmpeg")
            return sig, sr
        except Exception:
            pass

        # Fallback 2: Try librosa if available
        try:
            import librosa
            sig_np, sr = librosa.load(audio_file, sr=None, mono=False)
            if sig_np.ndim == 1:
                sig_np = sig_np[None, :]
            return torch.from_numpy(sig_np).float(), sr
        except Exception:
            pass

        # Fallback 3: Convert file using ffmpeg CLI to a temporary WAV file
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
            try:
                cmd = [
                    ffmpeg_bin, "-y", "-i", str(audio_file),
                    "-ar", "16000", "-ac", "2", "-f", "wav", tmp_wav_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                sig, sr = torchaudio.load(tmp_wav_path)
                return sig, sr
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_wav_path):
                    try:
                        os.remove(tmp_wav_path)
                    except OSError:
                        pass

        # Final attempt (will raise standard torchaudio error if all fallbacks fail)
        return torchaudio.load(audio_file)

    @staticmethod
    def rechannel(aud, new_channel=1):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        # Always average channels to mono to prevent stereo vs mono shortcut
        resig = sig.mean(dim=0, keepdim=True)
        if new_channel > 1:
            resig = resig.repeat(new_channel, 1)
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
        self.duration = 2500
        self.sr = 16000
        self.channel = 1
        self.shift_pct = 0.2
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

        sgram = AudioUtil.spectro_gram(aud, n_mels=128, n_fft=780, hop_len=195)

        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=random.randint(1,3), n_time_masks=random.randint(1,3))

        # Per-instance standardization + ImageNet normalization
        spec_mean = sgram.mean()
        spec_std = sgram.std() + 1e-6
        sgram = (sgram - spec_mean) / spec_std

        imagenet_mean = torch.tensor([0.485]).view(1, 1, 1)
        imagenet_std = torch.tensor([0.229]).view(1, 1, 1)
        sgram = (sgram * imagenet_std) + imagenet_mean
        sgram = (sgram - imagenet_mean) / imagenet_std

        return sgram, class_id


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_paths(data_dir):
    """Load audio files from data/fake and data/real.

    Label mapping:
    - 0: Fake (Replay Attack - Giọng nói bị ghi âm và phát lại qua thiết bị loa)
    - 1: Real (Live Voice - Giọng người thật đang nói trực tiếp)
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
    epochs=15,
    batch_size=16,
    learning_rate=1e-4,
    validation_split=0.2,
    seed=42,
    reset_model=False,
):
    """Fine-tune CRNNWithAttn to classify Live Voice (Real: 1) vs Replay Attack (Fake: 0)."""
    torch.manual_seed(seed)
    paths, labels = load_audio_paths(data_dir)
    dataset = SoundDS(paths, labels, mode="combined")
    train_indices, validation_indices = split_indices(labels, validation_split, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using hardware accelerator: {device.type.upper()}")
    checkpoint = Path(model_path)
    use_existing = checkpoint.exists() and not reset_model

    model = CRNNWithAttn(pretrained=not use_existing).to(device)
    if use_existing:
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            print(f"Loaded existing model checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Notice: Loading existing checkpoint failed ({e}). Initializing with clean pre-trained weights.")
            model = CRNNWithAttn(pretrained=True).to(device)
    else:
        print("Initializing model with clean ImageNet pre-trained ResNet18 backbone...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = -1.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {epochs} epochs on {len(train_indices)} train samples, {len(validation_indices)} val samples...")

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
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for features, target in validation_loader:
                preds = (torch.sigmoid(model(features.to(device))) >= 0.5).squeeze(1).cpu()
                target_bool = target.bool()
                tp += ((preds == 1) & (target_bool == 1)).sum().item()
                fp += ((preds == 1) & (target_bool == 0)).sum().item()
                fn += ((preds == 0) & (target_bool == 1)).sum().item()
                tn += ((preds == 0) & (target_bool == 0)).sum().item()

        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        average_loss = train_loss / len(train_indices)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Acc: {acc:.2%} | "
            f"Precision: {precision:.2%} | "
            f"Recall: {recall:.2%} | "
            f"F1-Score: {f1:.4f}"
        )

        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  --> Saved best model checkpoint (Val Acc: {acc:.2%}) to {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fake/real audio classifier")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/best_model10.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--reset-model", action="store_true", help="Start training from clean pre-trained ResNet weights")
    args = parser.parse_args()
    train(**vars(args))