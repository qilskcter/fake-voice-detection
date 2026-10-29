import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from models import CRNNWithAttn

try:
    import av  # PyAV (bundled ffmpeg) - decodes ASVspoof FLAC files that libsndfile rejects
    HAS_AV = True
except ImportError:
    HAS_AV = False


def _open_with_pyav(audio_file):
    """Decode any audio file with PyAV. Returns (tensor [channels, samples] float32 in [-1, 1], sr)."""
    with av.open(str(audio_file)) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            n_ch = len(frame.layout.channels)
            # Packed formats come as (1, samples * channels) interleaved -> (channels, samples)
            if not frame.format.is_planar and n_ch > 1:
                arr = arr.reshape(-1, n_ch).T
            chunks.append(arr)
    if not chunks:
        raise RuntimeError(f"PyAV decoded no audio from {audio_file}")
    x = np.concatenate(chunks, axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(x)), sr


class AudioUtil:

    @staticmethod
    def open(audio_file):
        """Open audio file with multiple fallback backends (PyAV, soundfile, librosa, ffmpeg).

        PyAV is tried first because libsndfile fails ("flac decoder lost sync") on
        many ASVspoof .flac files that are actually fine.
        """
        # Primary: PyAV (bundled ffmpeg)
        if HAS_AV:
            try:
                return _open_with_pyav(audio_file)
            except Exception:
                pass

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
    def fix_length(aud, max_ms, random_crop=False):
        """Make every clip exactly max_ms long WITHOUT inserting digital silence.

        - longer clips: random crop (train) or center crop (eval)
        - shorter clips: tile (repeat) the signal, then crop
        Zero-padding is avoided because it can become a shortcut when one class
        has systematically shorter clips than the other.
        """
        sig, sr = aud
        max_len = sr // 1000 * max_ms
        sig_len = sig.shape[1]
        if sig_len < max_len:
            reps = -(-max_len // sig_len)  # ceil division
            sig = sig.repeat(1, reps)
            sig_len = sig.shape[1]
        if sig_len > max_len:
            if random_crop:
                start = random.randint(0, sig_len - max_len)
            else:
                start = (sig_len - max_len) // 2
            sig = sig[:, start:start + max_len]
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
    """Audio dataset.

    Returns a Mel-spectrogram [1, 128, T] for the CRNN.
    train=True enables random crop; mode controls augmentation. Use
    mode="original" and train=False for validation/test.
    """

    def __init__(self, data_path, label, mode="original", train=True):
        self.label = label
        self.data_path = [str(p) for p in data_path]
        self.duration = 2500
        self.sr = 16000
        self.channel = 1
        self.shift_pct = 0.2
        self.mode = mode
        self.train = train

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        audio_file = self.data_path[idx]
        class_id = self.label[idx]

        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, self.channel)
        aud = AudioUtil.fix_length(aud, self.duration, random_crop=self.train)

        if self.mode == "time_shift":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
        elif self.mode == "add_noise":
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))
        elif self.mode == "pitch_shift":
            aud = AudioUtil.pitch_shift(aud, n_steps=random.randint(1, 3))
        elif self.mode == "combined":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))

        sgram = AudioUtil.spectro_gram(aud, n_mels=128, n_fft=780, hop_len=195)

        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=random.randint(1, 3), n_time_masks=random.randint(1, 3))

        # Per-instance standardization.
        # (The old extra ImageNet step "(x*std+mean-mean)/std" cancelled itself out, so it was removed.)
        sgram = (sgram - sgram.mean()) / (sgram.std() + 1e-6)
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


def speaker_of(path):
    """Speaker id = last '-' separated token of the file name.

    'FAKE-XIN_CHAO-DAT.flac' -> 'DAT'. ADAPT THIS for external datasets
    (e.g. parse the speaker id from the corpus metadata/protocol file).
    """
    return Path(path).stem.split("-")[-1].upper()


def split_by_speaker(paths, labels, val_speakers=None, validation_split=0.2, seed=42):
    """Speaker-disjoint split: no speaker appears in both train and validation."""
    speakers = sorted({speaker_of(p) for p in paths})
    if val_speakers:
        val_set = {s.upper() for s in val_speakers}
        unknown = val_set - set(speakers)
        if unknown:
            raise ValueError(f"Unknown speakers: {sorted(unknown)}. Available: {speakers}")
    else:
        rng = random.Random(seed)
        shuffled = speakers[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_split))
        val_set = set(shuffled[:n_val])

    train_idx = [i for i, p in enumerate(paths) if speaker_of(p) not in val_set]
    val_idx = [i for i, p in enumerate(paths) if speaker_of(p) in val_set]
    for name, idx in (("train", train_idx), ("validation", val_idx)):
        if len({labels[i] for i in idx}) < 2:
            raise ValueError(f"The {name} split must contain both classes. Validation speakers: {sorted(val_set)}")
    print(f"Validation speakers: {sorted(val_set)} | train speakers: {sorted(set(speakers) - val_set)}")
    return train_idx, val_idx


def compute_eer(scores, labels):
    """Equal Error Rate. labels: 1 = Real (positive), 0 = Fake. scores: P(real)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    y = y[np.argsort(-s)]
    tpr = np.cumsum(y) / pos
    fpr = np.cumsum(1 - y) / neg
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[k] + fpr[k]) / 2)


def train(
    data_dir="data",
    model_path=None,
    epochs=15,
    batch_size=16,
    learning_rate=1e-4,
    validation_split=0.2,
    val_speakers=None,
    seed=42,
    reset_model=False,
    final=False,
    save=True,
):
    """Train Live Voice (Real: 1) vs Replay Attack (Fake: 0).

    final=True : train on ALL speakers, no validation, save the last epoch (for the app).
    save=False : do not write a checkpoint (used by cross-validation).
    Returns (EER, accuracy) of the LAST epoch on the validation speaker(s), or None if final.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    model_path = model_path or "models/best_crnn.pth"

    if not HAS_AV:
        print("Warning: PyAV not installed (pip install av). Falling back to torchaudio/soundfile, "
              "which may fail on some .flac files.")

    paths, labels = load_audio_paths(data_dir)
    if final:
        train_indices, validation_indices = list(range(len(paths))), []
        print("FINAL mode: training on all speakers, no validation.")
    else:
        train_indices, validation_indices = split_by_speaker(paths, labels, val_speakers, validation_split, seed)

    def subset(indices, **kwargs):
        return SoundDS([paths[i] for i in indices], [labels[i] for i in indices], **kwargs)

    # Validation must NOT be augmented and must be deterministic.
    train_ds = subset(train_indices, mode="combined", train=True)
    val_ds = subset(validation_indices, mode="original", train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = None if final else DataLoader(val_ds, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Model: crnn | accelerator: {device.type.upper()}")
    checkpoint = Path(model_path)
    use_existing = checkpoint.exists() and not reset_model

    model = CRNNWithAttn(pretrained=not use_existing).to(device)
    if use_existing:
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            print(f"Loaded existing model checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Notice: Loading existing checkpoint failed ({e}). Using fresh pre-trained weights.")
            model = CRNNWithAttn(pretrained=True).to(device)

    n_real = sum(labels[i] == 1 for i in train_indices)
    n_fake = sum(labels[i] == 0 for i in train_indices)
    pos_weight = torch.tensor([n_fake / max(n_real, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_eer = float("inf")
    last_eer, last_acc = float("nan"), 0.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training {epochs} epochs on {len(train_indices)} train / {len(validation_indices)} val samples...")

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

        average_loss = train_loss / len(train_indices)
        scheduler.step()

        if final:
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {average_loss:.4f}")
            if save:
                torch.save(model.state_dict(), checkpoint)
            continue

        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        all_scores, all_targets = [], []
        with torch.no_grad():
            for features, target in validation_loader:
                prob = torch.sigmoid(model(features.to(device))).squeeze(1).cpu()
                all_scores.extend(prob.tolist())
                all_targets.extend(target.tolist())
                preds = prob >= 0.5
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
        eer = compute_eer(all_scores, all_targets)
        last_eer, last_acc = eer, acc

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Acc: {acc:.2%} | "
            f"Precision: {precision:.2%} | "
            f"Recall: {recall:.2%} | "
            f"F1: {f1:.4f} | "
            f"EER: {eer:.2%}"
        )

        if save and eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), checkpoint)
            print(f"  --> Saved best checkpoint (EER: {eer:.2%}) to {checkpoint}")

    return None if final else (last_eer, last_acc)


def cross_validate(data_dir="data", min_per_class=5, **kwargs):
    """Leave-one-speaker-out: every speaker is held out once, automatically.

    Speakers with fewer than `min_per_class` real or fake clips are always kept in
    training (too few clips to give a meaningful validation score).
    Reports the LAST-epoch score of each fold (no checkpoint picking on the held-out speaker).
    """
    import statistics
    paths, labels = load_audio_paths(data_dir)
    counts = {}
    for pth, lab in zip(paths, labels):
        counts.setdefault(speaker_of(pth), [0, 0])[lab] += 1
    eligible = [s for s, (fake, real) in sorted(counts.items()) if min(fake, real) >= min_per_class]
    print(f"Speakers (fake, real): {counts}")
    print(f"Held-out folds: {eligible}")
    if not eligible:
        raise ValueError("No speaker has enough clips of both classes. Lower --min-per-class or add data.")

    results = {}
    for spk in eligible:
        print(f"\n===== Fold: hold out {spk} =====")
        results[spk] = train(data_dir=data_dir, val_speakers=[spk], reset_model=True, save=False, **kwargs)

    print("\n===== Leave-one-speaker-out summary (last epoch) =====")
    for spk, (eer, acc) in results.items():
        print(f"{spk:>8}: EER {eer:.2%} | Acc {acc:.2%}")
    eers = [e for e, _ in results.values()]
    accs = [a for _, a in results.values()]
    print(f"    MEAN: EER {statistics.mean(eers):.2%} | Acc {statistics.mean(accs):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fake/real audio classifier")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default=None, help="default: models/best_crnn.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2, help="fraction of SPEAKERS used for validation")
    parser.add_argument("--val-speakers", type=lambda v: v.split(","), default=None, help="e.g. DAT,QUANG")
    parser.add_argument("--reset-model", action="store_true", help="Ignore an existing checkpoint")
    parser.add_argument("--cv", action="store_true", help="Automatic leave-one-speaker-out evaluation (no speaker to choose)")
    parser.add_argument("--min-per-class", type=int, default=5, help="--cv: min real AND fake clips for a speaker to be held out")
    parser.add_argument("--final", action="store_true", help="Train on ALL speakers and save the last epoch (model for the app)")
    args = vars(parser.parse_args())
    cv, min_per_class = args.pop("cv"), args.pop("min_per_class")
    if cv:
        for k in ("val_speakers", "reset_model", "final", "validation_split"):
            args.pop(k)
        cross_validate(min_per_class=min_per_class, **args)
    else:
        train(**args)
