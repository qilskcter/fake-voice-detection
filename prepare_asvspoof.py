"""Chuyển ASVspoof2019 sang cấu trúc data/real + data/fake cho training.py.

Ví dụ (PA = replay attack):
  python prepare_asvspoof.py ^
    --root "D:/ASVspoof2019/PA/ASVspoof2019_PA_train" ^
    --protocol "D:/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt" ^
    --out data --max-per-class 2000
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục chứa các file .flac (hoặc thư mục cha của 'flac')")
    ap.add_argument("--protocol", required=True, help="file protocol .txt")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-per-class", type=int, default=2000, help="giới hạn số file mỗi lớp (0 = lấy hết)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)

    # Lập chỉ mục tên file -> đường dẫn thật
    index = {p.stem: p for p in root.rglob("*.flac")}
    print(f"Tìm thấy {len(index)} file .flac trong {root}")

    items = {"real": [], "fake": []}
    with open(args.protocol, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            speaker, utt, key = parts[0], parts[1], parts[-1].lower()
            if utt not in index:
                continue
            items["real" if key == "bonafide" else "fake"].append((index[utt], utt, speaker))

    rng = random.Random(args.seed)
    for cls, lst in items.items():
        rng.shuffle(lst)
        if args.max_per_class > 0:
            lst = lst[: args.max_per_class]
        dst = out / cls
        dst.mkdir(parents=True, exist_ok=True)
        for src, utt, speaker in lst:
            # tên dạng REAL-<utt>-<SPEAKER> để speaker_of() trong training.py đọc đúng
            name = f"{cls.upper()}-{utt}-{speaker.replace('-', '_')}.flac"
            shutil.copy2(src, dst / name)
        print(f"{cls}: {len(lst)} file -> {dst}")


if __name__ == "__main__":
    main()
