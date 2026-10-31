"""
Script tiện ích prepare_asvspoof.py:
Trích xuất và chuyển đổi tập dữ liệu chuẩn quốc tế ASVspoof 2019 (Physical Access - PA)
sang cấu trúc phân loại 2 thư mục: data/real và data/fake cho hệ thống huấn luyện.

Cách hoạt động:
1. Đọc file protocol chính thức (.txt) của ASVspoof 2019 PA.
2. Bóc tách nhãn của từng câu thoại:
   - 'bonafide' (giọng người thật nói trực tiếp) -> Phân loại vào thư mục 'real'
   - 'spoof' (giọng ghi âm phát lại qua loa thiết bị) -> Phân loại vào thư mục 'fake'
3. Đổi tên file theo quy ước chuẩn: <CLASS>-<UTTERANCE_ID>-<SPEAKER_ID>.flac
   Ví dụ: REAL-PA_T_0001234-PA_0083.flac
   Quy ước này giúp hàm speaker_of() trong training.py dễ dàng nhận diện và tách người nói.
4. Sao chép các file âm thanh vào thư mục đích tương ứng.

Ví dụ cách chạy dòng lệnh:
  python prepare_asvspoof.py ^
    --root "D:/ASVspoof2019/PA/ASVspoof2019_PA_train" ^
    --protocol "D:/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt" ^
    --out data --max-per-class 2000
"""

# ==============================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import argparse      # Phân tích các cờ đối số truyền từ dòng lệnh (CLI options)
import random        # Sinh số ngẫu nhiên để xáo trộn danh sách file
import shutil        # Các thao tác sao chép file an toàn trên hệ điều hành
from pathlib import Path  # Xử lý đường dẫn file/thư mục hướng đối tượng


# ==============================================================================
# 2. HÀM XỬ LÝ CHÍNH (main)
# ==============================================================================
def main():
    # Khai báo bộ phân tích cú pháp tham số dòng lệnh
    ap = argparse.ArgumentParser(description="Chuyển ASVspoof sang cấu trúc data/real và data/fake")
    ap.add_argument("--root", required=True, help="Đường dẫn thư mục chứa các file âm thanh gốc .flac")
    ap.add_argument("--protocol", required=True, help="Đường dẫn tới file danh sách nhãn protocol (.txt)")
    ap.add_argument("--out", default="data", help="Thư mục đích lưu trữ kết quả (mặc định: data)")
    ap.add_argument("--max-per-class", type=int, default=2000,
                    help="Giới hạn số file tối đa lấy cho mỗi lớp (đặt 0 nếu muốn lấy toàn bộ dữ liệu)")
    ap.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên để xáo trộn dữ liệu cố định (mặc định: 42)")
    args = ap.parse_args()

    # Chuyển đổi đường dẫn sang đối tượng Path
    root, out = Path(args.root), Path(args.out)

    # 1. Quét và lập chỉ mục tên file gốc trong thư mục root:
    # { 'PA_T_0000001': Path('.../PA_T_0000001.flac') }
    index = {p.stem: p for p in root.rglob("*.flac")}
    print(f"Tìm thấy {len(index)} file .flac trong {root}")

    # 2. Đọc file protocol để phân loại từng câu thoại vào 'real' hoặc 'fake'
    items = {"real": [], "fake": []}
    with open(args.protocol, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            # Mỗi dòng hợp lệ trong file protocol có ít nhất 5 trường thông tin:
            # speaker_id, utterance_id, environment_id, attack_id, key (bonafide hoặc spoof)
            if len(parts) < 5:
                continue

            speaker, utt, key = parts[0], parts[1], parts[-1].lower()

            # Nếu file âm thanh tồn tại trên ổ cứng thì mới thêm vào danh sách
            if utt not in index:
                continue

            # Ánh xạ nhãn: 'bonafide' -> real, 'spoof' -> fake
            target_class = "real" if key == "bonafide" else "fake"
            items[target_class].append((index[utt], utt, speaker))

    # 3. Xáo trộn ngẫu nhiên và sao chép dữ liệu sang cấu trúc thư mục mới
    rng = random.Random(args.seed)
    for cls, lst in items.items():
        # Xáo trộn danh sách để lấy mẫu ngẫu nhiên đồng đều
        rng.shuffle(lst)

        # Cắt bớt nếu người dùng cài đặt giới hạn max_per_class
        if args.max_per_class > 0:
            lst = lst[: args.max_per_class]

        # Tạo thư mục đích: data/real hoặc data/fake
        dst = out / cls
        dst.mkdir(parents=True, exist_ok=True)

        # Sao chép và đổi tên file
        for src, utt, speaker in lst:
            # Đổi tên dạng: REAL-<utt>-<SPEAKER>.flac (ví dụ: REAL-PA_T_0005401-PA_0079.flac)
            # Dấu '-' cuối cùng dùng để tách Speaker ID trong training.py
            name = f"{cls.upper()}-{utt}-{speaker.replace('-', '_')}.flac"
            shutil.copy2(src, dst / name)

        print(f"Đã xử lý lớp {cls.upper()}: {len(lst)} file -> lưu tại {dst}")


if __name__ == "__main__":
    main()
