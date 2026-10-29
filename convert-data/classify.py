import os
import shutil

def main():
    base_dir = r"d:\DOCUMENTS\XLTN\GitHub\phu-khac-anh-dep-trai\data"
    source_dir = os.path.join(base_dir, "ASVspoof2017_V2_train")
    txt_file = os.path.join(source_dir, "ASVspoof2017_V2_train.trn.txt")

    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    moved_real = 0
    moved_fake = 0
    not_found = 0

    print("Đang đọc file nhãn và phân loại...")
    if not os.path.exists(txt_file):
        print(f"Không tìm thấy file: {txt_file}")
        return

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                label = parts[1]
                
                src_path = os.path.join(source_dir, filename)
                if os.path.exists(src_path):
                    if label == 'genuine':
                        shutil.move(src_path, os.path.join(real_dir, filename))
                        moved_real += 1
                    elif label == 'spoof':
                        shutil.move(src_path, os.path.join(fake_dir, filename))
                        moved_fake += 1
                else:
                    not_found += 1

    print(f"✅ Đã chuyển vào thư mục real: {moved_real} file")
    print(f"✅ Đã chuyển vào thư mục fake: {moved_fake} file")
    if not_found > 0:
        print(f"⚠️ Có {not_found} file được liệt kê trong txt nhưng không tồn tại trong thư mục.")

if __name__ == "__main__":
    main()
