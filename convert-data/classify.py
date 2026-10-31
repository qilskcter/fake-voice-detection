"""
Script tiện ích classify.py:
Đọc file protocol nhãn văn bản (.txt) của tập dữ liệu ASVspoof 2017 Version 2 (Train Set),
sau đó tự động phân loại và di chuyển các file âm thanh vào 2 thư mục tương ứng:
- 'genuine' (giọng người thật) -> Di chuyển vào thư mục 'real'
- 'spoof' (giọng phát lại qua loa) -> Di chuyển vào thư mục 'fake'
"""

import os      # Thao tác với hệ thống tệp tin (nối đường dẫn, kiểm tra sự tồn tại của file, tạo thư mục)
import shutil  # Thao tác di chuyển tệp tin (shutil.move) an toàn giữa các thư mục


def main():
    # 1. Định cấu hình đường dẫn thư mục nguồn và file nhãn txt
    base_dir = r"data"
    source_dir = os.path.join(base_dir, "ASVspoof2017_V2_train")
    txt_file = os.path.join(source_dir, "ASVspoof2017_V2_train.trn.txt")

    # 2. Thư mục đích phân loại
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    # Tạo thư mục real và fake nếu chưa có
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Các biến đếm số lượng thống kê
    moved_real = 0   # Đếm số file thật
    moved_fake = 0   # Đếm số file giả
    not_found = 0    # Đếm số file có trong danh sách nhưng không tìm thấy trên đĩa cứng

    print("Đang đọc file nhãn protocol và tiến hành phân loại...")
    if not os.path.exists(txt_file):
        print(f"Không tìm thấy file: {txt_file}")
        return

    # 3. Đọc từng dòng trong file nhãn .txt
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # Mỗi dòng chứa: <Tên file> <Nhãn: genuine hoặc spoof>
            if len(parts) >= 2:
                filename = parts[0]
                label = parts[1]

                src_path = os.path.join(source_dir, filename)
                # Kiểm tra nếu file âm thanh thực sự tồn tại trong thư mục nguồn
                if os.path.exists(src_path):
                    if label == 'genuine':
                        # Di chuyển vào thư mục real
                        shutil.move(src_path, os.path.join(real_dir, filename))
                        moved_real += 1
                    elif label == 'spoof':
                        # Di chuyển vào thư mục fake
                        shutil.move(src_path, os.path.join(fake_dir, filename))
                        moved_fake += 1
                else:
                    not_found += 1

    # 4. In kết quả thống kê chi tiết
    print(f"✅ Đã chuyển vào thư mục real: {moved_real} file")
    print(f"✅ Đã chuyển vào thư mục fake: {moved_fake} file")
    if not_found > 0:
        print(f"⚠️ Có {not_found} file được liệt kê trong txt nhưng không tồn tại trong thư mục.")


if __name__ == "__main__":
    main()
