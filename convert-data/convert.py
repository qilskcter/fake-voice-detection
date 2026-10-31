"""
Script tiện ích convert.py:
Chuyển đổi hàng loạt các file âm thanh định dạng .m4a (thường được thu từ ứng dụng ghi âm điện thoại iOS/Android)
sang định dạng .flac (Free Lossless Audio Codec) chất lượng cao không nén tổn hao, phục vụ quá trình huấn luyện AI.

Thư viện sử dụng:
- pydub (AudioSegment): Thao tác chuyển đổi và nén/giải mã âm thanh đa định dạng (yêu cầu ffmpeg).
"""

import os                   # Thao tác với hệ thống tệp tin (tạo thư mục, nối đường dẫn, quét file)
from pydub import AudioSegment  # Lớp xử lý âm thanh mạnh mẽ của thư viện pydub


def convert_m4a_to_flac(input_path, output_path):
    """
    Chuyển đổi một file âm thanh đơn lẻ từ .m4a sang .flac.
    - input_path: Đường dẫn tới file nguồn (.m4a).
    - output_path: Đường dẫn lưu file đích (.flac).
    Trả về True nếu thành công, False nếu thất bại.
    """
    # Tạo thư mục cha chứa file đích nếu thư mục đó chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Nạp dữ liệu âm thanh từ file .m4a
        audio = AudioSegment.from_file(input_path)
        # Xuất âm thanh sang định dạng FLAC
        audio.export(output_path, format="flac")
        print(f"[THÀNH CÔNG] {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        return True
    except Exception as e:
        # Bắt lỗi nếu file bị hỏng hoặc thiếu thư viện ffmpeg
        print(f"[LỖI] Không thể chuyển đổi {os.path.basename(input_path)}: {e}")
        return False


def batch_convert_folders(input_folder="input", output_folder="output", recursive=False):
    """
    Chuyển đổi hàng loạt (Batch Convert) tất cả các file .m4a trong thư mục nguồn sang .flac trong thư mục đích.
    - input_folder: Thư mục chứa các file .m4a cần chuyển đổi.
    - output_folder: Thư mục chứa các file .flac kết quả.
    - recursive: True: Quét đệ quy tất cả các thư mục con; False: Chỉ quét thư mục hiện tại.
    """
    if not os.path.exists(input_folder):
        print(f"Thư mục nguồn không tồn tại: {input_folder}")
        return

    success_count = 0  # Đếm số lượng file chuyển đổi thành công
    fail_count = 0     # Đếm số lượng file bị lỗi
    total_files = 0    # Tổng số file quét được

    print(f"Bắt đầu quét thư mục '{input_folder}'...")
    print(f"Lưu kết quả vào '{output_folder}'...\n")

    if recursive:
        # Chế độ quét đệ quy tất cả các thư mục con bên trong
        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                if filename.lower().endswith(".m4a"):
                    total_files += 1
                    input_file = os.path.join(root, filename)

                    # Tính toán đường dẫn tương đối để giữ nguyên cấu trúc cây thư mục ở đầu ra
                    rel_path = os.path.relpath(root, input_folder)
                    target_dir = os.path.join(output_folder, rel_path) if rel_path != "." else output_folder

                    # Tạo tên file mới với phần mở rộng .flac
                    output_filename = os.path.splitext(filename)[0] + ".flac"
                    output_file = os.path.join(target_dir, output_filename)

                    if convert_m4a_to_flac(input_file, output_file):
                        success_count += 1
                    else:
                        fail_count += 1
    else:
        # Chế độ quét phẳng trong 1 thư mục duy nhất
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".m4a"):
                total_files += 1
                input_file = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + ".flac"
                output_file = os.path.join(output_folder, output_filename)

                if convert_m4a_to_flac(input_file, output_file):
                    success_count += 1
                else:
                    fail_count += 1

    # In bảng thống kê kết quả sau khi hoàn thành
    print("\n" + "=" * 40)
    print(f"Hoàn tất! Đã xử lý {total_files} file:")
    print(f"- Thành công: {success_count}")
    print(f"- Thất bại: {fail_count}")


if __name__ == "__main__":
    # Cấu hình đường dẫn thư mục nguồn và đích
    input_dir = "input"
    output_dir = "output"

    # True nếu muốn quét cả thư mục con bên trong
    scan_subfolders = False

    # Thực thi chuyển đổi hàng loạt
    batch_convert_folders(input_dir, output_dir, recursive=scan_subfolders)
