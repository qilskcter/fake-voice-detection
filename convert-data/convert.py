import os
from pydub import AudioSegment


def convert_m4a_to_flac(input_path, output_path):
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  try:
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="flac")
    print(
        f"[THÀNH CÔNG] {os.path.basename(input_path)} ->"
        f" {os.path.basename(output_path)}"
    )
    return True
  except Exception as e:
    print(f"[LỖI] Không thể chuyển đổi {os.path.basename(input_path)}: {e}")
    return False


def batch_convert_folders(
    input_folder="input", output_folder="output", recursive=False
):
  if not os.path.exists(input_folder):
    print(f"Thư mục nguồn không tồn tại: {input_folder}")
    return

  success_count = 0
  fail_count = 0
  total_files = 0

  print(f"Bắt đầu quét thư mục '{input_folder}'...")
  print(f"Lưu kết quả vào '{output_folder}'...\n")

  if recursive:
    for root, dirs, files in os.walk(input_folder):
      for filename in files:
        if filename.lower().endswith(".m4a"):
          total_files += 1
          input_file = os.path.join(root, filename)

          rel_path = os.path.relpath(root, input_folder)
          target_dir = (
              os.path.join(output_folder, rel_path)
              if rel_path != "."
              else output_folder
          )

          output_filename = os.path.splitext(filename)[0] + ".flac"
          output_file = os.path.join(target_dir, output_filename)

          if convert_m4a_to_flac(input_file, output_file):
            success_count += 1
          else:
            fail_count += 1
  else:
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

  print("\n" + "=" * 40)
  print(f"Hoàn tất! Đã xử lý {total_files} file:")
  print(f"- Thành công: {success_count}")
  print(f"- Thất bại: {fail_count}")


if __name__ == "__main__":
  input_dir = "input"
  output_dir = "output"

  scan_subfolders = False

  batch_convert_folders(input_dir, output_dir, recursive=scan_subfolders)