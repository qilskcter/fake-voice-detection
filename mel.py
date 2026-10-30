import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Đọc file âm thanh (nên chuẩn hóa sample rate, ví dụ 16000Hz)
audio_path = r'D:\xltn\phu-khac-anh-dep-trai\data\real\AN_MAN_DAI_KHAI-KHANH.flac'
y, sr = librosa.load(audio_path, sr=16000)

# 2. Chuyển sang Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_fft=2048, 
    hop_length=512, 
    n_mels=128
)

# 3. Chuyển sang thang đo Decibel (Log scale) cho dễ nhìn và tối ưu cho Deep Learning
mel_spectros_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 4. Vẽ và hiển thị biểu đồ
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel_spectros_db, 
    sr=sr, 
    hop_length=512, 
    x_axis='time', 
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()