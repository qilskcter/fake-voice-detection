import os
import tempfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from models import CRNNWithAttn
from training import AudioUtil

# Page Configuration
st.set_page_config(
    page_title="Audio Classifier - Live Voice vs Replay Attack",
    page_icon="🎙️",
    layout="centered"
)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

myModel = CRNNWithAttn().to(device)

# Load pretrained model
@st.cache_resource
def load_model():
    model_path = './models/best_model10.pth'
    if os.path.exists(model_path):
        myModel.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    myModel.eval()
    return myModel

model = load_model()

# ImageNet normalization stats (for 1 channel input)
imagenet_mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(device)
imagenet_std = torch.tensor([0.229]).view(1, 1, 1, 1).to(device)

# Audio preprocessing
def preprocess(waveform, sample_rate):
    # Resample to 16kHz
    if sample_rate != 16000:
        resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    # Mono 1-channel
    waveform = waveform.mean(dim=0, keepdim=True)

    # Trim / pad to 2.5 seconds (40,000 samples)
    max_len = 40000
    all_waveforms = []
    for _ in range(waveform.shape[1] // max_len + 1):
        if waveform.shape[1] > max_len:
            all_waveforms.append(waveform[:, :max_len])
            waveform = waveform[:, max_len:]
        elif waveform.shape[1] < max_len:
            pad_len = max_len - waveform.shape[1]
            all_waveforms.append(torch.nn.functional.pad(waveform, (0, pad_len)))

    # Convert to MelSpectrogram
    all_spec = []
    for wf in all_waveforms:
        mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=780,
            hop_length=195,
            n_mels=128
        )(wf)
        mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

        # Per-instance standardization (matching training.py)
        spec_mean = mel_spec.mean()
        spec_std = mel_spec.std() + 1e-6
        mel_spec = (mel_spec - spec_mean) / spec_std

        all_spec.append(mel_spec)
    return all_spec

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .main-header {
            text-align: center;
            margin-bottom: 25px;
        }

        .prediction-box {
            border-radius: 16px;
            padding: 24px;
            margin-top: 24px;
            background-color: #ffffff;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
            border: 1px solid #eef0f2;
            text-align: center;
        }

        .prediction-label {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .confidence {
            font-size: 18px;
            color: #333333;
            margin-top: 14px;
            font-weight: 500;
        }

        .label-real {
            color: #2e7d32;
        }

        .label-fake {
            color: #c62828;
        }

        .sub-text {
            font-size: 15px;
            color: #666666;
            margin-top: 6px;
        }

        .stAudioInput {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🔊 Audio Classifier: Live Voice vs Replay Attack")
st.markdown(
    "<p style='font-size:16px; color:#555;'>Hệ thống phân loại giọng nói: Giọng người thật đang nói trực tiếp (<b>Live Voice</b>) vs Giọng ghi âm phát lại qua loa (<b>Replay Attack</b>)</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Select Input Method: Tabs
tab1, tab2 = st.tabs(["🎙️ Ghi âm trực tiếp", "📁 Tải lên file âm thanh"])

audio_source = None
file_name = "audio.wav"

with tab1:
    st.markdown("##### 🎙️ Nhấn vào nút bên dưới để ghi âm trực tiếp qua micro:")
    recorded_audio = st.audio_input("Ghi âm giọng nói của bạn")
    if recorded_audio is not None:
        audio_source = recorded_audio
        file_name = getattr(recorded_audio, "name", "recorded_audio.wav")

with tab2:
    st.markdown("##### 📁 Hoặc chọn file âm thanh từ thiết bị của bạn:")
    uploaded_file = st.file_uploader(
        "Tải lên file âm thanh (.wav, .flac, .mp3, .m4a, .aac, .ogg, .webm)",
        type=["wav", "flac", "mp3", "m4a", "ogg", "aac", "webm"]
    )
    if uploaded_file is not None:
        audio_source = uploaded_file
        file_name = uploaded_file.name

if audio_source is not None:
    file_ext = "." + file_name.split(".")[-1].lower() if "." in file_name else ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_source.read())
        tmp_file_path = tmp_file.name

    try:
        predicted_classes = []
        predicted = []
        
        try:
            waveform, sample_rate = AudioUtil.open(tmp_file_path)
        except Exception:
            waveform, sample_rate = torchaudio.load(tmp_file_path)

        input_tensors = preprocess(waveform, sample_rate)
        for input_tensor in input_tensors:
            input_tensor = input_tensor.unsqueeze(0).to(device)
            input_tensor = (input_tensor - imagenet_mean) / imagenet_std
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_classes.append(torch.sigmoid(outputs))
                predicted.append((torch.sigmoid(outputs) >= 0.5).int())

        mean_score = torch.mean(torch.cat(predicted_classes)).item()
        
        if mean_score >= 0.5:
            label = "Live Voice (Giọng nói trực tiếp)"
            label_class = "label-real"
            detail_msg = "✅ Tín hiệu âm thanh được xác định là giọng người thật đang phát biểu trực tiếp."
            badge_icon = "🟢"
        else:
            label = "Replay Attack (Phát lại qua loa)"
            label_class = "label-fake"
            detail_msg = "⚠️ Tín hiệu âm thanh bị nghi ngờ là bản ghi âm được phát lại qua thiết bị loa."
            badge_icon = "🔴"

        # Map extension to standard HTML5 audio MIME format
        mime_type_map = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".webm": "audio/webm",
        }
        audio_format = mime_type_map.get(file_ext, "audio/wav")

        st.markdown("---")
        st.subheader("🎧 Phát lại âm thanh vừa nhận:")
        st.audio(audio_source, format=audio_format)

        st.markdown(f"""
            <div class='prediction-box'>
                <div class='prediction-label {label_class}'>{badge_icon} Kết quả: {label}</div>
                <div class='sub-text'>{detail_msg}</div>
                <div class='confidence'>Độ tin cậy (Live Voice Score): <code>{mean_score:.2%}</code></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display progress indicator bar for score visual
        st.progress(min(max(mean_score, 0.0), 1.0))

    except Exception as e:
        st.error(f"Lỗi khi xử lý file âm thanh: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError:
                pass
