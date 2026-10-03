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
from ui_components import (
    load_css,
    render_hero_banner,
    render_metric_card,
    render_result_banner,
    render_sidebar_dashboard,
)

# Page Configuration
st.set_page_config(
    page_title="Voice Shield AI - Live Voice vs Replay Attack",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External CSS Stylesheet
load_css("src/styles.css")

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "NVIDIA GPU (CUDA)"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

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

# ImageNet normalization stats
imagenet_mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(device)
imagenet_std = torch.tensor([0.229]).view(1, 1, 1, 1).to(device)

def process_audio_file(file_path):
    """Preprocess audio using exact AudioUtil pipeline from training.py."""
    aud = AudioUtil.open(file_path)
    aud = AudioUtil.resample(aud, 16000)
    aud = AudioUtil.rechannel(aud, 1)
    
    sig, sr = aud
    max_len = 40000  # 2.5 seconds at 16kHz
    
    segments = []
    if sig.shape[1] <= max_len:
        padded_aud = AudioUtil.pad_trunc(aud, 2500)
        segments.append(padded_aud)
    else:
        step = max_len // 2
        for start in range(0, sig.shape[1] - max_len + 1, step):
            chunk = sig[:, start:start + max_len]
            segments.append((chunk, sr))
        if (sig.shape[1] - max_len) % step != 0:
            chunk = sig[:, -max_len:]
            segments.append((chunk, sr))

    specs = []
    for seg in segments:
        sgram = AudioUtil.spectro_gram(seg, n_mels=128, n_fft=780, hop_len=195)
        spec_mean = sgram.mean()
        spec_std = sgram.std() + 1e-6
        sgram = (sgram - spec_mean) / spec_std
        specs.append(sgram)

    total_duration_sec = sig.shape[1] / sr
    return specs, total_duration_sec

# --- SIDEBAR DASHBOARD ---
render_sidebar_dashboard(device_name)

# --- HERO BANNER ---
render_hero_banner()

# Layout: 2 columns (Left: Inputs & Controls, Right: Live Audio Player)
col_left, col_right = st.columns([1, 1], gap="medium")

audio_source = None
file_name = "audio.wav"

with col_left:
    st.markdown("#### Chọn Phương Thức Thu Âm")
    tab1, tab2 = st.tabs(["Ghi Âm Trực Tiếp", "Tải File Âm Thanh"])
    
    with tab1:
        st.write("Nhấn vào biểu tượng Micro bên dưới để ghi âm câu thoại:")
        recorded_audio = st.audio_input("Bắt đầu ghi âm")
        if recorded_audio is not None:
            audio_source = recorded_audio
            file_name = getattr(recorded_audio, "name", "recorded_audio.wav")

    with tab2:
        st.write("Tải lên file âm thanh từ máy tính:")
        uploaded_file = st.file_uploader(
            "Định dạng hỗ trợ: .wav, .flac, .mp3, .m4a, .aac, .ogg, .webm",
            type=["wav", "flac", "mp3", "m4a", "ogg", "aac", "webm"]
        )
        if uploaded_file is not None:
            audio_source = uploaded_file
            file_name = uploaded_file.name

# Extract raw audio bytes ONCE safely
audio_bytes = None
if audio_source is not None:
    if hasattr(audio_source, "getvalue"):
        audio_bytes = audio_source.getvalue()
    else:
        audio_bytes = audio_source.read()
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)

# Infer proper MIME type & extension
raw_mime = getattr(audio_source, "type", None) if audio_source is not None else None
file_ext = "." + file_name.split(".")[-1].lower() if "." in file_name else ".wav"

if raw_mime:
    if "webm" in raw_mime.lower():
        file_ext = ".webm"
    elif "ogg" in raw_mime.lower():
        file_ext = ".ogg"
    elif "mp3" in raw_mime.lower():
        file_ext = ".mp3"
    elif "wav" in raw_mime.lower():
        file_ext = ".wav"

# Process audio file for inference
specs, total_duration = None, 0.0
if audio_bytes is not None and len(audio_bytes) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        specs, total_duration = process_audio_file(tmp_file_path)
    except Exception as e:
        st.error(f"Lỗi khi đọc định dạng âm thanh: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError:
                pass

with col_right:
    st.markdown("#### Trình Phát Âm Thanh")
    if audio_bytes is None or len(audio_bytes) == 0:
        st.info("Hãy chọn Ghi âm trực tiếp hoặc Tải file lên ở cột bên trái để bắt đầu phân tích.")
    else:
        mime_type_map = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".webm": "audio/webm",
        }
        audio_format = raw_mime if raw_mime else mime_type_map.get(file_ext, "audio/wav")
        
        st.audio(audio_bytes, format=audio_format)
        st.caption(f"Tệp âm thanh: `{file_name}` ({len(audio_bytes) / 1024:.1f} KB)")

# --- INFERENCE & RESULTS SECTION ---
if specs is not None and len(specs) > 0:
    st.markdown("---")
    st.markdown("### Kết Quả Phân Tích Giọng Nói")
    
    try:
        scores = []
        for spec in specs:
            input_tensor = spec.unsqueeze(0).to(device)
            input_tensor = (input_tensor * imagenet_std) + imagenet_mean
            input_tensor = (input_tensor - imagenet_mean) / imagenet_std
            
            with torch.no_grad():
                out = model(input_tensor)
                score = torch.sigmoid(out).item()
                scores.append(score)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        fake_score = 1.0 - mean_score
        
        # Result Banner Render
        render_result_banner(mean_score)

        # 3 Detailed Metrics Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            render_metric_card(f"{mean_score:.2%}", "Xác Suất Live Voice (Real)", value_color="#10b981")
            
        with m_col2:
            render_metric_card(f"{fake_score:.2%}", "Xác Suất Replay Attack (Fake)", value_color="#ef4444")
            
        with m_col3:
            render_metric_card(f"{total_duration:.1f}s", f"Thời Lượng ({len(specs)} Phân Đoạn)")

        # # Live vs Fake Gauge Bar
        # st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown("##### 🎯 Biểu Đồ Ngưỡng Ra Quyết Định (Decision Threshold: 50.0%)")
        # st.progress(min(max(mean_score, 0.0), 1.0))
        
        # # Expander for Segment Details
        # with st.expander("🔍 Xem chi tiết điểm số từng phân đoạn (Segment Analysis)"):
        #     for idx, sc in enumerate(scores):
        #         st.write(f"- Phân đoạn {idx + 1} (2.5s): Live Score = **{sc:.2%}** | Verdict = {'🟢 Real' if sc >= 0.5 else '🔴 Fake'}")

    except Exception as e:
        st.error(f"Lỗi hệ thống khi dự đoán: {str(e)}")
