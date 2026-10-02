import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import tempfile
from models import CRNNWithAttn

# Create the model and put it on GPU / MPS if available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

myModel = CRNNWithAttn().to(device)

# Load your model
@st.cache_resource
def load_model():
    myModel.load_state_dict(torch.load('./models/best_model10.pth', map_location=device))
    myModel.eval()
    return myModel

model = load_model()

# ImageNet normalization stats (for 1 channel input)
imagenet_mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(device)
imagenet_std = torch.tensor([0.229]).view(1, 1, 1, 1).to(device)

# Audio preprocessing (Sliding Window for long audio)
def preprocess(file_path):
    from training import AudioUtil
    
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
        step = max_len // 2  # 50% overlap
        for start in range(0, sig.shape[1] - max_len + 1, step):
            chunk = sig[:, start:start + max_len]
            segments.append((chunk, sr))
        
        # Handle the last remaining part if any
        if (sig.shape[1] - max_len) % step != 0:
            chunk = sig[:, -max_len:]
            segments.append((chunk, sr))

    specs = []
    for seg in segments:
        sgram = AudioUtil.spectro_gram(seg, n_mels=128, n_fft=780, hop_len=195)
        # Per-instance standardization (exactly matching training.py)
        spec_mean = sgram.mean()
        spec_std = sgram.std() + 1e-6
        sgram = (sgram - spec_mean) / spec_std
        specs.append(sgram)
        
    return specs

# Inject custom CSS styles
st.markdown("""
    <style>
        /* Global background and font */
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }

        .reportview-container {
            background-color: #f9f9f9;
        }

        /* Center elements */
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        /* Styled prediction card */
        .prediction-box {
            border-radius: 12px;
            padding: 24px;
            margin-top: 20px;
            background-color: #ffffff;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            width: 85%;
            text-align: center;
        }

        .prediction-label {
            font-size: 24px;
            font-weight: bold;
        }

        .confidence {
            font-size: 18px;
            color: #555;
            margin-top: 10px;
        }

        .label-real {
            color: #2e7d32;
        }

        .label-fake {
            color: #c62828;
        }

        .sub-text {
            font-size: 14px;
            color: #666;
            margin-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='centered'>", unsafe_allow_html=True)
st.title("🔊 Audio Classifier: Live Voice vs Replay Attack")
st.markdown(
    "<p style='margin-bottom:20px;'>Hệ thống phân loại âm thanh: Giọng người thật nói trực tiếp (<b>Live Voice</b>) vs Giọng ghi âm phát lại qua loa (<b>Replay Attack</b>)</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Tải lên file âm thanh (.wav, .flac, .mp3, .m4a, .aac)", type=["wav", "flac", "mp3", "m4a", "ogg", "aac"])
st.markdown("<p style='text-align: center; margin: 10px 0;'><b>HOẶC</b></p>", unsafe_allow_html=True)
recorded_file = st.audio_input("Thu âm trực tiếp từ Micro (Live Voice)")

target_file = uploaded_file if uploaded_file else recorded_file

if target_file is not None:
    file_ext = "." + target_file.name.split(".")[-1].lower() if "." in target_file.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(target_file.read())
        tmp_file_path = tmp_file.name
    predicted_classes = []
    predicted = []
    input_tensors = preprocess(tmp_file_path)
    for input_tensor in input_tensors:
        input_tensor = input_tensor.unsqueeze(0).to(device)
        # Note: Do not apply imagenet normalization because training.py had a bug that effectively skipped it
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_classes.append(torch.sigmoid(outputs))
            predicted.append((torch.sigmoid(outputs) >= 0.5).int())

    mean_score = torch.mean(torch.cat(predicted_classes)).item()
    if mean_score >= 0.5:
        label = "Live Voice (Giọng nói trực tiếp)"
        label_class = "label-real"
        detail_msg = "Tín hiệu âm thanh được xác định là giọng người thật phát biểu trực tiếp."
    else:
        label = "Replay Attack (Phát lại qua loa)"
        label_class = "label-fake"
        detail_msg = "Tín hiệu âm thanh bị nghi ngờ là bản ghi âm được phát lại qua thiết bị loa."

    # Map extension to standard HTML5 audio MIME format for Streamlit
    mime_type_map = {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
    }
    audio_format = mime_type_map.get(file_ext, "audio/wav")

    # Display audio player and results
    st.audio(target_file, format=audio_format)
    st.markdown(f"""
        <div class='prediction-box'>
            <div class='prediction-label {label_class}'>Kết quả phân loại: {label}</div>
            <div class='sub-text'>{detail_msg}</div>
            <div class='confidence'>Độ tin cậy (Live Voice Score): <code>{mean_score:.2%}</code></div>
        </div>
    """, unsafe_allow_html=True)

