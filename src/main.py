import streamlit as st
import torchaudio
import torchaudio.transforms as T
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

# Audio preprocessing
def preprocess(waveform, sample_rate):
    # Resample if needed
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
    for waveform in all_waveforms:
        mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=780,
            hop_length=195,
            n_mels=128
        )(waveform)
        mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

        # Per-instance standardization (matching training.py)
        spec_mean = mel_spec.mean()
        spec_std = mel_spec.std() + 1e-6
        mel_spec = (mel_spec - spec_mean) / spec_std

        all_spec.append(mel_spec)
    return all_spec

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

if uploaded_file is not None:
    file_ext = "." + uploaded_file.name.split(".")[-1].lower() if "." in uploaded_file.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    predicted_classes = []
    predicted = []
    try:
        from training import AudioUtil
        waveform, sample_rate = AudioUtil.open(tmp_file_path)
    except Exception:
        try:
            waveform, sample_rate = torchaudio.load(tmp_file_path)
        except Exception:
            waveform, sample_rate = torchaudio.load(tmp_file_path, backend="soundfile")

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
    st.audio(uploaded_file, format=audio_format)
    st.markdown(f"""
        <div class='prediction-box'>
            <div class='prediction-label {label_class}'>Kết quả phân loại: {label}</div>
            <div class='sub-text'>{detail_msg}</div>
            <div class='confidence'>Độ tin cậy (Live Voice Score): <code>{mean_score:.2%}</code></div>
        </div>
    """, unsafe_allow_html=True)

