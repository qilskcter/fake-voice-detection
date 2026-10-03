import os
import streamlit as st

def load_css(css_file_path: str = "src/styles.css"):
    """Load and inject custom CSS stylesheet into Streamlit."""
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_sidebar_dashboard(device_name: str):
    """Render sidebar system info dashboard."""
    with st.sidebar:
        st.markdown("### System Dashboard")
        st.caption("Anti-Fraud Voice Authentication System")

        st.markdown("##### Hướng dẫn sử dụng:")
        st.markdown("""
        - **Thời lượng:** 2–4 giây âm thanh rõ ràng.
        - **Khoảng cách:** Đặt miệng cách micro 15–20cm.
        - **Môi trường:** Giảm bớt tiếng ồn quạt/máy lạnh xung quanh.
        """)

def render_hero_banner():
    """Render top application hero banner."""
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title"> Voice Biometrics Fraud Detection</div>
            <div class="hero-subtitle">
                Hệ thống AI phân loại sinh trắc học giọng nói nâng cao: Phát hiện và bảo vệ trước các cuộc tấn công ghi âm phát lại (<b>Replay Attack</b>).
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_result_banner(mean_score: float):
    """Render classification verdict card."""
    if mean_score >= 0.5:
        st.markdown("""
            <div class="result-card-real">
                <div class="result-badge-real">🟢 LIVE VOICE (Giọng Thật Trực Tiếp)</div>
                <div class="result-desc">Tín hiệu được xác nhận là giọng người thật nói trực tiếp. Độ tin cậy cao.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card-fake">
                <div class="result-badge-fake">🔴 REPLAY ATTACK (Phát Lại Qua Loa)</div>
                <div class="result-desc">Cảnh báo: Tín hiệu bị nghi ngờ là bản thu âm phát lại qua loa/thiết bị điện tử.</div>
            </div>
        """, unsafe_allow_html=True)

def render_metric_card(value: str, label: str, value_color: str = None):
    """Render individual metric summary card."""
    color_style = f' style="color: {value_color};"' if value_color else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value"{color_style}>{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)
