"""
Module ui_components.py: Chứa các hàm render giao diện người dùng (UI Components) cho ứng dụng Streamlit.

Bao gồm:
1. load_css: Tải và nhúng mã định dạng CSS tùy biến vào trang web.
2. render_sidebar_dashboard: Hiển thị bảng điều khiển thông tin phần cứng và hướng dẫn ở thanh bên (Sidebar).
3. render_hero_banner: Hiển thị biểu ngữ tiêu đề giới thiệu hệ thống ở đầu trang.
4. render_result_banner: Hiển thị thẻ kết quả phân loại nổi bật (Màu xanh: Giọng thật, Màu đỏ: Cảnh báo phát lại).
5. render_metric_card: Hiển thị các thẻ chỉ số % xác suất dạng số liệu trực quan (KPI Card).
"""

import os            # Tương tác với hệ thống tệp tin (kiểm tra file styles.css có tồn tại không)
import streamlit as st  # Thư viện xây dựng giao diện ứng dụng web nhanh trong Python


def load_css(css_file_path: str = "src/styles.css"):
    """
    Đọc file stylesheet CSS từ đường dẫn cục bộ và nhúng trực tiếp vào mã HTML của trang web Streamlit.
    Giúp tùy biến màu sắc, phông chữ Inter hiện đại, bo góc thẻ và tạo hiệu ứng đổ bóng.
    """
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            # Dùng st.markdown với unsafe_allow_html=True để cho phép nhúng mã thẻ <style>
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_dashboard(device_name: str):
    """
    Hiển thị thông tin hệ thống và hướng dẫn thu âm trên thanh bên trái (Sidebar) của Streamlit.
    - device_name: Tên thiết bị tính toán đang chạy mô hình (ví dụ: CPU, CUDA GPU, Apple Silicon MPS).
    """
    with st.sidebar:
        # Tiêu đề bảng điều khiển hệ thống
        st.markdown("### System Dashboard")
        st.caption("Anti-Fraud Voice Authentication System")

        # Hiển thị thông số phần cứng hiện tại
        st.markdown(f"**Phần cứng suy luận:** `{device_name}`")
        st.markdown("---")

        # Hướng dẫn chi tiết cho người dùng cách thu âm để đạt độ chính xác cao nhất
        st.markdown("##### Hướng dẫn sử dụng:")
        st.markdown("""
        - **Thời lượng khuyến nghị:** 2–4 giây âm thanh rõ ràng.
        - **Khoảng cách:** Đặt miệng cách micro khoảng 15–20cm.
        - **Môi trường:** Giảm bớt tiếng ồn quạt/máy lạnh hoặc tiếng tivi xung quanh.
        - **Thử nghiệm Replay:** Ghi âm một câu, sau đó mở lại qua loa điện thoại sát micro để kiểm tra cảnh báo.
        """)


def render_hero_banner():
    """
    Hiển thị biểu ngữ (Hero Banner) nổi bật ở phần đầu trang web.
    Bao gồm tên dự án, biểu tượng bảo vệ và mô tả ngắn gọn bài toán nhận diện sinh trắc học giọng nói.
    """
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">🛡️ Voice Biometrics Fraud Detection</div>
            <div class="hero-subtitle">
                Hệ thống AI phân loại sinh trắc học giọng nói nâng cao: Phát hiện và bảo vệ trước các cuộc tấn công ghi âm phát lại (<b>Replay Attack</b>).
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_result_banner(mean_score: float):
    """
    Hiển thị thẻ thông báo kết quả dự đoán sau khi phân tích âm thanh:
    - mean_score >= 0.5: Kết luận giọng người thật nói trực tiếp (LIVE VOICE) -> Thẻ màu xanh lá cây 🟢
    - mean_score < 0.5 : Cảnh báo giọng thu âm phát lại qua loa (REPLAY ATTACK) -> Thẻ màu đỏ cảnh báo 🔴
    """
    if mean_score >= 0.5:
        # Trường hợp xác thực thành công: Giọng thật
        st.markdown("""
            <div class="result-card-real">
                <div class="result-badge-real">🟢 LIVE VOICE (Giọng Thật Trực Tiếp)</div>
                <div class="result-desc">Tín hiệu được xác nhận là giọng người thật nói trực tiếp. Độ tin cậy cao.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Trường hợp phát hiện gian lận: Tấn công phát lại qua loa
        st.markdown("""
            <div class="result-card-fake">
                <div class="result-badge-fake">🔴 REPLAY ATTACK (Phát Lại Qua Loa)</div>
                <div class="result-desc">Cảnh báo: Tín hiệu bị nghi ngờ là bản thu âm phát lại qua loa/thiết bị điện tử.</div>
            </div>
        """, unsafe_allow_html=True)


def render_metric_card(value: str, label: str, value_color: str = None):
    """
    Hiển thị một thẻ chỉ số đo lường (KPI Metric Card) nhỏ gọn và đẹp mắt.
    - value: Giá trị hiển thị nổi bật dạng chữ lớn (ví dụ: '98.5%').
    - label: Nhãn chú thích nhỏ bên dưới (ví dụ: 'Xác suất Live Voice').
    - value_color: Mã màu tùy chỉnh cho số liệu (ví dụ: '#10b981' cho màu xanh, '#ef4444' cho màu đỏ).
    """
    # Nếu có màu tùy chỉnh thì áp dụng thuộc tính CSS style inline
    color_style = f' style="color: {value_color};"' if value_color else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value"{color_style}>{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)
