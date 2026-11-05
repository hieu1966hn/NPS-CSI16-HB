import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv

# Tải cấu hình từ file .env
load_dotenv()

# CẤU HÌNH GIAO DIỆN STREAMLIT
st.set_page_config(
    page_title="Shutter AI - Camera Price Predictor & Assistant",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện nhiếp ảnh sang trọng (Dark Theme)
st.markdown("""
    <style>
    .main-title {
        font-size: 40px !important;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px !important;
        text-align: center;
        color: #888888;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 22px !important;
        font-weight: 700;
        color: #FFFFFF;
        border-left: 5px solid #FF4B4B;
        padding-left: 10px;
        margin-bottom: 20px;
    }
    .card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333333;
        margin-bottom: 15px;
        color: #DDDDDD;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2ECC71;
    }
    </style>
""", unsafe_allow_html=True)

# TẢI MÔ HÌNH MACHINE LEARNING (CỦA QUANG MINH - ĐÃ FIX TARGET LEAKAGE)
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('camera_price_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoder = joblib.load('sensor_encoder.joblib')
        with open('feature_names.txt', 'r') as f:
            features = f.read().splitlines()
        return model, scaler, encoder, features
    except Exception as e:
        st.error(f"Lỗi tải mô hình ML: {e}")
        return None, None, None, None

model, scaler, encoder, features = load_ml_components()

# CẤU HÌNH GEMINI AI (CHO TAB TƯ VẤN)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    ai_model = None

# TẢI DỮ LIỆU ĐỂ EDA VÀ LỌC
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('camera_cleaned.csv')
        # Lọc bớt cột không cần thiết cho nhanh
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file camera_cleaned.csv: {e}")
        # Dữ liệu mẫu nếu bị lỗi
        return pd.DataFrame()

df = load_data()

# MENU ĐIỀU HƯỚNG SIDEBAR
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h2>📷 SHUTTER AI</h2></div>", unsafe_allow_html=True)
    menu = st.radio(
        "Menu chức năng:",
        ["🏡 Tổng Quan Dự Án", "🚀 Dự Đoán Giá Máy Ảnh", "📊 Phân Tích Dữ Liệu (EDA)", "🤖 Trợ Lý Tư Vấn Nhiếp Ảnh"],
        index=1 # Mặc định mở trang Predict
    )
    st.write("---")
    st.caption("⚙️ Trạng thái mô hình Machine Learning:")
    if model is not None:
        st.success("🟢 Đã nạp camera_price_model (Đã tối ưu)")
    else:
        st.error("🔴 Chưa kết nối mô hình")

# 🏡 TAB 1: TỔNG QUAN DỰ ÁN (HOME)
if menu == "🏡 Tổng Quan Dự Án":
    st.markdown("<div class='main-title'>📷 Shutter AI: Camera Price Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Hệ thống ứng dụng trí tuệ nhân tạo hỗ trợ dự đoán định giá và tư vấn mua sắm máy ảnh kỹ thuật số</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>🎯 Mục Tiêu Dự Án</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Dự án của bạn <b>Quang Minh</b> được phát triển nhằm giải quyết bài toán định giá máy ảnh dựa trên các thông số kỹ thuật thực tế 
    (Độ phân giải, Trọng lượng, Tốc độ FPS video, Thời lượng pin, Cảm biến...). Hệ thống giúp người dùng mới bắt đầu hoặc các cửa hàng 
    định giá chính xác máy ảnh cũ/mới dựa trên dữ liệu lớn thu thập từ thị trường Amazon và các trang review nhiếp ảnh chuyên nghiệp.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>📊 Thông Số Tập Dữ Liệu</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tổng Số Máy Ảnh", f"{len(df)} máy", delta="Từ 12+ hãng")
    with c2:
        st.metric("Hãng Phổ Biến", "Sony / Fujifilm", delta="Chiếm số lượng lớn")
    with c3:
        st.metric("Độ phân giải tối đa", f"{df['pixel_cleaned'].max():.1f} MP")
    with c4:
        st.metric("Giá máy cao nhất", f"{df['price_cleaned'].max():,.0f} Rp")

# 🚀 TAB 2: DỰ ĐOÁN GIÁ MÁY ẢNH (PREDICT)
elif menu == "🚀 Dự Đoán Giá Máy Ảnh":
    st.markdown("<div class='main-title'>🔮 Dự Đoán Giá Máy Ảnh Bằng Trí Tuệ Nhân Tạo</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Lỗi: Mô hình chưa được nạp thành công. Hãy kiểm tra các file *.joblib trong thư mục dự án.")
    else:
        col_in, col_out = st.columns([1.3, 1], gap="large")
        
        with col_in:
            st.markdown("<div class='section-header'>📋 Nhập Thông Số Kỹ Thuật Máy Ảnh</div>", unsafe_allow_html=True)
            
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                pixel_input = st.slider("Độ phân giải (Megapixels - MP):", min_value=5.0, max_value=60.0, value=24.0, step=0.1)
                weight_input = st.number_input("Trọng lượng máy bao gồm pin (Gam - g):", min_value=100, max_value=2000, value=450)
                max_fps_input = st.slider("Tốc độ khung hình video tối đa (Max FPS):", min_value=24, max_value=240, value=60)
            
            with sub_c2:
                battery_input = st.number_input("Thời lượng pin dự kiến (Số lần chụp - shots):", min_value=100, max_value=2000, value=350)
                sensor_list = list(encoder.classes_) if encoder is not None else ['APS-C', '1"', 'Four Thirds', 'Full Frame', '1/2.3"']
                sensor_input = st.selectbox("Kích thước cảm biến (Sensor Size):", options=sensor_list, index=0)
            
            # Tính toán weight_per_mp
            weight_per_mp_val = weight_input / pixel_input
            
        with col_out:
            st.markdown("<div class='section-header'>🔮 Kết Quả Dự Báo Từ Mô Hình</div>", unsafe_allow_html=True)
            
            if st.button("🔥 BẮT ĐẦU DỰ ĐOÁN GIÁ", type="primary", use_container_width=True):
                # Mã hóa cảm biến
                sensor_encoded = encoder.transform([sensor_input])[0]
                
                # Tạo dataframe đầu vào đúng định dạng features
                input_df = pd.DataFrame([[
                    pixel_input, weight_input, max_fps_input, battery_input, sensor_encoded, weight_per_mp_val
                ]], columns=['pixel_cleaned', 'weight', 'max_video_fps', 'battery_life', 'sensor_size_encoded', 'weight_per_mp'])
                
                # Chuẩn hóa đầu vào
                scaled_input = scaler.transform(input_df)
                
                # Dự đoán giá
                predicted_price_rp = model.predict(scaled_input)[0]
                # Đảm bảo không bị âm
                if predicted_price_rp < 0:
                    predicted_price_rp = 0.0
                
                # Đổi sang VND (1 Rp tương đương khoảng 1.6 VND)
                predicted_price_vnd = predicted_price_rp * 1.6
                
                st.success("### 🎉 Định giá máy ảnh hoàn tất!")
                
                st.markdown(f"""
                <div class='card'>
                    <p style='margin-bottom:5px; font-size:14px; color:#888888;'>GIÁ DỰ ĐOÁN (RUPIAH INDONESIA):</p>
                    <p class='metric-value'>{predicted_price_rp:,.0f} Rp</p>
                    <p style='margin-bottom:5px; font-size:14px; color:#888888; margin-top:15px;'>GIÁ QUY ĐỔI SANG VND (TẠM TÍNH):</p>
                    <p class='metric-value' style='color:#3498DB;'>{predicted_price_vnd:,.0f} VNĐ</p>
                    <p style='font-size:12px; color:#ff9f43; margin-top:10px;'>* Công thức quy đổi: 1 Rp ≈ 1.6 VNĐ (tỷ giá tham khảo).</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gợi ý phân khúc máy
                if predicted_price_rp < 8000000:
                    st.info("💡 **Phân khúc:** Máy ảnh giá rẻ (Entry-level) - Thích hợp cho người mới bắt đầu học nhiếp ảnh.")
                elif predicted_price_rp < 16000000:
                    st.info("💡 **Phân khúc:** Máy ảnh tầm trung (Mid-range) - Phù hợp cho chụp ảnh dịch vụ, bán chuyên nghiệp.")
                else:
                    st.info("💡 **Phân khúc:** Máy ảnh cao cấp (High-end/Professional) - Trang bị công nghệ tiên tiến, dành cho nhiếp ảnh gia chuyên nghiệp.")

# 📊 TAB 3: PHÂN TÍCH DỮ LIỆU & KHÁM PHÁ (EDA)
elif menu == "📊 Phân Tích Dữ Liệu (EDA)":
    st.markdown("<div class='main-title'>📊 Trực Quan Hóa & Phân Tích Dữ Liệu Lớn</div>", unsafe_allow_html=True)
    
    if df.empty:
        st.warning("Không tìm thấy dữ liệu camera_cleaned.csv để phân tích.")
    else:
        tab_dist, tab_relation, tab_brand = st.tabs(["📈 Phân Bố Giá Cả", "🧐 Tương Quan Thuộc Tính", "🏢 Phân Tích Hãng Sản Xuất"])
        
        with tab_dist:
            st.markdown("<div class='section-header'>Khảo sát phân bố giá bán máy ảnh trong tập dữ liệu</div>", unsafe_allow_html=True)
            # Biến đổi giá sang triệu Rp cho dễ đọc
            df['price_million_rp'] = df['price_cleaned'] / 1e6
            fig1 = px.histogram(df, x="price_million_rp", nbins=20, labels={"price_million_rp": "Giá bán (Triệu Rp)"},
                               title="Biểu đồ phân phối giá máy ảnh (Triệu Rp)", color_discrete_sequence=["#FF4B4B"])
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("""
            **Nhận xét:** Đa số các dòng máy ảnh tập trung ở phân khúc giá từ **5 triệu đến 18 triệu Rp**. Phân khúc máy siêu cao cấp (>22 triệu Rp) có số lượng ít hơn hẳn.
            """)
            
        with tab_relation:
            st.markdown("<div class='section-header'>Mối liên hệ giữa Độ phân giải (Megapixel) và Giá bán</div>", unsafe_allow_html=True)
            fig2 = px.scatter(df, x="pixel_cleaned", y="price_cleaned", color="sensor_size",
                             size="weight", hover_name="camera_name",
                             labels={"pixel_cleaned": "Megapixel", "price_cleaned": "Giá bán (Rp)"},
                             title="Tương quan giữa Megapixel, Kích thước cảm biến và Giá bán (Kích thước bóng đại diện cho Trọng lượng)")
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab_brand:
            st.markdown("<div class='section-header'>Giá bán trung bình theo từng Hãng sản xuất</div>", unsafe_allow_html=True)
            avg_price = df.groupby('brand')['price_cleaned'].mean().reset_index().sort_values(by='price_cleaned', ascending=False)
            fig3 = px.bar(avg_price, x='brand', y='price_cleaned', color='brand',
                          labels={'brand': 'Hãng sản xuất', 'price_cleaned': 'Giá trung bình (Rp)'},
                          title='Bảng xếp hạng giá máy ảnh trung bình theo hãng')
            st.plotly_chart(fig3, use_container_width=True)

# 🤖 TAB 4: TRỢ LÝ TƯ VẤN NHIẾP ẢNH AI (CHATBOT)
elif menu == "🤖 Trợ Lý Tư Vấn Nhiếp Ảnh":
    st.markdown("<div class='main-title'>🤖 Trợ Lý Tư Vấn Nhiếp Ảnh AI Thông Thái</div>", unsafe_allow_html=True)
    
    if not GEMINI_API_KEY or ai_model is None:
        st.warning("⚠️ **Thiếu API Key:** Vui lòng thiết lập khóa GEMINI_API_KEY trong file `.env` hoặc hệ thống secrets để kích hoạt Chatbot tư vấn.")
    else:
        st.info("Trợ lý AI được tích hợp Gemini-2.5-flash để hỗ trợ tư vấn chọn mua máy ảnh, so sánh thông số kịch bản chụp ảnh cho bạn.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Mình là chuyên gia tư vấn nhiếp ảnh AI. Bạn cần hỗ trợ chọn máy ảnh, lens hay tư vấn kỹ thuật chụp ảnh nào không?"}
            ]
            
        # Hiển thị lịch sử chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_input = st.chat_input("Nhập câu hỏi nhiếp ảnh tại đây...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.spinner("Trợ lý AI đang phân tích yêu cầu..."):
                try:
                    # Truyền context dữ liệu máy ảnh vào prompt
                    context = (
                        f"Bạn là một chuyên gia tư vấn máy ảnh và nhiếp ảnh cao cấp. "
                        f"Đây là dữ liệu sơ bộ một số dòng máy ảnh có sẵn của cửa hàng để bạn tham khảo tư vấn: {df[['camera_name', 'brand', 'price_cleaned', 'pixel_cleaned', 'sensor_size']].head(10).to_string()}.\n"
                        f"Hãy trả lời câu hỏi bằng tiếng Việt chuyên nghiệp, hữu ích và nhiệt tình.\n\n"
                        f"Câu hỏi của khách hàng: {user_input}"
                    )
                    response = ai_model.generate_content(context)
                    ai_reply = response.text
                except Exception as e:
                    ai_reply = f"❌ Lỗi kết nối API Gemini: {e}"
                    
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            with st.chat_message("assistant"):
                st.write(ai_reply)
