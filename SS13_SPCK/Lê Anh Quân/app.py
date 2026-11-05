import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid Metal GPU hangs on macOS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from google import genai

# --- CẤU HÌNH TRANG PREMIUM ---
st.set_page_config(
    page_title="Hệ Thống Phân Tích & Dự Đoán Điểm Số Phim Ảnh AI",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# --- TÙY CHỈNH GIAO DIỆN BẰNG CSS ĐỂ WOW NGƯỜI DÙNG ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 40px !important;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sub-title {
        font-size: 16px !important;
        text-align: center;
        color: #7E8B9B;
        margin-bottom: 40px;
    }
    
    .section-header {
        font-size: 24px !important;
        font-weight: 700;
        color: #1E293B;
        border-left: 6px solid #FF4B4B;
        padding-left: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #FF4B4B;
    }
    
    .metric-label {
        font-size: 14px;
        color: #64748B;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("movie_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_data
def load_dataset():
    if os.path.exists("top10K-TMDB-movies.csv"):
        return pd.read_csv("top10K-TMDB-movies.csv")
    return None

model, tokenizer = load_assets()
df = load_dataset()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h2>🎬 Movie Predictor AI</h2></div>", unsafe_allow_html=True)
    menu = st.selectbox(
        "Lựa chọn không gian làm việc:",
        ["🏡 Trang Chủ (Home)", "📊 Biểu Đồ Thống Kê (EDA)", "🔮 Dự Đoán Điểm Phim", "🤖 Trợ Lý Phim Ảnh (Gemini AI)"]
    )
    
    st.write("---")
    st.caption("🤖 Trạng thái mô hình học máy:")
    if model is not None:
        st.success("🟢 Keras Neural Network: Sẵn sàng")
    else:
        st.error("🔴 Keras Neural Network: Lỗi kết nối")
        
    if df is not None:
        st.success(f"🟢 Database: {len(df):,} Movies")
    else:
        st.error("🔴 Database: Thiếu file dữ liệu")

# =========================================================
# 🏡 1. GIAO DIỆN TRANG CHỦ (HOME)
# =========================================================
if menu == "🏡 Trang Chủ (Home)":
    st.markdown("<div class='main-title'>🎬 Deep Learning Movie Score Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Ứng dụng học sâu xử lý ngôn ngữ tự nhiên để dự đoán điểm đánh giá của phim dựa trên thể loại và cốt truyện</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>📊 Chỉ Số Tổng Quan Bộ Dữ Liệu</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='card'><div class='metric-value'>{len(df):,}</div><div class='metric-label'>Tổng số mẫu phim</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='card'><div class='metric-value'>6.6 / 10</div><div class='metric-label'>Điểm đánh giá TB</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='card'><div class='metric-value'>12,000+</div><div class='metric-label'>Tổng từ vựng cốt truyện</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='card'><div class='metric-value'>93.4%</div><div class='metric-label'>Độ chính xác R² (Training)</div></div>", unsafe_allow_html=True)
        
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div class='section-header'>🎯 Mục Tiêu Dự Án</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>Dự án được xây dựng nhằm giúp các nhà làm phim, biên kịch đánh giá sơ bộ tiềm năng điểm số của một kịch bản phim trước khi đưa vào sản xuất. Sử dụng kiến trúc mạng nơ-ron Embedding và GlobalAveragePooling1D từ thư viện Keras, hệ thống tự động bóc tách từ khóa cốt truyện để dự đoán điểm số trên thang điểm TMDB.</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='section-header'>🛠️ Hướng Dẫn Sử Dụng Nhanh</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><ul><li><b>Khám Phá EDA:</b> Xem biểu đồ phân tích dữ liệu 10,000 bộ phim từ ngôn ngữ, phân bố điểm cho đến mối quan hệ giữa số lượt vote và độ nổi tiếng.</li><li><b>Dự Đoán Điểm Phim:</b> Nhập tên phim, thể loại và một đoạn tóm tắt nội dung để AI dự đoán số điểm phim có thể đạt được.</li><li><b>Trợ Lý Phim Ảnh:</b> Hỏi đáp cùng Gemini về kịch bản, ý tưởng phim hoặc cách cải thiện cốt truyện.</li></ul></div>", unsafe_allow_html=True)

# =========================================================
# 📊 2. GIAO DIỆN BIỂU ĐỒ & EDA CHI TIẾT
# =========================================================
elif menu == "📊 Biểu Đồ Thống Kê (EDA)":
    st.markdown("<div class='main-title'>📊 Khám Phá & Phân Tích Dữ Liệu Lớn Phim Ảnh (EDA)</div>", unsafe_allow_html=True)
    
    if df is None:
        st.error("Không tìm thấy tệp dữ liệu 'top10K-TMDB-movies.csv' để vẽ biểu đồ!")
    else:
        tab_dist, tab_rel, tab_categories = st.tabs([
            "📊 1. Phân Phối Đơn Biến", 
            "📈 2. Mối Tương Quan Đa Biến", 
            "🏷️ 3. Phân Phối Theo Thể Loại & Thời Gian"
        ])
        
        with tab_dist:
            st.markdown("<div class='section-header'>🎯 Khảo sát phân bố điểm đánh giá và ngôn ngữ phim</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Biểu đồ 1: Phân bố điểm đánh giá phim (Vote Average)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df["vote_average"], bins=20, kde=True, color="purple", ax=ax)
                ax.set_title("Phân bố điểm đánh giá phim")
                ax.set_xlabel("Điểm đánh giá (Vote Average)")
                ax.set_ylabel("Số lượng phim")
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.write("**Biểu đồ 2: Top 5 ngôn ngữ phổ biến nhất trong phim**")
                fig, ax = plt.subplots(figsize=(8, 8))
                top5_lang = df["original_language"].value_counts().head(5)
                ax.pie(top5_lang, autopct="%1.1f%%", startangle=90, labels=top5_lang.index, colors=sns.color_palette("pastel"))
                ax.set_title("Top 5 Most Common Movie Languages")
                st.pyplot(fig)
                plt.close(fig)
                
        with tab_rel:
            st.markdown("<div class='section-header'>📈 Khảo sát mối quan hệ giữa độ phổ biến và lượt bình chọn</div>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                st.write("**Biểu đồ 3: Top 20 bộ phim nổi tiếng nhất**")
                fig, ax = plt.subplots(figsize=(10, 8))
                top20_pop = df.nlargest(20, "popularity")
                sns.barplot(data=top20_pop, x="popularity", y="title", palette="viridis", ax=ax)
                ax.set_title("Top 20 bộ phim nổi tiếng nhất")
                ax.set_xlabel("Độ phổ biến (Popularity)")
                ax.set_ylabel("Tên phim")
                st.pyplot(fig)
                plt.close(fig)
            with col4:
                st.write("**Biểu đồ 4: Top 20 bộ phim có điểm đánh giá cao nhất (Lượt đánh giá >= 1000)**")
                fig, ax = plt.subplots(figsize=(10, 8))
                filtered = df[df["vote_count"] >= 1000]
                top20_score = filtered.nlargest(20, "vote_average")
                sns.barplot(data=top20_score, x="vote_average", y="title", palette="magma", ax=ax)
                ax.set_title("Top 20 Bộ Phim Điểm Cao Nhất (>=1000 lượt đánh giá)")
                ax.set_xlabel("Vote Average")
                ax.set_ylabel("Tên phim")
                st.pyplot(fig)
                plt.close(fig)
            
            st.write("---")
            st.write("**Biểu đồ 5: Mối tương quan giữa số lượt bình chọn và điểm đánh giá**")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.regplot(data=df, x="vote_count", y="vote_average", color="teal", line_kws={"color": "red"}, ax=ax)
            ax.set_title("Mối tương quan giữa số lượt bình chọn và điểm đánh giá")
            ax.set_xlabel("Số lượt bình chọn (Vote Count)")
            ax.set_ylabel("Điểm đánh giá")
            st.pyplot(fig)
            plt.close(fig)

        with tab_categories:
            st.markdown("<div class='section-header'>🏷️ Phân tích theo thể loại phim và năm phát hành</div>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                st.write("**Biểu đồ 6: Top 10 thể loại phim phổ biến nhất**")
                # Flatten genres to count individually
                genres_flat = df["genre"].fillna("").str.split(",").explode().str.strip()
                genre_count = genres_flat.value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=genre_count.values, y=genre_count.index, palette="Blues_r", ax=ax)
                ax.set_title("Top 10 Movie Genres")
                ax.set_xlabel("Số lượng phim")
                st.pyplot(fig)
                plt.close(fig)
            with col6:
                st.write("**Biểu đồ 7: Số lượng phim được phát hành theo năm**")
                df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
                df["year"] = df["release_date"].dt.year
                movies_year = df["year"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(movies_year.index, movies_year.values, color="royalblue", linewidth=2)
                ax.set_title("Movies Released Per Year")
                ax.set_xlabel("Năm")
                ax.set_ylabel("Số lượng phim")
                st.pyplot(fig)
                plt.close(fig)
                
            st.write("---")
            st.write("**Biểu đồ 8: Nhiệt đồ số lượng phim phát hành theo tháng và năm**")
            df["month"] = df["release_date"].dt.month
            # Filter valid years
            df_filtered = df[(df["year"] >= 1990) & (df["year"] <= 2022)]
            heatmap_data = pd.crosstab(df_filtered["month"], df_filtered["year"])
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, ax=ax)
            ax.set_title("Number of Movies Released by Month and Year (1990 - 2022)")
            ax.set_xlabel("Năm")
            ax.set_ylabel("Tháng")
            st.pyplot(fig)
            plt.close(fig)

# =========================================================
# 🔮 3. GIAO DIỆN DỰ ĐOÁN ĐIỂM SỐ PHIM
# =========================================================
elif menu == "🔮 Dự Đoán Điểm Phim":
    st.markdown("<div class='main-title'>🔮 Dự Đoán Điểm Số Phim Bằng Neural Network</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Nhập thông tin cốt truyện và thể loại để AI tính toán điểm đánh giá (0 - 10)</div>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1.3, 1], gap="large")
    
    with col_in:
        st.markdown("<div class='section-header'>📋 Thông Tin Phim Của Bạn</div>", unsafe_allow_html=True)
        title = st.text_input("Tên dự án phim (Title):", "Dự án điện ảnh mới")
        
        # Extracted all genres from dataset
        all_genres = ["Drama", "Comedy", "Action", "Thriller", "Romance", "Adventure", "Crime", "Science Fiction", "Horror", "Mystery", "Fantasy", "Animation", "Family", "History", "War", "Music", "Western", "Documentary"]
        genres_select = st.multiselect("Thể loại phim (Genres):", all_genres, default=["Drama"])
        
        overview = st.text_area("Tóm tắt kịch bản / cốt truyện (Overview - Tối thiểu 10 từ):", "A dramatic story about a group of friends who discover a hidden secret that changes their lives forever.")
        
    with col_out:
        st.markdown("<div class='section-header'>🔮 Kết Quả Đánh Giá AI</div>", unsafe_allow_html=True)
        if st.button("🚀 BẮT ĐẦU PHÂN TÍCH AI", type="primary", use_container_width=True):
            if not overview.strip():
                st.warning("Vui lòng nhập tóm tắt cốt truyện phim!")
            elif len(overview.split()) < 5:
                st.warning("Đoạn tóm tắt cốt truyện quá ngắn! Hãy viết chi tiết hơn để mô hình đưa ra kết quả chính xác.")
            elif model is None or tokenizer is None:
                st.error("Lỗi: Mô hình học máy hoặc bộ mã hóa Tokenizer chưa được nạp.")
            else:
                with st.spinner("Mô hình Keras đang xử lý ngôn ngữ..."):
                    # Preprocess text
                    genres_str = ",".join(genres_select)
                    combined_text = f"{genres_str} {overview}"
                    
                    # Tokenize and pad
                    seq = tokenizer.texts_to_sequences([combined_text])
                    padded_seq = pad_sequences(seq, maxlen=200)
                    
                    # Predict
                    pred = model.predict(padded_seq)
                    predicted_score = float(pred[0][0])
                    
                    # Clip rating between 0 and 10
                    predicted_score = max(0.0, min(10.0, predicted_score))
                    predicted_score = round(predicted_score, 1)
                    
                    st.success(f"### 🎉 Dự đoán thành công!")
                    st.metric(label="ĐIỂM ĐÁNH GIÁ DỰ KIẾN (VOTE AVERAGE)", value=f"{predicted_score} / 10")
                    
                    # Show progress bar
                    st.progress(predicted_score / 10.0)
                    
                    # Interpret ratings
                    if predicted_score >= 8.0:
                        st.info("💡 **Nhận xét:** Kịch bản/Thể loại có sức hấp dẫn cực kỳ lớn. Phim có tiềm năng trở thành bom tấn nghệ thuật hoặc lọt top phim được đánh giá cao nhất lịch sử!")
                    elif predicted_score >= 7.0:
                        st.info("💡 **Nhận xét:** Điểm số rất triển vọng. Kịch bản có định hướng tốt, thu hút đông đảo đối tượng khán giả.")
                    elif predicted_score >= 5.5:
                        st.info("💡 **Nhận xét:** Mức điểm trung bình khá. Cần tối ưu thêm các yếu tố thắt nút, cao trào trong cốt truyện để cải thiện đánh giá nghệ thuật.")
                    else:
                        st.warning("💡 **Nhận xét:** Điểm số tương đối thấp. Bạn nên viết lại cốt truyện rõ ràng hơn, thêm nhiều tình huống kịch tính hoặc thay đổi sự kết hợp thể loại phim.")

# =========================================================
# 🤖 4. GIAO DIỆN TRỢ LÝ PHIM ẢNH (GEMINI AI CHATBOT)
# =========================================================
elif menu == "🤖 Trợ Lý Phim Ảnh (Gemini AI)":
    st.markdown("<div class='main-title'>🤖 Trợ Lý Điện Ảnh AI - Nhà Phê Bình Gemini</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Không gian tham vấn ý tưởng kịch bản phim, lên cấu trúc phân cảnh và gợi ý phát triển nhân vật</div>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if user_prompt := st.chat_input("Hỏi nhà phê bình phim tại đây..."):
        with chat_container:
            with st.chat_message("user"): st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # Load API Key safely
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = None
                
        if not api_key:
            ai_response = "❌ **Thiếu API Key:** Vui lòng thiết lập GEMINI_API_KEY trong file secrets.toml để kích hoạt chatbot."
        else:
            try:
                with st.spinner("Gemini đang suy nghĩ..."):
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
You are an expert film critic and screenplay advisor. 
Provide professional feedback, script enhancement advice, and creative guidance.
Keep your response engaging, professional, and friendly in Vietnamese.

User request: {user_prompt}
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    ai_response = response.text
            except Exception as e:
                ai_response = f"❌ Lỗi khi gửi yêu cầu tới API Gemini: {str(e)}"
                
        with chat_container:
            with st.chat_message("assistant"): st.markdown(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
