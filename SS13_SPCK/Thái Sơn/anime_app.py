import streamlit as st  
import pandas as pd     
import google.generativeai as genai
import os              
from dotenv import load_dotenv  
from ml_module import AnimePredictor  # Import module ML chuẩn OOP vừa tạo

# Tải cấu hình từ file .env
load_dotenv()

# CẤU HÌNH GIAO DIỆN
st.set_page_config(
    page_title="Otaku Hub - Advanced Dashboard & Predictor",
    layout="wide"
)

# CẤU HÌNH GEMINI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("Lỗi: Không tìm thấy 'GEMINI_API_KEY' trong file .env. Vui lòng kiểm tra lại!")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Sử dụng mô hình thế hệ mới ổn định
model = genai.GenerativeModel('gemini-2.5-flash')

# KHỞI TẠO VÀ CACHE MÔ HÌNH MACHINE LEARNING
@st.cache_resource
def init_ml_model():
    predictor = AnimePredictor()
    accuracy = predictor.train()
    return predictor, accuracy

predictor, model_accuracy = init_ml_model()

# TẢI DỮ LIỆU ĐỂ HIỂN THỊ VÀ LẤY OPTION CHO FORM
@st.cache_data
def load_base_data():
    df = pd.read_csv('anime-filtered.csv')
    df = df[df['Type'] == 'TV'].copy()
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score', 'Name', 'Genres', 'Source'])
    
    # Lấy danh sách thể loại, nguồn gốc, rating và studio phục vụ cho Form nhập liệu đồng bộ
    all_genres = sorted(list(set([g.strip() for sublist in df['Genres'].str.split(',') for g in sublist])))
    all_sources = sorted(df['Source'].unique().tolist())
    all_ratings = sorted(df['Rating'].fillna('Unknown').unique().tolist())
    all_studios = sorted(df['Studios'].fillna('Unknown').unique().tolist())
    
    return df, all_genres, all_sources, all_ratings, all_studios

df, list_genres, list_sources, list_ratings, list_studios = load_base_data()

# GIAO DIỆN CHÍNH 
st.title("Otaku Hub v2: Hệ Thống Xếp Hạng, Dự Đoán & Tư Vấn Anime")

# Thanh bên trái (Sidebar) phục vụ cho tính năng Lọc bảng xếp hạng cũ
st.sidebar.header("Bộ Lọc Bảng Xếp Hạng")
selected_genres = st.sidebar.multiselect("Chọn thể loại yêu thích:", options=list_genres, default=[])
min_score = st.sidebar.slider("Điểm số tối thiểu:", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

# Xử lý lọc dữ liệu nền cho Tab 1
df_filtered = df[df['Score'] >= min_score]
if selected_genres:
    for genre in selected_genres:
        df_filtered = df_filtered[df_filtered['Genres'].str.contains(genre, na=False)]

# PHÂN CHIA CÁC TAB CHỨC NĂNG CHÍNH
tab1, tab2, tab3 = st.tabs(["Xếp Hạng Trực Quan", "Dự Đoán Điểm Số (ML)", "Trò Chuyện Cùng AI Otaku"])

# TAB 1: BẢNG XẾP HẠNG TRỰC QUAN
with tab1:
    st.subheader("Danh Sách Anime Phù Hợp Định Dạng TV")
    st.caption(f"Tìm thấy {len(df_filtered)} bộ anime thỏa mãn bộ lọc.")
    
    df_ranking = df_filtered.sort_values(by='Score', ascending=False).head(20)
    if not df_ranking.empty:
        st.dataframe(
            df_ranking[['Name', 'Score', 'Genres', 'Episodes', 'Source']],
            column_config={
                "Name": "Tên Anime",
                "Score": st.column_config.NumberColumn("Điểm số", format="%.2f ⭐"),
                "Genres": "Thể loại",
                "Episodes": "Số tập",
                "Source": "Nguồn gốc tác phẩm"
            },
            use_container_width=True,
            hide_index=True
        )
        st.subheader("Biểu đồ điểm số Top 10 Anime đứng đầu")
        st.bar_chart(data=df_ranking.head(10), x="Name", y="Score", color="#FF4B4B")
    else:
        st.warning("Không tìm thấy kết quả phù hợp với bộ lọc hiện tại.")

# TAB 2: FORM NHẬP LIỆU DỰ ĐOÁN MODEL ML
with tab2:
    st.subheader("Mô Hình Dự Đoán Điểm Số Cho Dự Án Anime Sắp Ra Mắt")
    st.info(f"**Thông tin mô hình:** Đã được huấn luyện bằng thuật toán Random Forest. Độ chính xác ước lượng (R2 Score): **{model_accuracy*100:.2f}%**.")
    
    # Tạo form nhập liệu đầy đủ tham số cho mô hình nâng cao
    with st.form("ml_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            input_name = st.text_input("Tên dự án Anime (Tùy chọn):", value="Anime Project X")
            input_episodes = st.number_input("Số tập phim dự kiến phát sóng:", min_value=1, max_value=1000, value=12)
            input_members = st.number_input("Số lượng thành viên theo dõi dự kiến (Members):", min_value=0, value=50000, step=1000)
            input_favorites = st.number_input("Số lượng yêu thích dự kiến (Favorites):", min_value=0, value=1500, step=100)
            
        with col2:
            input_source = st.selectbox("Nguồn gốc phát triển nội dung (Source):", options=list_sources, index=list_sources.index('Manga') if 'Manga' in list_sources else 0)
            input_rating = st.selectbox("Phân loại giới hạn độ tuổi (Rating):", options=list_ratings, index=0)
            input_studio = st.selectbox("Hãng sản xuất dự kiến (Studio):", options=list_studios, index=0)
            input_genres = st.multiselect("Lựa chọn các thể loại cấu thành (Genres):", options=list_genres, default=['Action'] if 'Action' in list_genres else [])
            
        input_sypnopsis = st.text_area("Tóm tắt nội dung / Kịch bản (Synopsis):", value="An exciting adventure of heroes fighting for peace in a fantasy world.")
            
        submit_prediction = st.form_submit_button("Bắt đầu tính toán & Dự đoán")
        
    if submit_prediction:
        if not input_genres:
            st.warning("Vui lòng chọn ít nhất một thể loại cho Anime!")
        else:
            with st.spinner("Mô hình đang phân tích các đặc trưng đặc tả và văn bản cốt truyện..."):
                # Gửi ĐẦY ĐỦ 8 THAM SỐ vào hàm dự đoán khớp với thiết kế file OOP
                pred_score = predictor.predict_score(
                    episodes=input_episodes, 
                    members=input_members, 
                    favorites=input_favorites, 
                    source=input_source, 
                    rating=input_rating, 
                    genres_list=input_genres, 
                    studio=input_studio, 
                    sypnopsis_text=input_sypnopsis
                )
                
                st.success(f"Kết quả phân tích thành công cho dự án: **{input_name}**")
                st.metric(label="Điểm số dự kiến đạt được (Predicted Score)", value=f"{pred_score:.2f} / 10 ⭐")

# TAB 3: TRÒ CHUYỆN CÙNG AI OTAKU
with tab3:
    st.subheader("Tư Vấn & Tóm Tắt Cùng AI Otaku Thông Thái")
    user_query = st.text_input("Nhập câu hỏi của bạn tại đây:", placeholder="Ví dụ: Gợi ý cho mình vài bộ anime học đường kết hợp hành động đỉnh cao nhé?")
    
    if st.button("Hỏi Otaku AI"):
        if user_query and GEMINI_API_KEY:
            with st.spinner("AI Otaku đang lục lại ký ức, đợi giây lát..."):
                try:
                    system_prompt = (
                        "Bạn là một vị Otaku thông thái, đam mê và có am hiểu sâu sắc về thế giới Anime. "
                        "Hãy trả lời câu hỏi sau của người dùng bằng tiếng Việt với phong cách thân thiện, "
                        "vui vẻ, thi thoảng đan xen các thuật ngữ anime quen violent (waifu, husbando, xịn sò, peak...). "
                        "Nếu tóm tắt phim, hãy viết lôi cuốn, ngắn gọn và không tiết lộ (spoil) kết phim.\n\n"
                        f"Câu hỏi từ người dùng: {user_query}"
                    )
                    response = model.generate_content(system_prompt)
                    st.chat_message("assistant").write(response.text)
                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}")
        elif not user_query:
            st.info("Vui lòng nhập nội dung câu hỏi.")