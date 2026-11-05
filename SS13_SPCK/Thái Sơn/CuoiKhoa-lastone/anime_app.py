import streamlit as st  
import pandas as pd     
import numpy as np
import google.generativeai as genai
import os              
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv  
from ml_module import AnimePredictor  # Import module ML chuẩn OOP

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
    df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
    df['Members'] = pd.to_numeric(df['Members'], errors='coerce').fillna(0)
    df['Favorites'] = pd.to_numeric(df['Favorites'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Score', 'Name', 'Genres', 'Source'])
    
    # Lấy danh sách thể loại, nguồn gốc, rating và studio phục vụ cho Form nhập liệu đồng bộ
    all_genres = sorted(list(set([g.strip() for sublist in df['Genres'].str.split(',') for g in sublist])))
    all_sources = sorted(df['Source'].unique().tolist())
    all_ratings = sorted(df['Rating'].fillna('Unknown').unique().tolist())
    all_studios = sorted(df['Studios'].fillna('Unknown').unique().tolist())
    
    return df, all_genres, all_sources, all_ratings, all_studios

df, list_genres, list_sources, list_ratings, list_studios = load_base_data()

# GIAO DIỆN CHÍNH 
st.title("Otaku Hub: Hệ Thống Xếp Hạng, Dự Đoán & Tư Vấn Anime")

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
tab1, tab2, tab3 = st.tabs(["Xếp Hạng & Khám Phá Dữ Liệu (EDA)", "Dự Đoán Điểm Số (ML)", "Trò Chuyện Cùng AI Otaku"])

# TAB 1: BẢNG XẾP HẠNG TRỰC QUAN VÀ BIỂU ĐỒ EDA
with tab1:
    st.subheader("Danh Sách Anime Phù Hợp Định Dạng TV")
    st.caption(f"Tìm thấy {len(df_filtered)} bộ anime thỏa mãn bộ lọc hiện tại.")
    
    df_ranking = df_filtered.sort_values(by='Score', ascending=False).head(20)
    if not df_ranking.empty:
        # Bảng dữ liệu tương tác rộng
        st.dataframe(
            df_ranking[['Name', 'Score', 'Genres', 'Episodes', 'Source', 'Studios']],
            column_config={
                "Name": "Tên Anime",
                "Score": st.column_config.NumberColumn("Điểm số", format="%.2f ⭐"),
                "Genres": "Thể loại",
                "Episodes": "Số tập",
                "Source": "Nguồn gốc tác phẩm",
                "Studios": "Xưởng phim"
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.subheader("📊 Phân Tích Khám Phá Dữ Liệu (EDA Dashboard)")
        
        # HÀNG BIỂU ĐỒ 1: Top 10 của Streamlit & Phân phối điểm số (Seaborn)
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**1. Top 10 Anime có điểm số cao nhất hiện tại**")
            st.bar_chart(data=df_ranking.head(10), x="Name", y="Score", color="#FF4B4B")
            
        with col_chart2:
            st.markdown("**2. Phân phối điểm số toàn tập dữ liệu (Score Distribution)**")
            fig1, ax1 = plt.subplots(figsize=(6, 4.2))
            sns.histplot(df['Score'], bins=20, kde=True, color='#FF4B4B', ax=ax1)
            ax1.set_xlabel('Điểm số (Score)')
            ax1.set_ylabel('Số lượng phim')
            st.pyplot(fig1)
            
        st.markdown("---")
        # HÀNG BIỂU ĐỒ 2: Top Studio chất lượng nhất & Ma trận tương quan
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("**3. Top 10 Xưởng phim (Studios) có điểm đánh giá trung bình cao nhất**")
            studio_stats = df[df['Studios'] != 'Unknown'].groupby('Studios').agg({'Score': 'mean', 'Name': 'count'})
            top_studios = studio_stats[studio_stats['Name'] >= 3].sort_values(by='Score', ascending=False).head(10)
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_studios['Score'], y=top_studios.index, palette='Reds_r', ax=ax2)
            ax2.set_xlabel('Điểm số trung bình')
            ax2.set_ylabel('Tên Xưởng phim (Studio)')
            st.pyplot(fig2)
            
        with col_chart4:
            st.markdown("**4. Ma trận tương quan giữa các thuộc tính (Correlation Matrix)**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            corr_features = df[['Score', 'Episodes', 'Members', 'Favorites']].corr()
            sns.heatmap(corr_features, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax3)
            st.pyplot(fig3)

        st.markdown("---")
        # HÀNG BIỂU ĐỒ 3 (MỚI BỔ SUNG): Mối quan hệ Members-Score & Phân tích Boxplot theo Source
        col_chart5, col_chart6 = st.columns(2)

        with col_chart5:
            st.markdown("**5. Mối liên hệ giữa Quy mô cộng đồng (Members) và Điểm số (Score)**")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            # Sử dụng regplot để vẽ điểm phân tán kèm đường xu hướng linear regression mẫu mực
            sns.regplot(data=df, x='Members', y='Score', scatter_kws={'alpha':0.3, 'color':'#FF4B4B'}, line_kws={'color':'blue'}, ax=ax4)
            ax4.set_xlabel('Số lượng thành viên (Members)')
            ax4.set_ylabel('Điểm số (Score)')
            st.pyplot(fig4)

        with col_chart6:
            st.markdown("**6. Phân tích phân phối Điểm số theo Nguồn gốc tác phẩm (Source)**")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            # Lọc ra top 6 nguồn phổ biến nhất để biểu đồ không bị rối chữ
            top_sources = df['Source'].value_counts().head(6).index
            df_top_sources = df[df['Source'].isin(top_sources)]
            
            sns.boxplot(data=df_top_sources, x='Source', y='Score', palette='Set2', ax=ax5)
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')
            ax5.set_xlabel('Nguồn gốc (Source)')
            ax5.set_ylabel('Điểm số (Score)')
            st.pyplot(fig5)
            
    else:
        st.warning("Không tìm thấy kết quả phù hợp với bộ lọc hiện tại.")

# TAB 2: FORM NHẬP LIỆU DỰ ĐOÁN MODEL ML
with tab2:
    st.subheader("Mô Hình Dự Đoán Điểm Số Cho Dự Án Anime Sắp Ra Mắt")
    st.info(f"**Thông tin mô hình:** Đã được huấn luyện bằng thuật toán Random Forest. Độ chính xác ước lượng (R2 Score): **{model_accuracy*100:.2f}%**.")
    
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
                        "vui vẻ, thi thoảng đan xen các thuật ngữ anime quen thuộc (waifu, husbando, xịn sò, peak...). "
                        "Nếu tóm tắt phim, hãy viết lôi cuốn, ngắn gọn và không tiết lộ (spoil) kết phim.\n\n"
                        f"Câu hỏi từ người dùng: {user_query}"
                    )
                    response = model.generate_content(system_prompt)
                    st.chat_message("assistant").write(response.text)
                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}")
        elif not user_query:
            st.info("Vui lòng nhập nội dung câu hỏi.")