# TỔNG QUAN SẢN PHẨM CUỐI KHÓA (SPCK) — LỚP NPS-CSI16-HB
**Khóa học:** Computer Scientist Intensive (CSI) — MindX School  
**Cơ sở:** Hòa Bình / Hà Nội  
**Mục tiêu đầu ra:** Web Application (Streamlit), Phân tích khám phá dữ liệu (EDA $\ge$ 8 biểu đồ), Mô hình Machine Learning tối ưu đạt $R^2/\text{Accuracy} \ge 80\%$, Chatbot AI hỗ trợ người dùng (Gemini API), Mã nguồn chuẩn hướng đối tượng (OOP).

---

## DANH SÁCH ĐỀ TÀI & SẢN PHẨM CỦA HỌC VIÊN

| STT | Học viên | Tên đề tài & Ứng dụng | Bộ dữ liệu (Dataset) | Công nghệ & Tính năng chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Hải Long** | **Game Steam Explorer & AI Reviewer** | Steam store games dataset (Kaggle) | Streamlit (Dark Mode Steam theme), Cosine Similarity gợi ý game theo Tags, Chatbot Gemini đóng vai Game thủ review game. |
| **2** | **Phúc Nguyên** | **Hệ thống phân tích kinh tế vĩ mô & Tư vấn đầu tư Việt Nam** | Macro indicators 2015–2026 (CPI, P/E VN-Index, Lãi suất, US10Y, VN10Y, DXY) | Random Forest / Machine Learning dự báo điểm mua bán, Dashboard đa chỉ số vĩ mô, Trợ lý AI cố vấn tài chính vĩ mô. |
| **3** | **Quốc Tuấn** | **Hệ thống dự đoán kết quả học tập & Tư vấn học tập bằng AI** | Student Performance Factors (Kaggle) | Deep Learning / Neural Network dự đoán điểm số học sinh, Streamlit Web App, Phân tích biểu đồ tương quan, Slide thuyết trình hoàn chỉnh (PDF). |
| **4** | **Bảo An** | **Dự đoán calo tiêu thụ & Tư vấn dinh dưỡng thể thao** | Calories Burned dataset | Mô hình Keras/Neural Network dự đoán calo tiêu hao theo chỉ số cơ thể & bài tập, Giao diện tương tác người dùng Streamlit. |
| **5** | **Khánh Lâm** | **Hệ thống phân tích & Dự án cuối khóa** | Bộ dữ liệu khảo sát chuyên sâu | EDA trực quan hóa dữ liệu, tài liệu mô tả kiến trúc và giải pháp. |
| **6** | **Quang Minh** | **Định giá máy ảnh kỹ thuật số cũ & Trợ lý tư vấn thiết bị** | Amazon & Digital Camera datasets | Tiền xử lý & làm sạch dữ liệu máy ảnh, Mô hình hồi quy Scikit-learn định giá thiết bị theo thông số kỹ thuật, Chatbot tư vấn chọn máy ảnh cho người mới. |
| **7** | **Quốc Quân** | **Dự đoán khả năng trúng tuyển Đại học (Admission Predictor)** | Graduate Admission dataset | Huấn luyện mô hình Scikit-learn đánh giá tỷ lệ trúng tuyển dựa trên GPA, GRE, TOEFL, Research; Giao diện Streamlit tra cứu kết quả trực quan. |
| **8** | **Chí Vinh** | **Dashboard phân tích & Dự báo thất nghiệp toàn cầu 2014–2024** | Global Unemployment dataset | Hệ thống Dashboard biểu đồ chuyên sâu (Choropleth Map, Violin, Boxplot, Multi-line), Mô hình dự báo tỷ lệ thất nghiệp theo quốc gia và độ tuổi, Bot tư vấn việc làm. |
| **9** | **Thái Sơn** | **Phân tích xu hướng & Dự đoán đánh giá Anime TV Shows** | Anime TV Shows dataset 2023 | Thuật toán hồi quy dự đoán Score Anime, Dashboard phân tích xu hướng Studio & Thể loại, Chatbot Gemini AI Otaku tóm tắt và đánh giá tác phẩm. |
| **10** | **Lê Anh Quân** | **Hệ thống gợi ý phim & Chatbot trợ lý điện ảnh** | Top 10K TMDB Movies & MovieLens | Mô hình gợi ý phim kết hợp NLP / Tokenizer, Tìm kiếm phim theo sở thích ngôn ngữ tự nhiên, Trợ lý điện ảnh Gemini review và tóm tắt phim. |

---

## HƯỚNG DẪN CHẠY THỬ MÃ NGUỒN CỦA HỌC VIÊN
1. Di chuyển vào thư mục của học viên cần kiểm tra:
   ```bash
   cd "SS13_SPCK/<Tên_Học_Viên>"
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Khởi chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py
   # hoặc tên file chính của học viên (ví dụ: streamlit_app.py / anime_app.py)
   ```
