1. Giao diện & Cấu hình hệ thống:

   - Sử dụng nền tảng Streamlit, thiết kế giao diện Dark Mode theo tone màu chủ đạo của Steam.
   - Áp dụng cơ chế lưu bộ nhớ đệm (@st.cache_data, @st.cache_resource) giúp ứng dụng chạy mượt mà, không phải tải lại dữ liệu hoặc huấn luyện lại mô hình ML mỗi khi tương tác.
2. Tab 1 — Tìm kiếm & Gợi ý (Tính năng cốt lõi):

   - Công cụ tìm kiếm thông minh: Hỗ trợ tìm kiếm theo tên, viết tắt phổ biến (CSGO, GTA5, TF2...) và tự động sửa lỗi chính tả nhẹ.
   - Tính năng "Tìm Nhanh": Cung cấp sẵn 10 nút bấm cho các tựa game nổi tiếng để xem nhanh kết quả.
   - Thuật toán gợi ý: Sử dụng kỹ thuật TF-IDF và Cosine Similarity dựa trên Genres/Tags/Categories để đề xuất 3 game tương đồng nhất.
   - Tích hợp Trí tuệ nhân tạo: Sử dụng mô hình Gemini (gemini-2.5-flash) để tự động tạo nhận xét ngắn gọn, khách quan bằng tiếng Việt cho game được đề xuất.
   - Dự đoán mức độ yêu thích: Gọi mô hình ML để dự báo xác suất (%) game đó được cộng đồng đánh giá tích cực (>= 75%).
3. Tab 2 — Phân tích dữ liệu (Dashboard):

   - Hiển thị các chỉ số KPIs chính: Tổng số game, Giá trung bình, % Game miễn phí, Lượt đánh giá tích cực kỷ lục.
   - Hệ thống trực quan hóa gồm 11 biểu đồ (Matplotlib & Seaborn):
     + Top 10 thể loại phổ biến.
     + Tỷ lệ game hỗ trợ Tiếng Anh.
     + Phân phối giá bán (Dưới $60).
     + Xu hướng phát hành game theo năm.
     + Tương quan đánh giá Tích cực vs Tiêu cực.
     + Ma trận tương quan giữa các biến số.
     + Thời gian chơi trung bình theo thể loại chính.
     + So sánh phân phối giá giữa nhóm game Indie và nhóm game khác.
     + Top 10 Nhà phát hành nhiều game nhất.
     + Tỷ lệ hỗ trợ đa nền tảng (Windows, macOS, Linux).
     + Tỷ lệ đánh giá tích cực theo từng khoảng giá bán.
   - Mỗi biểu đồ đều đi kèm hộp văn bản "Insight" đúc kết nhận xét từ dữ liệu thực tế.
4. Tab 3 — Mô hình Machine Learning (AI Dashboard):

   - Thuật toán sử dụng: Gradient Boosting Classifier giải bài toán phân loại nhị phân (Dự đoán game có đạt >= 75% đánh giá tích cực hay không).
   - Cơ chế tối ưu hóa: Sử dụng kỹ thuật StratifiedKFold (3-fold) kết hợp RandomizedSearchCV để tự động tìm bộ siêu tham số tốt nhất khi chạy.
   - Chỉ số đánh giá mô hình: Hiển thị trực quan các điểm số Accuracy, Precision, Recall, F1-Score và ROC-AUC.
   - Biểu đồ trực quan phụ trợ:

   + Ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap để xem tỷ lệ dự đoán đúng/sai.
     + Độ quan trọng của đặc trưng (Feature Importance) hiển thị Top 15 thuộc tính ảnh hưởng nhất đến kết quả dự đoán (Ví dụ: Thời gian chơi, giá bán, số thành tựu...).

Slide của bạn: [canva.link/otjq2mvv2rzibaz](https://canva.link/otjq2mvv2rzibaz)
