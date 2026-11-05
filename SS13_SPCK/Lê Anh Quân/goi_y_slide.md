# GỢI Ý CẤU TRÚC VÀ NỘI DUNG SLIDE THUYẾT TRÌNH SPCK

Chào Anh Quân, thầy đã xem qua phần mô tả dự án trong file `readme.md` và mã nguồn thực tế của em. Thầy phát hiện ra một **mâu thuẫn kỹ thuật lớn** giữa cách em ghi ý tưởng và phần code thực tế:

* **Mô tả cũ trong Readme:** Em ghi là *"Huấn luyện mô hình Random Forest bằng Scikit-learn để làm hệ thống gợi ý phim"*.
* **Code thực tế:** Em thực tế đang huấn luyện **Mạng nơ-ron học sâu (Deep Learning Sequential Model bằng Keras/TensorFlow)** để **dự đoán điểm đánh giá phim (Rating Predictor)**. Còn tính năng **Gợi ý phim** lại được em tích hợp thông qua **Gemini AI Chatbot**.

Nếu mang slide cũ đi thuyết trình, em sẽ bị giám khảo trừ điểm vì slide không khớp với code. Thầy gợi ý em chỉnh sửa tên đề tài và cấu trúc lại các trang slide Canva theo bộ khung **10 slide chuẩn** dưới đây:

---

## 📋 KHUNG 10 SLIDE TRÌNH BÀY ĐỀ XUẤT

### Slide 1: Trang tiêu đề
* **Tiêu đề gợi ý:** **Hệ Thống Dự Đoán Điểm Số & Tư Vấn Phim Ảnh Bằng Trí Tuệ Nhân Tạo AI**
* **Thông tin:** Người thực hiện: Lê Anh Quân - Lớp: CSI16-HB.

### Slide 2: Hiện trạng & Vấn đề thực tế
* **Nội dung:** 
  * Hàng ngàn bộ phim ra mắt mỗi năm khiến các nhà làm phim, biên kịch gặp khó khăn trong việc đánh giá tiềm năng kịch bản trước khi bấm máy.
  * Khán giả bị quá tải thông tin, khó tìm được bộ phim phù hợp với sở thích cá nhân hóa một cách nhanh chóng.

### Slide 3: Giải pháp đề xuất
* **Nội dung:** Xây dựng ứng dụng Web tích hợp hai tầng trí tuệ nhân tạo:
  * **Tầng 1 (Deep Learning):** Dự đoán điểm đánh giá (thang điểm 0 - 10) của một bộ phim từ văn bản tóm tắt cốt truyện và các thể loại phim.
  * **Tầng 2 (Generative AI):** Trợ lý ảo tư vấn, phân tích kịch bản và gợi ý phim cá nhân hóa qua ngôn ngữ tự nhiên.

### Slide 4: Bộ dữ liệu huấn luyện (Dataset)
* **Nội dung:**
  * Nguồn dữ liệu: **TMDB 10K Movies** (từ Kaggle).
  * Quy mô: 10,000 bộ phim.
  * Các trường thông tin sử dụng: Tên phim, thể loại (genres), cốt truyện (overview), và điểm đánh giá thực tế (`vote_average`) làm nhãn mục tiêu.

### Slide 5 & Slide 6: Phân tích dữ liệu trực quan (EDA)
* **Nội dung:** Chọn đưa 3 biểu đồ nổi bật nhất từ Notebook lên kèm nhận xét ngắn:
  * **Biểu đồ phân phối điểm phim:** Cho thấy điểm số phân bố chuẩn dạng hình chuông (tập trung chủ yếu ở mức 6.0 - 7.0).
  * **Biểu đồ ma trận tương quan:** Thể hiện mối tương quan giữa số lượt vote và điểm đánh giá (phim có nhiều lượt vote có xu hướng đạt điểm cao hơn).
  * **Top thể loại phim:** Hài kịch (Comedy) và Chính kịch (Drama) chiếm số lượng áp đảo.

### Slide 7: Thiết lập Mô hình học sâu (Keras Deep Learning)
* **Nội dung:** Trình bày kiến trúc mạng nơ-ron hồi quy Keras Sequential:
  * Lớp **Embedding** (nhận diện 10,000 từ vựng cốt truyện, chuyển thành vector 128 chiều).
  * Lớp **GlobalAveragePooling1D** (nén vector để giảm chiều dữ liệu).
  * Các lớp ẩn **Dense** (128 và 64 nơ-ron) kết hợp hàm kích hoạt **ReLU** để học các đặc trưng phi tuyến.
  * Lớp đầu ra **Dense(1)** để dự báo điểm số liên tục.

### Slide 8: Kết quả huấn luyện & Đánh giá mô hình
* **Nội dung:** 
  * Hàm loss sử dụng khi huấn luyện: **Mean Squared Error (MSE)**.
  * Chỉ số sai số đo lường: **Mean Absolute Error (MAE)** đạt **0.27** trên tập kiểm thử.
  * *Ý nghĩa:* Sai số trung bình của mô hình chỉ chênh lệch khoảng **0.27 điểm** so với điểm thực tế của khán giả bình chọn (mức độ dự đoán cực kỳ chính xác).

### Slide 9: Demo Giao diện Web App (Streamlit)
* **Nội dung:** Đưa các hình ảnh chụp màn hình thực tế của ứng dụng Web lên slide:
  * Giao diện form nhập cốt truyện và nút phân tích AI.
  * Tab trực quan hóa 8 biểu đồ thống kê tương tác.
  * Khung Chatbot trò chuyện cùng nhà phê bình phim Gemini.

### Slide 10: Tổng kết & Hướng phát triển
* **Nội dung:**
  * *Ưu điểm:* Giao diện mượt mà, trực quan hóa tốt, mô hình học sâu có độ chính xác cao, bảo mật API Key an toàn.
  * *Hạn chế & Hướng đi tiếp theo:* Tích hợp thêm dữ liệu doanh thu phòng vé để giúp nhà sản xuất đánh giá thêm về khả năng sinh lời của kịch bản phim.
