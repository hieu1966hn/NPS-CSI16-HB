Tên dự án: Hệ thống Khám phá Game Steam & Trợ lý Đánh giá Game Tự động.

Link Dataset (Kaggle): kaggle.com/datasets/nikdavis/steam-store-games (Dữ liệu về 27,000+ game Steam bao gồm thể loại, giá, đánh giá).

Vấn đề cần giải quyết: Dựa trên những game người dùng đã từng chơi (nhập vào 3-5 game), hệ thống dùng ML gợi ý ra tựa game "chân ái" tiếp theo. Điểm nhấn là Gemini AI sẽ đọc qua các review của game đó để viết một bài "tóm tắt dụ dỗ" (pitch) cực kỳ thuyết phục, lầy lội theo văn phong game thủ.

Kiến thức áp dụng:

CS Basic: Xử lý chuỗi tên game, vòng lặp tìm kiếm tag thể loại game.

CS Advanced: Lập trình OOP class quản lý thư viện game của người dùng. Xử lý ngoại lệ gọi API khi lấy thêm thông tin giá game hiện tại.

CS Intensive:

Pandas: Lọc và nhóm dữ liệu để tìm ra các game có tỷ lệ Positive Review > 80%.

Scikit-learn: Áp dụng Content-Based Filtering (lọc cộng tác) bằng Cosine Similarity để so sánh các "Tags" của game cũ và game mới nhằm tìm ra mức độ phù hợp.

Matplotlib/Seaborn: Vẽ biểu đồ xu hướng thể loại game đang hot trong năm.

Streamlit: Xây dựng web app với giao diện Dark Mode xịn xò như ứng dụng Steam.

Generative AI (Gemini): Tóm tắt điểm mạnh yếu của game. VD : Prompt: "Dựa vào các tags [Hành động, Máu me, Cốt truyện sâu sắc] của tựa game này. Hãy đóng vai một game thủ try-hard, viết một bài review ngắn gọn, dùng nhiều tiếng lóng gaming để dụ dỗ tôi mua tựa game này ngay lập tức."
