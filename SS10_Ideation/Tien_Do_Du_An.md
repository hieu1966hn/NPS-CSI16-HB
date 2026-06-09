# BẢNG PHÂN CHIA NHIỆM VỤ DỰ ÁN CUỐI KHÓA - LỚP NPS-CSI16-HB

Dưới đây là kế hoạch chi tiết từng tuần cho từng học viên nhằm đảm bảo đáp ứng đầy đủ yêu cầu đầu ra (Web App Streamlit, Mô hình ML tối ưu Accuracy >= 80%, Trực quan hóa dữ liệu >= 8 biểu đồ thuộc 5 dạng khác nhau và rút ra Insight/Next Action, Tích hợp Chatbot Gemini).

---

## 1. LỘ TRÌNH TỔNG QUAN (4 TUẦN)

*   **Tuần 1: Khởi động & Chuẩn bị dữ liệu (Ideation & Data Setup)**
    *   Chốt ý tưởng và chuẩn bị bộ dữ liệu (dataset) sạch từ Kaggle hoặc nguồn tin cậy.
    *   Tải dữ liệu về, viết notebook phân tích sơ bộ (EDA cơ bản), kiểm tra kích thước dữ liệu, các cột thuộc tính và giá trị thiếu (null).
*   **Tuần 2: Phân tích trực quan & Huấn luyện mô hình (Data Visualization & Model Training)**
    *   **Trực quan hóa**: Vẽ tối thiểu **8 biểu đồ** thuộc ít nhất **5 dạng khác nhau** (Bar, Line, Pie, Scatter, Heatmap, Boxplot, Violin, Histogram, Map...). Viết nhận xét rút ra **Insight** (Sự thật ngầm hiểu) và **Next Action** (Hành động tiếp theo).
    *   **Học máy**: Tiền xử lý dữ liệu (xử lý null, chuẩn hóa dữ liệu, mã hóa biến phân loại). Chia tập Train/Test. Huấn luyện mô hình Machine Learning (Hồi quy hoặc Phân loại) và tối ưu đạt độ chính xác **Accuracy/R2-Score >= 80%**.
*   **Tuần 3: Xây dựng Giao diện & Tích hợp Chatbot AI (Web App & Chatbot Integration)**
    *   **Giao diện**: Thiết kế giao diện Streamlit (Dark/Light mode, sắp xếp bố cục trực quan). Đưa 8 biểu đồ phân tích dữ liệu lên Web App.
    *   **Tính năng ML**: Tạo form nhập liệu cho người dùng -> Đưa vào model dự đoán -> Trả kết quả dự đoán trực quan trên Web App.
    *   **Chatbot**: Tích hợp API Gemini thành một trợ lý ảo tư vấn riêng theo chủ đề của dự án (đóng vai PT ảo, cố vấn học tập, chuyên gia phim ảnh, cố vấn tài chính...).
*   **Tuần 4: Đóng gói, Tối ưu & Triển khai (Packaging & Deployment)**
    *   Áp dụng kiến thức **CS Advanced (OOP)** để đóng gói mã nguồn thành các Class rõ ràng (ví dụ: `DataAnalyzer`, `PredictorModel`, `GeminiChatbot`).
    *   Xử lý ngoại lệ (Exception Handling) khi gọi API Gemini hoặc khi người dùng nhập sai dữ liệu.
    *   Đẩy code lên GitHub và **deploy lên internet** thông qua **Streamlit Community Cloud** (hoặc Hugging Face Spaces). Kiểm thử toàn bộ ứng dụng và chuẩn bị slide thuyết trình.

---

## 2. BẢNG PHÂN CHIA NHIỆM VỤ CHI TIẾT TỪNG HỌC VIÊN (CẬP NHẬT MỚI NHẤT)

| Học viên | Tên ứng dụng & Dataset đề xuất | Nhiệm vụ Tuần 1 | Nhiệm vụ Tuần 2 | Nhiệm vụ Tuần 3 | Nhiệm vụ Tuần 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Đào Bảo An** | **Dự đoán calo tiêu hao & Lên kế hoạch tập luyện**<br>*(Dataset: fmendes/fmendesdat263xdemos)* | - Tải dataset Kaggle.<br>- Đọc dữ liệu bằng Pandas.<br>- Xác định các đặc trưng đầu vào (Nhịp tim, Thời gian tập, Giới tính, Tuổi) và đầu ra cần dự đoán (Calo). | - Vẽ 8 biểu đồ (Line chart nhịp tim, Scatter duration-calo, Heatmap tương quan...).<br>- Rút ra insight tập luyện.<br>- Train model Random Forest Regressor tối ưu R2-Score >= 80%. | - Xây dựng giao diện nhập chỉ số cơ thể & bài tập.<br>- Kết nối mô hình dự đoán Calo lên web.<br>- Tích hợp Gemini Chatbot đóng vai **PT ảo** tư vấn thực đơn và bài tập. | - Viết code theo chuẩn OOP.<br>- Xử lý lỗi nhập liệu và API.<br>- Deploy lên Streamlit Cloud.<br>- Chuẩn bị slide demo sản phẩm. |
| **Lê Anh Quân** | **Hệ thống gợi ý phim**<br>*(Dataset: top-rated-tmdb-movies-10k & movielens-dataset)* | - Tải bộ dữ liệu từ Kaggle.<br>- Đọc và liên kết thông tin phim, lượt đánh giá.<br>- Thiết lập khung sườn dự án. | - Vẽ 8 biểu đồ (Bar chart thể loại phổ biến, Line chart xu hướng điểm theo năm, Scatter thời lượng vs điểm...).<br>- Train mô hình Random Forest gợi ý phim.<br>- Đạt Accuracy/R2 >= 80%. | - Thiết kế Web App gợi ý phim bằng Streamlit.<br>- Giao diện nhập sở thích bằng ngôn ngữ tự nhiên.<br>- Tích hợp Gemini đóng vai **Trợ lý phim ảnh** tư vấn, tóm tắt nội dung phim. | - Sử dụng OOP thiết kế Class gợi ý và quản lý cơ sở dữ liệu phim.<br>- Đẩy code lên GitHub & deploy Streamlit Cloud.<br>- Kiểm tra hoạt động của chatbot Gemini. |
| **Quốc Quân** | **Hệ thống dự đoán khả năng đỗ đại học**<br>*(Dataset: Admission_Predict.csv)* | - Tải dataset Kaggle.<br>- Khảo sát các biến: GPA, GRE, TOEFL, SOP, LOR, Research.<br>- Xác định biến mục tiêu (Chance of Admit). | - Vẽ 8 biểu đồ (Scatter GPA - Đỗ đại học, Bar rating trường, Heatmap tương quan...).<br>- Rút ra insight tuyển sinh.<br>- Train model Linear Regression/Random Forest Regressor đạt R2 >= 80%. | - Thiết kế form nhập điểm GRE, TOEFL, GPA trên Streamlit.<br>- Xuất ra phần trăm cơ hội đỗ.<br>- Tích hợp Gemini đóng vai **Cố vấn tuyển sinh** gợi ý cách nâng cao hồ sơ. | - Cấu trúc lại code sử dụng OOP.<br>- Bổ sung try-except xử lý nhập điểm quá giới hạn.<br>- Deploy ứng dụng công khai.<br>- Chuẩn bị slide báo cáo. |
| **Quang Minh** | **Hệ thống Dự đoán Giá trị và Tư vấn Máy ảnh Số**<br>*(Dataset: petrabayupangestu/camaera-digital-specification)* | - Tải dataset thông số kỹ thuật máy ảnh.<br>- Khảo sát các thuộc tính: Hãng, độ phân giải, loại cảm biến, giá bán.<br>- Setup thư mục code. | - Vẽ 8 biểu đồ (Bar phân bố hãng máy, Boxplot giá theo loại cảm biến, Scatter megapixel vs giá...).<br>- Rút ra insight định giá máy ảnh.<br>- Train model Linear Regression đạt R2 >= 80%. | - Xây dựng ứng dụng Streamlit định giá máy ảnh cũ.<br>- Tạo form nhập thông số máy để dự đoán giá trị.<br>- Tích hợp Gemini đóng vai **Chuyên gia máy ảnh** tư vấn chọn mua thiết bị phù hợp. | - Đóng gói code theo chuẩn OOP.<br>- Xử lý ngoại lệ dữ liệu nhập.<br>- Deploy app lên Streamlit Cloud.<br>- Chuẩn bị demo thuyết trình. |
| **Khánh Lâm** | *Chưa chốt ý tưởng mới*<br>**Gợi ý: Phân tích phim ảnh & Dự đoán doanh thu phòng vé**<br>*(Dataset: tmdb-movie-metadata)* | - Chọn ý tưởng (Doanh thu phim) & tải dataset.<br>- Nộp Form đăng ký ý tưởng.<br>- Khảo sát dữ liệu phim ảnh. | - Vẽ 8 biểu đồ điện ảnh (Scatter budget vs revenue, Bar top đạo diễn...).<br>- Phân tích xu hướng thị hiếu khán giả.<br>- Train model hồi quy dự đoán doanh thu phim (R2 >= 80%). | - Thiết kế Web giới thiệu và dự đoán phim Streamlit.<br>- Cho phép nhập thông tin dự án phim (kinh phí, diễn viên...) để xem dự báo.<br>- Tích hợp Gemini đóng vai **Nhà phê bình phim** tóm tắt review. | - Tổ chức mã nguồn dạng OOP sạch sẽ.<br>- Deploy ứng dụng công khai.<br>- Đảm bảo hoạt động ổn định không lỗi API. |
| **Hải Long** | **Khám phá Game Steam & Trợ lý Đánh giá Game Tự động**<br>*(Dataset: nikdavis/steam-store-games)* | - Tải dataset Kaggle.<br>- Lọc sạch dữ liệu các game trống tên/tag.<br>- Tạo danh sách các thể loại game phổ biến để phục vụ gợi ý. | - Vẽ 8 biểu đồ (Bar chart top game, Pie chart tỷ lệ game trả phí, Heatmap tag...).<br>- Xây dựng thuật toán gợi ý game bằng Cosine Similarity dựa trên Tags.<br>- Đạt độ chính xác gợi ý tốt. | - Tạo giao diện Streamlit mang phong cách **Steam Dark Mode**.<br>- Hiển thị kết quả gợi ý.<br>- Tích hợp Gemini đóng vai **Game thủ try-hard** review lầy lội, dụ dỗ mua game. | - Đóng gói thuật toán gợi ý và chatbot thành các Class.<br>- Triển khai ứng dụng lên Streamlit Cloud.<br>- Test tính năng nhập game của người dùng. |
| **Thái Sơn** | **Phân tích xu hướng & Dự đoán Đánh giá Anime TV Shows**<br>*(Dataset: forgetabhi/anime-tv-shows-dataset-2023)* | - Tải dataset Kaggle.<br>- Lọc cột Type chỉ giữ lại định dạng 'TV'.<br>- Khảo sát phân phối của Score và số tập (episodes). | - Vẽ 8 biểu đồ (Line chart score qua các năm, Bar top studio...).<br>- Phân tích xu hướng Anime hot.<br>- Train model hồi quy dự đoán Score dựa trên tập phim và độ nổi tiếng (R2 >= 80%). | - Thiết kế giao diện xếp hạng Anime trực quan.<br>- Tạo bộ lọc tìm kiếm theo thể loại.<br>- Tích hợp Gemini đóng vai **Otaku thông thái** tư vấn và tóm tắt nhanh nội dung Anime. | - Tối ưu hóa tốc độ load dữ liệu Pandas.<br>- Chuyển đổi mã nguồn sang OOP.<br>- Deploy ứng dụng công khai.<br>- Hoàn thiện tài liệu giới thiệu. |
| **Chí Vinh** | **Phân tích và dự đoán xu hướng thất nghiệp toàn cầu**<br>*(Dataset: sazidthe1/global-unemployment-data)* | - Tải dữ liệu thất nghiệp toàn cầu giai đoạn 2014-2024.<br>- Khảo sát dữ liệu thô.<br>- Phân chia nhóm dữ liệu thanh niên (15-24) và người lớn. | - Thiết kế 8 biểu đồ trực quan (Multi-line chart xu hướng theo năm, Box plot so sánh trước/sau Covid, Heatmap tương quan...).<br>- Train model (Linear Regression, Random Forest, XGBoost) dự báo tỷ lệ thất nghiệp (R2 >= 80%). | - Xây dựng giao diện Dashboard phân tích thất nghiệp toàn cầu trên Streamlit.<br>- Tích hợp tính năng dự đoán chênh lệch.<br>- Tích hợp Gemini đóng vai **Chuyên gia lao động vĩ mô** trả lời câu hỏi phân tích. | - Đóng gói toàn bộ mô hình và thuật toán vẽ biểu đồ vào cấu trúc OOP.<br>- Deploy app công khai.<br>- Kiểm tra hoạt động ổn định và bảo mật API Key trong Secrets. |
| **Quốc Tuấn** | **Dự đoán yếu tố ảnh hưởng tới học tập học sinh**<br>*(Dataset: devansodariya/student-performance-data)* | - Tải dataset Kaggle.<br>- Tìm hiểu các cột thông tin gia đình, thói quen sinh hoạt và điểm số.<br>- Thiết lập môi trường dự án. | - Vẽ 8 biểu đồ (Boxplot điểm số theo mức rượu bia, Scatter thời gian học vs điểm...).<br>- Trích xuất insight giáo dục.<br>- Train model Classification dự đoán lực học đạt Accuracy >= 80%. | - Thiết kế web phân tích các chỉ số học sinh.<br>- Form nhập liệu dự đoán kết quả học tập dự kiến.<br>- Tích hợp Gemini đóng vai **Cố vấn học đường** khuyên bảo thói quen tốt. | - Đóng gói các tính năng vào Class quản lý học sinh.<br>- Deploy sản phẩm lên Streamlit Cloud.<br>- Kiểm tra tương thích giao diện trên thiết bị di động. |
| **Phúc Nguyên** | **Hệ thống phân tích vĩ mô đầu tư dài hạn VN**<br>*(Dataset: VN Economic, US-VN Bond Yield, Yahoo Finance...)* | - Thu thập dữ liệu vĩ mô (lợi suất trái phiếu, lạm phát, DXY, P/E thị trường, tỷ giá...) từ các nguồn chỉ định.<br>- Tạo notebook và chuẩn bị dữ liệu. | - Vẽ 8 biểu đồ vĩ mô (Line chart lợi suất trái phiếu, Bar chart PE...).<br>- Tính toán Chỉ số sức mạnh đầu tư từ các thuộc tính đầu vào.<br>- Huấn luyện mô hình Random Forest dự đoán điểm mua/bán (R2/Accuracy >= 80%). | - Xây dựng trang Dashboard vĩ mô thị trường Việt Nam bằng Streamlit.<br>- Giao diện nhập thông số vĩ mô để đánh giá độ an toàn đầu tư.<br>- Tích hợp Gemini đóng vai **Cố vấn tài chính vĩ mô**. | - Áp dụng OOP để phân tách module nạp dữ liệu, dự báo và AI tư vấn.<br>- Cài đặt Secrets bảo mật API Key trên Streamlit Cloud.<br>- Deploy ứng dụng công khai. |

---

## 3. HƯỚNG DẪN CÁC BƯỚC TRIỂN KHAI VÀ DEPLOY (DÀNH CHO GIÁO VIÊN & HỌC VIÊN)

### Bước 1: Quản lý API Key Gemini an toàn
Tuyệt đối không hardcode API Key trực tiếp vào code Python khi đẩy lên GitHub. Hướng dẫn học viên dùng file `.env` ở local:
```python
# Ở local: tạo file .env
GEMINI_API_KEY = "AIzaSy..."

# Trong code app.py sử dụng thư viện python-dotenv
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
```

### Bước 2: Chuẩn bị file đóng gói môi trường (`requirements.txt`)
Để Streamlit Cloud cài đặt thư viện cần thiết, ở Tuần 4 học viên cần tạo file `requirements.txt` trong thư mục gốc của dự án:
```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
google-generativeai
python-dotenv
```

### Bước 3: Deploy lên Streamlit Community Cloud
1. Đẩy toàn bộ mã nguồn lên một Repository công khai (Public) trên GitHub (bao gồm file `app.py`, model đã lưu dưới dạng `.pkl` hoặc `.joblib`, bộ dữ liệu CSV, và file `requirements.txt`).
2. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
3. Nhấn **New app**, chọn Repository, Branch (thường là `main`), và Main file path (ví dụ: `app.py`).
4. **Cấu hình API Key trên Cloud**: Vào mục **Advanced settings** -> Điền khóa bí mật vào phần **Secrets** để ứng dụng chạy trên internet lấy được API Key mà không bị lộ:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
5. Nhấn **Deploy** và đợi 2-3 phút để nhận đường link ứng dụng public!
