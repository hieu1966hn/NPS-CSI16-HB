Vấn đề cần giải quyết : Xác định các quốc gia có độ vênh lớn nhất giữa tỷ lệ thất nghiệp của thanh niên và người trưởng thành. Sự bất ổn này có tăng mạnh sau khi dịch COVID-19 (2020- 2022)



PHIẾU ĐĂNG KÝ Ý TƯỞNG DỰ ÁN CUỐI KHÓA
Họ và tên
Nguyễn Chí Vinh

Tên dự án / Chủ đề

Phân tích và dự đoán xu hướng thất nghiệp toàn cầu giai đoạn 2014–2024: Tác động của COVID-19 đến khoảng cách thất nghiệp giữa thanh niên và người trưởng thành

Link Dataset (Kaggle)

Global Unemployment Data Dataset (https://www.kaggle.com/datasets/sazidthe1/global-unemployment-data)

Vấn đề cần giải quyết

Thất nghiệp là một trong những chỉ số kinh tế quan trọng phản ánh sức khỏe của thị trường lao động. Tuy nhiên, tỷ lệ thất nghiệp giữa các nhóm tuổi thường không đồng đều, đặc biệt là nhóm thanh niên (15–24 tuổi) thường chịu ảnh hưởng nặng nề hơn khi xảy ra khủng hoảng kinh tế.

Dự án tập trung giải quyết các câu hỏi:

Quốc gia nào có độ chênh lệch lớn nhất giữa tỷ lệ thất nghiệp thanh niên và người trưởng thành?
Khoảng cách thất nghiệp này thay đổi như thế nào trong giai đoạn 2014–2024?
Đại dịch COVID-19 (2020–2022) có làm gia tăng sự bất ổn trên thị trường lao động hay không?
Có thể dự đoán xu hướng thất nghiệp trong các năm tiếp theo dựa trên dữ liệu lịch sử hay không?
Mục tiêu dự án
Mục tiêu phân tích
Phân tích xu hướng thất nghiệp toàn cầu từ 2014–2024.
Xác định Top quốc gia có khoảng cách thất nghiệp thanh niên – trưởng thành lớn nhất.
So sánh tình hình trước và sau COVID-19.
Tìm ra các khu vực chịu ảnh hưởng mạnh nhất.
Mục tiêu Machine Learning

Xây dựng mô hình dự đoán:

Tỷ lệ thất nghiệp thanh niên.
Tỷ lệ thất nghiệp người trưởng thành.
Mức độ chênh lệch thất nghiệp trong tương lai.
Đầu ra sản phẩm

1. Web Application

Xây dựng bằng Streamlit.

Người dùng có thể:

Chọn quốc gia.
Chọn giai đoạn thời gian.
Xem biểu đồ thất nghiệp.
Xem mức độ ảnh hưởng của COVID-19.
Nhận dự đoán tỷ lệ thất nghiệp trong tương lai.
2. Machine Learning Model
Bài toán

Regression (Dự đoán giá trị liên tục)

Input
Quốc gia
Năm
Khu vực
Tỷ lệ thất nghiệp các năm trước
Output
Tỷ lệ thất nghiệp dự báo
Độ chênh lệch thất nghiệp thanh niên – trưởng thành
Thuật toán dự kiến
Linear Regression
Random Forest Regressor
XGBoost Regressor

So sánh các mô hình để chọn mô hình tốt nhất.

Chỉ số đánh giá
R² Score
MAE
RMSE

Mục tiêu:

Accuracy tương đương ≥ 80%
R² ≥ 0.8
3. Chatbot AI

Tích hợp chatbot trên Web App sử dụng API Gemini/OpenAI.

Ví dụ người dùng có thể hỏi:

"Quốc gia nào có tỷ lệ thất nghiệp thanh niên cao nhất năm 2022?"
"COVID-19 ảnh hưởng đến Việt Nam như thế nào?"


"Dự đoán thất nghiệp của Mỹ năm tới là bao nhiêu?"
"Top 10 quốc gia có khoảng cách thất nghiệp lớn nhất?"

Chatbot sẽ truy vấn dữ liệu đã phân tích và trả lời trực tiếp.

Phân tích dữ liệu và Visualization

Dự án sẽ có tối thiểu 8 dashboard và trên 5 loại biểu đồ khác nhau.

Dashboard 1: Tổng quan thất nghiệp toàn cầu
Line Chart
KPI Cards
Dashboard 2: Xu hướng thất nghiệp theo năm
Multi-Line Chart
Dashboard 3: Top quốc gia thất nghiệp cao nhất
Bar Chart
Dashboard 4: Top quốc gia có độ chênh lệch lớn nhất
Horizontal Bar Chart
Dashboard 5: Trước và sau COVID-19
Box Plot
Violin Plot
Dashboard 6: Phân bố tỷ lệ thất nghiệp
Histogram
Dashboard 7: Tương quan các chỉ số
Heatmap
Dashboard 8: Bản đồ thế giới
Choropleth Map
Insight dự kiến
Insight 1

Nhóm thanh niên có tỷ lệ thất nghiệp cao hơn đáng kể so với người trưởng thành ở hầu hết các quốc gia.

Insight 2

Giai đoạn 2020–2022 xuất hiện mức tăng thất nghiệp đột biến tại nhiều quốc gia do COVID-19.

Insight 3

Một số quốc gia phục hồi nhanh sau đại dịch, trong khi một số quốc gia vẫn duy trì khoảng cách thất nghiệp lớn.

Insight 4

Các nước đang phát triển có xu hướng chịu tác động mạnh hơn về thất nghiệp thanh niên.

Next Actions
Đối với Chính phủ
Tăng cường chương trình hỗ trợ việc làm cho thanh niên.
Đầu tư đào tạo kỹ năng nghề.
Xây dựng chính sách phục hồi lao động sau khủng hoảng.
Đối với doanh nghiệp
Tăng cơ hội thực tập và tuyển dụng sinh viên mới tốt nghiệp.
Đẩy mạnh đào tạo lại (reskilling).
Kiến thức áp dụng
CS Basic
Biến, vòng lặp
Hàm
Điều kiện
List, Dictionary
Xử lý dữ liệu
CS Advanced
OOP
File Handling
Exception Handling
Thuật toán sắp xếp, tìm kiếm
CS Intensive
Pandas
NumPy
Matplotlib
Seaborn
Plotly
Scikit-Learn
Machine Learning
Streamlit
API Integration (Gemini/OpenAI)
