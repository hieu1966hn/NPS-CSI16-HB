Tên dự án: Hệ thống phân tích vĩ mô đầu tư dài hạn thị trường Việt Nam
Link Dataset (Kaggle):
https://www.investing.com
https://fred.stlouisfed.org
https://www.gso.gov.vn
https://sbv.gov.vn
https://vietstock.vn
https://fiintrade.vn
https://wichart.vn
https://finance.yahoo.com
https://github.com/thinh-vu/vnstock

Vấn đề cần giải quyết: Hỗ trợ nhà đầu tư đánh giá xu hướng thị trường, phân tích kinh tế VN = chỉ số sức mạnh đầu tư xác định thời điểm mua bán nắm giữ thị trường cổ phiếu dựa vào các biến vĩ mô như lợi suất trái phiếu chính phủ Mỹ Việt nam, lạm phát, dxy, PE toàn thị trường, tỷ giá và lãi suất việt nam
Kiến thức áp dụng:
CS Basic: Cấu trúc điều khiển, vòng lặp, hàm, xử lý chuỗi.
CS Advanced: Lập trình hướng đối tượng (OOP) để quản lý mô hình, xử lý ngoại lệ khi gọi API.
CS Intensive: Pandas (tiền xử lý dữ liệu), Scikit-learn (huấn luyện mô hình Random Forest), Matplotlib/Seaborn (trực quan hóa), Streamlit (xây dựng Web App), Generative AI (tích hợp API Gemini).

## Support khó khăn về dữ liệu

Bạn đang cần 2 nhóm dữ liệu từ 2015 đến 2026:

- P/E của VNINDEX.
- Lãi suất huy động VND kỳ hạn 12 tháng của các ngân hàng lớn tại Việt Nam.

Đây là phần dữ liệu tài chính khá khó vì nhiều nguồn chỉ hiển thị dữ liệu hiện tại, không cho tải lịch sử đầy đủ miễn phí. Vì vậy, nên xử lý theo hướng sau:

### 1. Chốt lại phạm vi dữ liệu trước khi tìm

Không nên tìm ngay dữ liệu theo ngày vì sẽ rất mất thời gian. Với dự án học máy/Streamlit ở mức lớp học, em nên bắt đầu bằng dữ liệu theo năm hoặc theo tháng.

Gợi ý dùng bản đầu tiên:

- Mốc thời gian: 2015-2026.
- Tần suất: theo năm.
- Giá trị năm 2026: ghi rõ là dữ liệu tạm tính/cập nhật đến thời điểm thu thập, vì năm 2026 chưa kết thúc.
- Với lãi suất ngân hàng: chọn 4 ngân hàng lớn, dễ giải thích: Vietcombank, BIDV, VietinBank, Agribank.
- Nếu thiếu đủ 4 ngân hàng ở một năm nào đó, có thể dùng trung bình nhóm ngân hàng quốc doanh hoặc dùng Vietcombank làm đại diện, nhưng phải ghi rõ trong phần mô tả dữ liệu.

### 2. Cách lấy P/E VNINDEX

Nguồn nên kiểm tra:

- CafeF - mục Dữ liệu thị trường, phần Định giá VN-Index: https://cafef.vn/du-lieu.chn
- VietstockFinance: https://finance.vietstock.vn/
- Wichart: https://wichart.vn/
- FiinTrade: https://fiintrade.vn/

Lưu ý cho em:

- P/E VNINDEX thường không dễ lấy qua API miễn phí như giá VNINDEX.
- Yahoo Finance, Investing.com, vnstock thường hỗ trợ giá chỉ số tốt hơn là P/E toàn thị trường.
- Nếu nguồn không cho tải lịch sử P/E, em có thể thu thập thủ công theo năm từ biểu đồ/báo cáo thị trường, hoặc dùng Wichart/FiinTrade nếu có tài khoản.
- Không nên trộn nhiều cách tính P/E khác nhau mà không ghi chú. P/E trailing, forward P/E và P/E theo ETF đại diện là 3 loại khác nhau.

Phương án đơn giản cho bản MVP:

Tạo file `macro_dataset.csv` với cột:

```csv
year,vnindex_pe,source_pe,note
2015,,,
2016,,,
...
2026,,,"2026 là dữ liệu tạm tính"
```

Sau đó điền dần bằng dữ liệu từ CafeF/Wichart/Vietstock/FiinTrade. Nếu một nguồn có đủ lịch sử thì ưu tiên dùng một nguồn duy nhất.

### 3. Cách lấy lãi suất huy động 12 tháng

Nguồn nên kiểm tra:

- Trang lãi suất hiện tại của Vietcombank: https://www.vietcombank.com.vn/vi-VN/KHCN/Cong-cu-Tien-ich/KHCN---Lai-suat
- Trang lãi suất hiện tại của VietinBank: https://www.vietinbank.vn/en/lai-suat-khcn
- CafeF - bảng lãi suất ngân hàng: https://cafef.vn/du-lieu/lai-suat-ngan-hang.chn
- World Bank - Deposit interest rate của Việt Nam: https://data.worldbank.org/indicator/FR.INR.DPST?locations=VN
- SBV - mục lãi suất và thông tin hoạt động ngân hàng: https://sbv.gov.vn/webcenter/portal/vi/menu/rm/ls

Lưu ý quan trọng:

- Website ngân hàng thường chỉ hiển thị lãi suất hiện tại, không có lịch sử 2015-2026 đầy đủ.
- Nếu cần chính xác từng ngân hàng theo từng năm, em nên tìm thêm các bài báo theo từng tháng/năm, ví dụ: "lãi suất Vietcombank kỳ hạn 12 tháng tháng 12/2021".
- Nếu mục tiêu chính là mô hình vĩ mô, em có thể dùng chỉ số "Deposit interest rate (%) - Viet Nam" của World Bank làm biến đại diện. Biến này không phải riêng kỳ hạn 12 tháng và không phải riêng Big 4, nhưng dễ lấy, thống nhất và có độ tin cậy cao.
- Dữ liệu World Bank hiện thường chỉ có đến các năm đã công bố chính thức, nên 2024-2026 cần bổ sung bằng CafeF, SBV, báo cáo công ty chứng khoán hoặc bảng lãi suất hiện tại.

Gợi ý cấu trúc dữ liệu:

```csv
year,vcb_12m,bidv_12m,ctg_12m,agribank_12m,big4_avg_12m,wb_deposit_rate,note
2015,,,,,,,
2016,,,,,,,
...
2026,,,,,,,"2026 là dữ liệu tạm tính/cập nhật đến ngày lấy dữ liệu"
```

### 4. Cách làm thực tế để không bị kẹt

Lộ trình nên làm:

1. Làm bản dữ liệu theo năm trước, chưa cần theo tháng.
2. Dùng một nguồn chính cho P/E VNINDEX, ví dụ Wichart/CafeF/Vietstock.
3. Dùng World Bank Deposit Interest Rate làm biến lãi suất đại diện trong bản MVP.
4. Nếu còn thời gian, nâng cấp bằng cách thêm lãi suất 12 tháng của Big 4 từ các bài báo/bảng lãi suất lưu trữ.
5. Trong báo cáo, viết rõ: "Do hạn chế về dữ liệu lịch sử công khai, dự án sử dụng lãi suất tiền gửi bình quân/đại diện thay cho lãi suất 12 tháng từng ngân hàng."

### 5. Gợi ý cho phần trình bày với thầy/cô

Em có thể nói:

> Dữ liệu P/E VNINDEX và lãi suất huy động 12 tháng theo từng ngân hàng không có sẵn đầy đủ qua API miễn phí. Vì vậy, em sẽ làm bản MVP với dữ liệu theo năm, dùng P/E VNINDEX từ một nguồn thống nhất như CafeF/Wichart/Vietstock, và dùng Deposit Interest Rate của World Bank hoặc trung bình lãi suất Big 4 làm biến đại diện. Khi có thêm thời gian, em sẽ bổ sung dữ liệu theo tháng và đối chiếu với báo cáo/ngân hàng.

Hướng này hợp lý hơn là cố tìm một bộ dữ liệu hoàn hảo ngay từ đầu.
