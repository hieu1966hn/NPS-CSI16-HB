# Tổng hợp chấm bổ sung

Nguồn đối chiếu:

- Bài nộp học viên trong thư mục `SS5_CP`.
- Quy ước chấm giữ nguyên theo rubric buổi 5:
  - Trắc nghiệm: 10 câu, mỗi câu đúng 0.5 điểm, tối đa 5 điểm.
  - Tự luận: tối đa 5 điểm theo 4 tiêu chí: chia train/validation-test, tiền xử lý, CNN + FCN, accuracy đạt yêu cầu.

## Bảng điểm tổng hợp

| Học viên | Trắc nghiệm | Tự luận | Tổng | Minh chứng nhanh |
|---|---:|---:|---:|---|
| Bảo An | 4.0 (8/10) | 3.75/5 | 7.75/10 | Có chia train-val/test; có tiền xử lý; có CNN + FCN; không thấy log accuracy rõ ràng |
| Vũ Quốc Tuấn | 0.0 (0/10) | 5.00/5 | 5.00/10 | Có chia train-val/test; có tiền xử lý; có CNN + FCN; best train = 0.878, val = 0.847 |

## Nhận xét từng học viên

### Bảo An

- Điểm: Trắc nghiệm 4.0/5 (8/10), Tự luận 3.75/5, Tổng 7.75/10.
- File đã đối chiếu: `Trắc nghiệm.md`, `ss5-checkpoint1-csi16.ipynb`.
- Điểm mạnh học viên: Phần trắc nghiệm làm khá tốt, nắm chắc phần lớn kiến thức lý thuyết của buổi học. Bài thực hành có chia dữ liệu train và validation/test, đúng hướng theo yêu cầu đề bài. Học viên có thực hiện tiền xử lý ảnh trước khi đưa vào mô hình.
- Điểm cần cải thiện: Không thấy log huấn luyện hoặc chỉ số accuracy rõ ràng trong bài nộp nên chưa đủ cơ sở cộng điểm tiêu chí độ chính xác.
- Lời khuyên: Cần bổ sung hoặc lưu lại output huấn luyện đầy đủ, đồng thời thử thêm số epoch, batch size, image size hoặc data augmentation để đẩy độ chính xác lên ngưỡng yêu cầu.

### Vũ Quốc Tuấn

- Điểm: Trắc nghiệm 0.0/5 (0/10), Tự luận 5.00/5, Tổng 5.00/10.
- File đã đối chiếu: `[CSI - Checkpoints] Buổi 5_Bài làm.docx`, `csi-checkpoint-1.ipynb`.
- Điểm mạnh học viên: Bài thực hành có chia dữ liệu train và validation/test, đúng hướng theo yêu cầu đề bài. Học viên có thực hiện tiền xử lý ảnh trước khi đưa vào mô hình. Mô hình đã có cấu trúc CNN kèm khối phân loại FCN, và kết quả huấn luyện đạt yêu cầu với train = 0.878, val = 0.847.
- Điểm cần cải thiện: Không thấy phần trả lời trắc nghiệm rõ ràng trong bài nộp nên phần này tạm tính 0 điểm.
- Lời khuyên: Cần bổ sung rõ phần trắc nghiệm vào bài nộp ở các lần sau. Phần thực hành đang làm tốt, nên tiếp tục phát huy sự chỉn chu khi xây dựng và huấn luyện mô hình.

## Thống kê câu sai trắc nghiệm

### Theo từng học viên

- Bảo An: sai Câu 4 (A -> D), Câu 8 (C -> A).
- Vũ Quốc Tuấn: không thấy phần trả lời trắc nghiệm rõ ràng trong bài nộp.

### Theo từng câu

- Câu 1: sai 0 bạn.
- Câu 2: sai 0 bạn.
- Câu 3: sai 0 bạn.
- Câu 4: sai 1 bạn. Bảo An.
- Câu 5: sai 0 bạn.
- Câu 6: sai 0 bạn.
- Câu 7: sai 0 bạn.
- Câu 8: sai 1 bạn. Bảo An.
- Câu 9: sai 0 bạn.
- Câu 10: sai 0 bạn.
