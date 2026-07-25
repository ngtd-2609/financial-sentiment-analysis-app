# Phân công công việc

| Thành viên | Vai trò | Công việc chính | Tỷ trọng |
|---|---|---|---:|
| Nguyễn Tùng Dương | Dữ liệu, tiền xử lý và báo cáo | Thu thập Financial PhraseBank, kiểm tra nhãn, chuẩn hóa văn bản, EDA, loại trùng, chia dữ liệu, viết phần dữ liệu và đồng bộ báo cáo | 34% |
| Phạm Thế Duy | Mô hình và đánh giá | Xây dựng baseline, huấn luyện, validation, đánh giá test, confusion matrix, phân tích lỗi, xuất Pipeline triển khai và kiểm tra tương thích model | 33% |
| Ngô Quang Thiện | Demo, PDF và trình bày | Thiết kế Streamlit app, chức năng nhập câu, chức năng PDF, trực quan hóa kết quả, slide, README deploy và kiểm tra giao diện | 33% |

Tổng tỷ trọng: **100%**.

## Chi tiết công việc

### Nguyễn Tùng Dương - 34%

- Tìm hiểu bài toán sentiment analysis trong miền tài chính.
- Thu thập và kiểm tra Financial PhraseBank.
- Kiểm tra nhãn, câu rỗng, nhãn thiếu và câu trùng.
- Chuẩn hóa Unicode, khoảng trắng và dữ liệu đầu vào cho TF-IDF.
- Thực hiện EDA và nhận xét phân bố nhãn.
- Chia train/validation/test và kiểm tra data leakage.
- Viết và đồng bộ phần dữ liệu, phương pháp trong báo cáo.

### Phạm Thế Duy - 33%

- Xây dựng Dummy, Naive Bayes, Logistic Regression và Linear SVM.
- Đánh giá validation bằng Accuracy, Macro Precision, Macro Recall, Macro-F1 và Weighted-F1.
- Chọn mô hình bằng validation Macro-F1.
- Đánh giá test sau khi khóa mô hình.
- Tạo confusion matrix và phân tích lỗi.
- Xuất Pipeline TF-IDF + Linear SVM để triển khai.
- Kiểm tra model load lại bằng joblib.

### Ngô Quang Thiện - 33%

- Thiết kế ứng dụng Streamlit.
- Xây dựng chức năng dự đoán một câu tài chính.
- Xây dựng chức năng tải và phân tích PDF.
- Tổng hợp nhãn và trực quan hóa kết quả.
- Viết README hướng dẫn GitHub và Streamlit Cloud.
- Cập nhật slide theo hướng ứng dụng.
- Chuẩn bị câu hỏi bảo vệ và kiểm tra bản cuối.
