# Financial Sentiment Analysis Application

Ứng dụng NLP phân loại cảm xúc câu tài chính thành ba nhãn: **Negative**, **Neutral** và **Positive**. Bản deploy sử dụng pipeline **TF-IDF + Logistic Regression**, đồng bộ với kết quả tốt nhất từ notebook.

## Tác giả

| Họ tên | GitHub |
|---|---|
| Nguyễn Tùng Dương | [@ngtd-2609](https://github.com/ngtd-2609) |
| Phạm Thế Duy | [@duyphamdevx](https://github.com/duyphamdevx) |
| Ngô Quang Thiện | [@thyelmot](https://github.com/thyelmot) |

> Kết quả chỉ phản ánh sắc thái ngôn ngữ trong văn bản và không phải khuyến nghị đầu tư hoặc đánh giá toàn diện sức khỏe tài chính của doanh nghiệp.

## Demo

- Link Streamlit: `https://financial-sentiment-analysis-ntd.streamlit.app/`

![Giao diện dự đoán câu](assets/app_preview.png)

## Kết quả mô hình Logistic Regression

| Chỉ số | Giá trị |
|---|---:|
| Validation Macro-F1 | 0.8058 |
| Test Accuracy | 0.8571 |
| Test Macro-F1 | 0.8100 |
| Test Macro Precision | 0.8487 |
| Test Macro Recall | 0.7842 |
| Test Weighted-F1 | 0.8545 |

Các chỉ số trên được lấy từ kết quả đánh giá mô hình Logistic Regression
trong notebook của dự án.

## Kiến trúc

```text
Financial PhraseBank
    -> kiểm tra và tiền xử lý
    -> TF-IDF unigram/bigram
    -> Logistic Regression
    -> dự đoán câu tài chính hoặc câu trích từ PDF
```

Ứng dụng Streamlit không huấn luyện lại mô hình khi chạy. App chỉ tải file `models/financial_sentiment_pipeline.joblib` đã được xuất từ notebook hoặc script huấn luyện.

## Cấu trúc repository

```text
financial-sentiment-analysis-app/
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── models/
│   ├── financial_sentiment_pipeline.joblib   # tạo sau khi chạy notebook/script
│   └── model_metadata.json
├── notebook/
│   └── Financial_Sentiment_Analysis_Fixed_collab.ipynb
├── assets/
├── sample_data/
├── docs/
├── tests/
├── scripts/
│   └── train_export_model.py
└── .streamlit/
    └── config.toml
```

## Cài đặt local

```bash
git clone https://github.com/ngtd-2609/financial-sentiment-analysis-app.git
cd financial-sentiment-analysis-app
python -m venv .venv
```

Yêu cầu Python 3.12+ (đã test với Python 3.13.5).

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## Tạo model deploy

Trong môi trường có Internet, chạy:

```bash
python scripts/train_export_model.py
```

Lệnh này tải Financial PhraseBank, huấn luyện các baseline, chọn mô hình bằng validation Macro-F1 và lưu:

```text
models/financial_sentiment_pipeline.joblib
models/model_metadata.json
```

Có thể thay bằng cách mở notebook và chạy phần **Xuất mô hình để triển khai Streamlit**.

## Chạy ứng dụng

```bash
streamlit run app.py
```

Mở địa chỉ local thường là:

```text
http://localhost:8501
```

## Deploy Streamlit Community Cloud

1. Push repository lên GitHub.
2. Đăng nhập Streamlit Community Cloud.
3. Chọn repository.
4. Chọn branch `main`.
5. Main file path: `app.py`.
6. Deploy.

Lệnh GitHub cơ bản:

```bash
git init
git add .
git commit -m "Initial financial sentiment application"
git branch -M main
git remote add origin https://github.com/ngtd-2609/financial-sentiment-analysis-app.git
git push -u origin main
```

## Cách sử dụng app

### Dự đoán một câu

Nhập một câu tài chính tiếng Anh, ví dụ:

```text
Revenue increased by 20 percent compared with the previous year.
```

Ứng dụng trả về nhãn dự đoán và xác suất của từng lớp. Các xác suất được lấy trực tiếp từ phương thức `predict_proba` của mô hình Logistic Regression.

### Phân tích PDF

Tải lên một file PDF có lớp văn bản. Ứng dụng sẽ trích xuất câu, dự đoán nhãn từng câu, tổng hợp tỷ lệ nhãn và cho tải CSV kết quả.

PDF scan chưa được OCR trong phiên bản đầu.

## Chạy kiểm thử

```bash
pytest
```

Nếu chưa có file model joblib, các test liên quan đến model sẽ được skip; các test metadata và xử lý câu vẫn chạy.

## Hạn chế

- Mô hình huấn luyện trên câu tài chính tiếng Anh, chưa tối ưu cho tiếng Việt.
- Logistic Regression cung cấp xác suất dự đoán, tuy nhiên xác suất này
  không nên được xem là mức bảo đảm tuyệt đối về tính chính xác.
- Biểu diễn TF-IDF còn hạn chế trong việc nắm bắt ngữ cảnh dài,
  hàm ý và các quan hệ ngữ nghĩa phức tạp.
- PDF scan cần OCR, chưa hỗ trợ tự động trong phiên bản đầu.
- Có domain shift giữa Financial PhraseBank và báo cáo thường niên.
- Kết quả không phải khuyến nghị đầu tư.

## Thành viên

| Thành viên | Công việc | Tỷ trọng |
|---|---|---:|
| Nguyễn Tùng Dương | Dữ liệu, tiền xử lý, EDA và báo cáo | 33% |
| Phạm Thế Duy | Xây dựng baseline, huấn luyện, lựa chọn mô hình, đánh giá test, confusion matrix và phân tích lỗi | 33% |
| Ngô Quang Thiện | Demo ứng dụng, PDF, slide và kiểm tra cuối | 33% |

## Tài liệu

- Báo cáo: `docs/Bao_Cao_NLP_Theo_Huong_Ung_Dung.docx`
- Slide: `docs/Slide_NLP_Theo_Huong_Ung_Dung.pptx`
- Notebook: `notebook/Financial_Sentiment_Analysis_Fixed_collab.ipynb`
- Câu hỏi bảo vệ: `docs/Cau_Hoi_Bao_Ve.docx`

## Ghi chú bản quyền dữ liệu

Mã nguồn do nhóm xây dựng có thể dùng theo MIT License. Dataset Financial PhraseBank và các báo cáo doanh nghiệp có điều khoản sử dụng riêng; repository không tuyên bố sở hữu các nguồn dữ liệu này.

## Hướng dẫn đồng tác giả fork repo

Các thành viên [@duyphamdevx](https://github.com/duyphamdevx) và [@thyelmot](https://github.com/thyelmot) có thể thêm repo này vào GitHub cá nhân bằng cách:

1. Vào trang repo: https://github.com/ngtd-2609/financial-sentiment-analysis-app
2. Bấm nút **Fork** (góc trên phải)
3. Chọn tài khoản cá nhân → bấm **Create fork**
4. Repo sẽ xuất hiện tại `https://github.com/<username>/financial-sentiment-analysis-app`
