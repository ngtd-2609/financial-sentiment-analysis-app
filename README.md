# Financial Sentiment Analysis Application

Ứng dụng NLP phân loại cảm xúc câu tài chính thành ba nhãn: **Negative**, **Neutral** và **Positive**. Bản deploy sử dụng pipeline **TF-IDF + Linear SVM**, được chọn bằng **validation Macro-F1** trong notebook huấn luyện.

> Kết quả chỉ phản ánh sắc thái ngôn ngữ trong văn bản và không phải khuyến nghị đầu tư hoặc đánh giá toàn diện sức khỏe tài chính của doanh nghiệp.

## Demo

- Link Streamlit: `[STREAMLIT_APP_URL]`

![Giao diện dự đoán câu](assets/app_preview.png)

## Kết quả mô hình

| Metric | Giá trị |
|---|---:|
| Validation Macro-F1 | 0.7971 |
| Test Accuracy | 0.8649 |
| Test Macro-F1 | 0.8115 |

## Kiến trúc

```text
Financial PhraseBank
    -> kiểm tra và tiền xử lý
    -> TF-IDF unigram/bigram
    -> Linear SVM
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
│   └── Financial_Sentiment_Analysis_Application.ipynb
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
.venv\Scriptsctivate
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

Ứng dụng trả về nhãn dự đoán và decision score nếu mô hình hỗ trợ. Với Linear SVM, decision score **không phải xác suất**.

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
- Linear SVM không cung cấp xác suất chuẩn nếu chưa calibration.
- PDF scan cần OCR, chưa hỗ trợ tự động trong phiên bản đầu.
- Có domain shift giữa Financial PhraseBank và báo cáo thường niên.
- Kết quả không phải khuyến nghị đầu tư.

## Thành viên

| Thành viên | Công việc | Tỷ trọng |
|---|---|---:|
| Nguyễn Tùng Dương | Dữ liệu, tiền xử lý, EDA và báo cáo | 34% |
| Phạm Thế Duy | Mô hình, đánh giá, phân tích lỗi và xuất Pipeline triển khai | 33% |
| Ngô Quang Thiện | Demo ứng dụng, PDF, slide và kiểm tra cuối | 33% |

## Tài liệu

- Báo cáo: `docs/Bao_Cao_NLP_Theo_Huong_Ung_Dung.docx`
- Slide: `docs/Slide_NLP_Theo_Huong_Ung_Dung.pptx`
- Notebook: `notebook/Financial_Sentiment_Analysis_Application.ipynb`
- Câu hỏi bảo vệ: `docs/Cau_Hoi_Bao_Ve.docx`

## Ghi chú bản quyền dữ liệu

Mã nguồn do nhóm xây dựng có thể dùng theo MIT License. Dataset Financial PhraseBank và các báo cáo doanh nghiệp có điều khoản sử dụng riêng; repository không tuyên bố sở hữu các nguồn dữ liệu này.
