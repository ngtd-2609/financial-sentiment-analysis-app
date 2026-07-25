# Models

Thư mục này chứa model deploy và metadata.

- `financial_sentiment_pipeline.joblib` — Pipeline TF-IDF +
  Logistic Regression, được huấn luyện trên tập train và validation
  gồm tổng cộng 2.930 câu.
- `model_metadata.json` — Thông tin model, phiên bản, và kết quả đánh giá.

Để tạo lại model:

```bash
python scripts/train_export_model.py
```

hoặc mở `notebook/Financial_Sentiment_Analysis_Fixed_collab.ipynb` và chạy toàn bộ notebook.
