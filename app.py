from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import analyze_pdf, load_metadata, normalize_label, predict_sentence

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "financial_sentiment_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy file models/financial_sentiment_pipeline.joblib. "
            "Hãy chạy notebook hoặc scripts/train_export_model.py để xuất model trước khi deploy."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_app_metadata() -> dict:
    return load_metadata(METADATA_PATH)


def show_metadata(metadata: dict) -> None:
    if not metadata:
        st.warning("Chưa có model_metadata.json hoặc file chưa đọc được.")
        return
    cols = st.columns(3)
    cols[0].metric("Validation Macro-F1", metadata.get("validation_macro_f1", "N/A"))
    cols[1].metric("Test Accuracy", metadata.get("test_accuracy", "N/A"))
    cols[2].metric("Test Macro-F1", metadata.get("test_macro_f1", "N/A"))
    st.json(metadata, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="📊", layout="wide")
    st.title("Financial Sentiment Analysis")
    st.caption("Phân tích cảm xúc câu tài chính và báo cáo PDF bằng TF-IDF + Linear SVM")
    st.info(
        "Kết quả chỉ phản ánh sắc thái ngôn ngữ trong văn bản và không phải là khuyến nghị đầu tư "
        "hoặc đánh giá toàn diện sức khỏe tài chính của doanh nghiệp."
    )

    metadata = load_app_metadata()
    try:
        model = load_model()
        model_ready = True
    except Exception as exc:
        model = None
        model_ready = False
        st.error(str(exc))
        st.markdown(
            "**Cách khắc phục:** chạy `python scripts/train_export_model.py` trong môi trường có Internet "
            "hoặc mở notebook và chạy phần `Xuất mô hình để triển khai Streamlit`, sau đó commit file joblib lên GitHub."
        )

    tab_text, tab_pdf, tab_info = st.tabs(["Dự đoán một câu", "Phân tích PDF", "Thông tin mô hình"])

    with tab_text:
        st.subheader("Dự đoán cảm xúc một câu tài chính")
        examples = {
            "Positive": "Revenue increased by 20 percent compared with the previous year.",
            "Neutral": "The company was established in 1995.",
            "Negative": "Operating profit declined because of higher raw material costs.",
        }
        selected = st.selectbox("Chọn câu ví dụ", ["Tự nhập"] + list(examples.keys()))
        default_text = examples.get(selected, "")
        sentence = st.text_area("Nhập câu tiếng Anh trong lĩnh vực tài chính", value=default_text, height=120)
        if st.button("Phân tích câu", type="primary", disabled=not model_ready):
            try:
                result = predict_sentence(model, sentence)
                st.success(f"Nhãn dự đoán: **{result['predicted_label']}**")
                if result["scores"]:
                    score_df = pd.DataFrame(
                        [{"label": k, "decision_score": v} for k, v in result["scores"].items()]
                    ).sort_values("decision_score", ascending=False)
                    st.caption("Decision score không phải là xác suất.")
                    st.dataframe(score_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(str(exc))

    with tab_pdf:
        st.subheader("Phân tích một file PDF")
        uploaded = st.file_uploader("Tải lên PDF có lớp văn bản", type=["pdf"])
        if uploaded is not None:
            st.write(f"File: `{uploaded.name}` - {uploaded.size / 1024 / 1024:.2f} MB")
        if st.button("Phân tích PDF", type="primary", disabled=(not model_ready or uploaded is None)):
            try:
                with st.spinner("Đang trích xuất văn bản và dự đoán..."):
                    df, meta = analyze_pdf(model, uploaded.getvalue())
                st.success(f"Đã phân tích {meta['sentence_count']:,} câu từ {meta['page_count']} trang bằng {meta['engine']}.")
                summary = df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
                summary["ratio_percent"] = (summary["count"] / summary["count"].sum() * 100).round(2)
                left, right = st.columns([1, 1])
                with left:
                    st.dataframe(summary, width="stretch", hide_index=True)
                with right:
                    st.bar_chart(summary.set_index("label")["count"])
                st.subheader("Kết quả từng câu")
                st.dataframe(df.head(200), width="stretch", hide_index=True)
                st.download_button(
                    "Tải CSV kết quả",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="financial_sentiment_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(str(exc))

    with tab_info:
        st.subheader("Thông tin mô hình")
        show_metadata(metadata)
        st.markdown(
            """
            **Ghi chú triển khai**
            - Ứng dụng tải model một lần bằng `st.cache_resource`.
            - Không huấn luyện lại model khi người dùng truy cập.
            - PDF scan chưa được OCR trong phiên bản đầu.
            - Linear SVM chưa calibration nên app hiển thị decision score, không hiển thị xác suất.
            """
        )


if __name__ == "__main__":
    main()
