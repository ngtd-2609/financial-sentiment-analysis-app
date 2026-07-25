from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import (
    analyze_pdf,
    load_metadata,
    normalize_label,
    predict_sentence,
    softmax_confidence,
)

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
        if st.button("Phan tich cau", type="primary", disabled=not model_ready):
            try:
                result = predict_sentence(model, sentence)
                label = result["predicted_label"]
                emoji_map = {"Positive": "Tich cuc", "Neutral": "Trung lap", "Negative": "Tieu cuc"}
                display = emoji_map.get(label, label)
                color_map = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
                color = color_map.get(label, "blue")
                st.markdown(
                    f"<h3 style='color:{color}'>Ket qua: {display}</h3>",
                    unsafe_allow_html=True,
                )
                if result["scores"]:
                    conf = softmax_confidence(result["scores"])
                    conf_df = pd.DataFrame(
                        [{"Cam xuc": emoji_map.get(k, k), "Do tin cay (%)": v}
                         for k, v in sorted(conf.items(), key=lambda x: -x[1])]
                    )
                    st.caption(
                        "Do tin cay duoc tinh tu diem phan quyet cua mo hinh (Linear SVM), "
                        "khong phai xac suat chinh thuc. Gia tri cang cao thi mo hinh cang chac chan ve nhan do."
                    )
                    st.dataframe(conf_df, width="stretch", hide_index=True)

                    with st.expander("Chi tiet ky thuat (danh cho nguoi trong nganh)"):
                        st.caption(
                            "Decision score la diem phan quyet cua LinearSVC truoc khi phan nguong. "
                            "Gia tri duong va lon hon thi nhan do duoc mo hinh uu tien hon. "
                            "Day la diem tho, CHUA qua sigmoid/calibration nen khong phai xac suat."
                        )
                        raw_df = pd.DataFrame(
                            [{"Label (ky thuat)": k, "Decision Score": round(v, 4)}
                             for k, v in sorted(result["scores"].items(), key=lambda x: -x[1])]
                        )
                        st.dataframe(raw_df, width="stretch", hide_index=True)
                        st.markdown(
                            f"**Mo hinh:** LinearSVC | "
                            f"**Vectorizer:** TF-IDF unigram+bigram | "
                            f"**Predicted:** `{label}`"
                        )
            except Exception as exc:
                st.warning(str(exc))

    with tab_pdf:
        st.subheader("Phân tích một file PDF")
        uploaded = st.file_uploader("Tải lên PDF có lớp văn bản", type=["pdf"])
        if uploaded is not None:
            st.write(f"File: `{uploaded.name}` - {uploaded.size / 1024 / 1024:.2f} MB")
        if st.button("Phân tích PDF", type="primary", disabled=(not model_ready or uploaded is None)):
            try:
                with st.spinner("Dang trich xuat van ban va du doan..."):
                    df, meta = analyze_pdf(model, uploaded.getvalue())
                st.success(
                    f"Da phan tich {meta['sentence_count']:,} cau "
                    f"tu {meta['page_count']} trang (dung {meta['engine']})."
                )

                # --- Summary table (friendly) ---
                summary_raw = df["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="So luong")
                emoji_map2 = {"Positive": "Tich cuc", "Neutral": "Trung lap", "Negative": "Tieu cuc"}
                summary_raw["Cam xuc"] = summary_raw["predicted_label"].map(emoji_map2)
                summary_raw["Ti le (%)"] = (summary_raw["So luong"] / summary_raw["So luong"].sum() * 100).round(1)
                summary_display = summary_raw[["Cam xuc", "So luong", "Ti le (%)"]]

                left, right = st.columns([1, 1])
                with left:
                    st.markdown("**Tong hop cam xuc**")
                    st.dataframe(summary_display, width="stretch", hide_index=True)
                with right:
                    chart_data = summary_raw.set_index("Cam xuc")["So luong"]
                    st.bar_chart(chart_data)

                # --- Per-sentence table (friendly) ---
                st.subheader("Ket qua tung cau")
                display_cols = ["STT", "Noi dung cau", "Cam xuc", "Do tin cay (%)"]
                st.dataframe(df[display_cols].head(200), width="stretch", hide_index=True)

                with st.expander("Chi tiet ky thuat"):
                    st.caption(
                        "Bang day du bao gom: nhan ky thuat (Negative/Neutral/Positive), "
                        "do tin cay % (softmax tren decision score), "
                        "va diem % rieng le cua tung nhan."
                    )
                    tech_cols = ["STT", "Noi dung cau", "predicted_label",
                                 "Do tin cay (%)", "Tich cuc (%)", "Trung lap (%)", "Tieu cuc (%)"]
                    st.dataframe(df[tech_cols].head(200), width="stretch", hide_index=True)

                # CSV export keeps all columns for analysis
                st.download_button(
                    "Tai CSV ket qua",
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
            **Ghi chu trien khai**
            - Ung dung tai model mot lan bang `st.cache_resource`.
            - Khong huan luyen lai model khi nguoi dung truy cap.
            - PDF scan chua duoc OCR trong phien ban dau.
            - "Do tin cay" duoc tinh theo softmax tren diem phan quyet cua LinearSVM — la uoc tinh truc quan, khong phai xac suat chinh xac.
            """)


if __name__ == "__main__":
    main()
