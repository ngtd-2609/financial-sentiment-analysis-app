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

    # ── Tab 1: Dự đoán một câu ──────────────────────────────────────────────
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
                label = result["predicted_label"]
                emoji_map = {"Positive": "Tích cực", "Neutral": "Trung lập", "Negative": "Tiêu cực"}
                display = emoji_map.get(label, label)
                color_map = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
                color = color_map.get(label, "blue")
                st.markdown(
                    f"<h3 style='color:{color}'>Kết quả: {display}</h3>",
                    unsafe_allow_html=True,
                )
                if result["scores"]:
                    conf = softmax_confidence(result["scores"])
                    conf_df = pd.DataFrame(
                        [{"Cảm xúc": emoji_map.get(k, k), "Độ tin cậy (%)": v}
                         for k, v in sorted(conf.items(), key=lambda x: -x[1])]
                    )
                    st.caption(
                        "Độ tin cậy được tính từ điểm phân quyết của mô hình (Linear SVM), "
                        "không phải xác suất chính thức. Giá trị càng cao thì mô hình càng chắc chắn về nhãn đó."
                    )
                    st.dataframe(conf_df, width="stretch", hide_index=True)

                    with st.expander("Chi tiết kỹ thuật"):
                        st.caption(
                            "Decision score là điểm phân quyết của LinearSVC trước khi phân ngưỡng. "
                            "Giá trị dương và lớn hơn thì nhãn đó được mô hình ưu tiên hơn. "
                            "Đây là điểm thô, CHƯA qua sigmoid/calibration nên không phải xác suất."
                        )
                        raw_df = pd.DataFrame(
                            [{"Nhãn": k, "Decision Score": round(v, 4)}
                             for k, v in sorted(result["scores"].items(), key=lambda x: -x[1])]
                        )
                        st.dataframe(raw_df, width="stretch", hide_index=True)
                        st.markdown(
                            f"**Mô hình:** LinearSVC | "
                            f"**Vectorizer:** TF-IDF unigram+bigram | "
                            f"**Nhãn dự đoán:** `{label}`"
                        )
            except Exception as exc:
                st.warning(str(exc))

    # ── Tab 2: Phân tích PDF ─────────────────────────────────────────────────
    with tab_pdf:
        st.subheader("Phân tích một file PDF")
        uploaded = st.file_uploader("Tải lên PDF có lớp văn bản", type=["pdf"])
        if uploaded is not None:
            st.write(f"File: `{uploaded.name}` - {uploaded.size / 1024 / 1024:.2f} MB")
        if st.button("Phân tích PDF", type="primary", disabled=(not model_ready or uploaded is None)):
            try:
                with st.spinner("Đang trích xuất văn bản và dự đoán..."):
                    df, meta = analyze_pdf(model, uploaded.getvalue())
                st.success(
                    f"Đã phân tích {meta['sentence_count']:,} câu "
                    f"từ {meta['page_count']} trang (dùng {meta['engine']})."
                )

                # --- Bảng tóm tắt ---
                summary_raw = df["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="Số lượng")
                emoji_map2 = {"Positive": "Tích cực", "Neutral": "Trung lập", "Negative": "Tiêu cực"}
                summary_raw["Cảm xúc"] = summary_raw["predicted_label"].map(emoji_map2)
                summary_raw["Tỉ lệ (%)"] = (summary_raw["Số lượng"] / summary_raw["Số lượng"].sum() * 100).round(1)
                summary_display = summary_raw[["Cảm xúc", "Số lượng", "Tỉ lệ (%)"]]

                left, right = st.columns([1, 1])
                with left:
                    st.markdown("**Tổng hợp cảm xúc**")
                    st.dataframe(summary_display, width="stretch", hide_index=True)
                with right:
                    chart_data = summary_raw.set_index("Cảm xúc")["Số lượng"]
                    st.bar_chart(chart_data)

                # --- Bảng từng câu (thân thiện) ---
                st.subheader("Kết quả từng câu")
                display_cols = ["STT", "Nội dung câu", "Cảm xúc", "Độ tin cậy (%)"]
                # Đổi tên cột sang tiếng Việt có dấu để hiển thị
                df_display = df[["STT", "Noi dung cau", "Cam xuc", "Do tin cay (%)"]].copy()
                df_display.columns = display_cols
                st.dataframe(df_display.head(200), width="stretch", hide_index=True)

                with st.expander("Chi tiết kỹ thuật"):
                    st.caption(
                        "Bảng đầy đủ bao gồm: nhãn kỹ thuật (Negative/Neutral/Positive), "
                        "độ tin cậy % (softmax trên decision score), "
                        "và điểm % riêng lẻ của từng nhãn."
                    )
                    tech_cols_src = ["STT", "Noi dung cau", "predicted_label",
                                     "Do tin cay (%)", "Tich cuc (%)", "Trung lap (%)", "Tieu cuc (%)"]
                    tech_cols_display = ["STT", "Nội dung câu", "Nhãn kỹ thuật",
                                         "Độ tin cậy (%)", "Tích cực (%)", "Trung lập (%)", "Tiêu cực (%)"]
                    df_tech = df[tech_cols_src].copy()
                    df_tech.columns = tech_cols_display
                    st.dataframe(df_tech.head(200), width="stretch", hide_index=True)

                # CSV xuất đầy đủ cột để phân tích
                st.download_button(
                    "Tải CSV kết quả",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="financial_sentiment_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(str(exc))

    # ── Tab 3: Thông tin mô hình ─────────────────────────────────────────────
    with tab_info:
        st.subheader("Thông tin mô hình")
        show_metadata(metadata)
        st.markdown(
            """
            **Ghi chú triển khai**
            - Ứng dụng tải model một lần bằng `st.cache_resource`.
            - Không huấn luyện lại model khi người dùng truy cập.
            - PDF scan chưa được OCR trong phiên bản đầu.
            - "Độ tin cậy" được tính theo softmax trên điểm phân quyết của LinearSVM — là ước tính trực quan, không phải xác suất chính xác.
            """
        )


if __name__ == "__main__":
    main()
