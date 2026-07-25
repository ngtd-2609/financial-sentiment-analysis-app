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


def load_gemini():
    """Khởi tạo Gemini client từ Streamlit Secrets."""
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def ask_gemini(gemini_model, sentence: str, label: str, confidence: float) -> str:
    """Gọi Gemini để giải thích kết quả phân loại."""
    label_vi = {"Positive": "Tích cực", "Neutral": "Trung lập", "Negative": "Tiêu cực"}.get(label, label)
    prompt = f"""Bạn là chuyên gia phân tích ngôn ngữ tài chính.
Mô hình NLP (TF-IDF + Linear SVM) vừa phân loại câu tiếng Anh sau đây là **{label_vi}** với độ tin cậy **{confidence:.1f}%**:

"{sentence}"

Hãy giải thích ngắn gọn (2-4 câu) bằng tiếng Việt:
1. Tại sao câu này được coi là {label_vi}?
2. Những từ/cụm từ nào trong câu cho thấy điều đó?

Trả lời tự nhiên, dễ hiểu, không dùng markdown phức tạp."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        return f"Không thể lấy giải thích từ Gemini: {exc}"


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

    gemini_model = load_gemini()

    tab_text, tab_pdf, tab_chat, tab_info = st.tabs([
        "Dự đoán một câu", "Phân tích PDF", "💬 Hỏi đáp (AI)", "Thông tin mô hình"
    ])

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
                df_display = df[["STT", "Noi dung cau", "Cam xuc", "Do tin cay (%)"]].copy()
                df_display.columns = ["STT", "Nội dung câu", "Cảm xúc", "Độ tin cậy (%)"]
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

                st.download_button(
                    "Tải CSV kết quả",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="financial_sentiment_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(str(exc))

    # ── Tab 3: Chatbot AI ────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("💬 Hỏi đáp với AI")
        st.caption(
            "Nhập câu tài chính tiếng Anh — mô hình sẽ phân loại cảm xúc, "
            "sau đó Gemini AI sẽ giải thích lý do bằng tiếng Việt."
        )

        if not model_ready:
            st.error("Model chưa sẵn sàng. Vui lòng kiểm tra lại.")
        elif gemini_model is None:
            st.warning(
                "Chưa tìm thấy GEMINI_API_KEY trong Streamlit Secrets. "
                "Vào Settings → Secrets và thêm: `GEMINI_API_KEY = \"key_cua_ban\"`"
            )
        else:
            # Khởi tạo lịch sử chat
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Hiển thị lịch sử
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Ô nhập liệu
            if prompt := st.chat_input("Nhập câu tài chính tiếng Anh..."):
                # Hiển thị câu người dùng
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Phân loại bằng model
                try:
                    result = predict_sentence(model, prompt)
                    label = result["predicted_label"]
                    conf = softmax_confidence(result["scores"])
                    best_conf = conf.get(label, 0.0)
                    label_vi = {"Positive": "Tích cực", "Neutral": "Trung lập", "Negative": "Tiêu cực"}.get(label, label)
                    color_map = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
                    icon = color_map.get(label, "🔵")

                    # Gọi Gemini để giải thích
                    with st.spinner("Gemini đang giải thích..."):
                        explanation = ask_gemini(gemini_model, prompt, label, best_conf)

                    # Tạo phản hồi đầy đủ
                    reply = (
                        f"{icon} **Kết quả: {label_vi}** (độ tin cậy: {best_conf:.1f}%)\n\n"
                        f"{explanation}"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                except Exception as exc:
                    err_msg = f"Lỗi: {exc}"
                    with st.chat_message("assistant"):
                        st.warning(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

            # Nút xóa lịch sử
            if st.session_state.get("chat_history"):
                if st.button("🗑️ Xóa lịch sử chat"):
                    st.session_state.chat_history = []
                    st.rerun()

    # ── Tab 4: Thông tin mô hình ─────────────────────────────────────────────
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
            - Tab "Hỏi đáp" dùng Gemini 1.5 Flash để giải thích kết quả phân loại.
            """
        )


if __name__ == "__main__":
    main()
