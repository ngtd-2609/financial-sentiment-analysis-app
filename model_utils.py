from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_DISPLAY = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "Negative": "Negative",
    "Neutral": "Neutral",
    "Positive": "Positive",
}

VALID_LABELS = {"Negative", "Neutral", "Positive"}


def normalize_label(label: Any) -> str:
    """Normalize numeric or string labels to display labels."""
    return LABEL_DISPLAY.get(label, str(label).strip().capitalize())


def get_classifier(model: Any) -> Any:
    """Return the last estimator if the model is a scikit-learn Pipeline."""
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def decision_scores(model: Any, texts: list[str]) -> list[dict[str, float]]:
    """Return decision scores if available. These are not probabilities."""
    if not hasattr(model, "decision_function"):
        return [{} for _ in texts]
    raw = model.decision_function(texts)
    classifier = get_classifier(model)
    classes = list(getattr(classifier, "classes_", []))
    if not classes:
        classes = ["negative", "neutral", "positive"]
    results: list[dict[str, float]] = []
    if getattr(raw, "ndim", 1) == 1:
        for value in raw:
            results.append({"score": float(value)})
    else:
        for row in raw:
            results.append({normalize_label(c): float(v) for c, v in zip(classes, row)})
    return results


def predict_sentence(model: Any, sentence: str) -> dict[str, Any]:
    """Predict sentiment for a single sentence."""
    text = sentence.strip()
    if not text:
        raise ValueError("Nội dung câu đang trống.")
    pred = model.predict([text])[0]
    scores = decision_scores(model, [text])[0]
    return {"sentence": text, "predicted_label": normalize_label(pred), "scores": scores}


def split_sentences(text: str, min_words: int = 5, max_words: int = 120) -> list[str]:
    """Simple sentence splitter for financial reports."""
    if text is None or not text.strip():
        return []
    text = re.sub(r"\s+", " ", text).strip()
    # Avoid splitting common abbreviations too aggressively in a basic way.
    protected = text.replace("U.S.", "US").replace("e.g.", "eg").replace("i.e.", "ie")
    raw = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for sent in raw:
        sent = sent.strip()
        words = sent.split()
        if min_words <= len(words) <= max_words:
            sentences.append(sent)
    return sentences


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int, str]:
    """Extract text from PDF using PyMuPDF, then pdfplumber as fallback."""
    if not pdf_bytes:
        raise ValueError("File PDF đang trống.")
    # Primary: PyMuPDF
    try:
        import fitz
        pages = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    pages.append(page_text)
        text = "\n".join(pages)
        if len(text.strip()) >= 200:
            return text, page_count, "PyMuPDF"
    except Exception:
        text = ""
        page_count = 0
    # Fallback: pdfplumber
    try:
        import io
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
        text = "\n".join(pages)
        return text, page_count, "pdfplumber"
    except Exception as exc:
        raise ValueError(f"Không thể trích xuất văn bản từ PDF: {exc}") from exc


def analyze_pdf(model: Any, pdf_bytes: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract sentences from a PDF and predict sentiment in batch."""
    text, page_count, engine = extract_pdf_text(pdf_bytes)
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Không tìm thấy câu hợp lệ. PDF có thể là bản scan hoặc quá nhiều bảng biểu.")
    preds = model.predict(sentences)
    score_list = decision_scores(model, sentences)
    rows = []
    for idx, (sentence, pred, scores) in enumerate(zip(sentences, preds, score_list), start=1):
        row = {"sentence_index": idx, "sentence": sentence, "predicted_label": normalize_label(pred)}
        for label, score in scores.items():
            row[f"score_{label.lower()}"] = score
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = {"page_count": page_count, "engine": engine, "sentence_count": len(df)}
    return df, meta


def load_metadata(path: Path) -> dict[str, Any]:
    """Load model metadata if available."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
