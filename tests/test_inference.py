from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from model_utils import split_sentences

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "financial_sentiment_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

VALID_LABELS = {"negative", "neutral", "positive"}


# --- Metadata tests ---

def test_metadata_exists():
    """1. Metadata tồn tại."""
    assert METADATA_PATH.exists(), "model_metadata.json không tồn tại"


def test_metadata_readable():
    """7. Metadata đọc được."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)


def test_metadata_has_required_fields():
    """8. Metadata đủ trường."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required = [
        "project_name", "model_name", "vectorizer", "labels",
        "selection_metric", "validation_macro_f1",
        "test_accuracy", "test_macro_f1",
        "training_samples", "test_samples",
    ]
    for key in required:
        assert key in metadata, f"Thiếu trường '{key}' trong metadata"


# --- Model tests ---

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo; chạy scripts/train_export_model.py trước.")
def test_model_exists():
    """1. Model tồn tại."""
    assert MODEL_PATH.exists()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo.")
def test_model_loads():
    """2. Model load được."""
    model = joblib.load(MODEL_PATH)
    assert model is not None


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo.")
def test_model_has_predict():
    """3. Model có method predict."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Model không có method 'predict'"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo.")
def test_predict_single_sentence():
    """4. Dự đoán một câu không lỗi."""
    model = joblib.load(MODEL_PATH)
    pred = model.predict(["Revenue increased by 20 percent compared with the previous year."])[0]
    assert str(pred).lower() in VALID_LABELS


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo.")
def test_predict_batch():
    """5. Dự đoán batch không lỗi."""
    model = joblib.load(MODEL_PATH)
    sentences = [
        "Revenue increased by 20 percent compared with the previous year.",
        "The company was established in 1995.",
        "Operating profit declined because of higher raw material costs.",
    ]
    preds = model.predict(sentences)
    assert len(preds) == 3


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model joblib chưa được tạo.")
def test_predict_labels_valid():
    """6. Nhãn thuộc negative, neutral, positive."""
    model = joblib.load(MODEL_PATH)
    sentences = [
        "Revenue increased by 20 percent compared with the previous year.",
        "The company was established in 1995.",
        "Operating profit declined because of higher raw material costs.",
    ]
    preds = model.predict(sentences)
    for pred in preds:
        assert str(pred).lower() in VALID_LABELS, f"Nhãn '{pred}' không hợp lệ"


# --- split_sentences tests ---

def test_split_sentences_empty():
    """9. split_sentences('') không lỗi."""
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences(None) == []


def test_split_sentences_filters_short():
    """10. Câu quá ngắn bị loại."""
    text = "Ok. Revenue increased by 20 percent compared with the previous year. Bad."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert "Revenue increased" in sentences[0]
