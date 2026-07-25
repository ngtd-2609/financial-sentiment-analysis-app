from __future__ import annotations

import json
import platform
import re
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

SEED = 42
LABEL_ORDER = ["negative", "neutral", "positive"]
DATA_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def normalize_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", str(text).strip())


def download_dataset() -> Path:
    zip_path = DATA_DIR / "FinancialPhraseBank-v1.0.zip"
    if zip_path.exists():
        print(f"Dataset already exists: {zip_path}")
        return zip_path
    print(f"Downloading dataset from {DATA_URL} ...")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, open(zip_path, "wb") as out:
        shutil.copyfileobj(response, out)
    print(f"Downloaded: {zip_path}")
    return zip_path


def load_dataset() -> pd.DataFrame:
    zip_path = download_dataset()
    with zipfile.ZipFile(zip_path) as zf:
        target = [n for n in zf.namelist() if n.endswith("Sentences_75Agree.txt")][0]
        rows = []
        with zf.open(target) as f:
            for raw_line in f:
                line = raw_line.decode("latin-1").strip()
                if not line or "@" not in line:
                    continue
                sentence, label = line.rsplit("@", 1)
                label = label.strip().lower()
                if label in LABEL_ORDER:
                    rows.append({
                        "sentence": sentence.strip(),
                        "label": label,
                    })
    df = pd.DataFrame(rows)
    print(f"Original data: {len(df):,} sentences")

    # Quality check
    empty_count = df["sentence"].str.strip().eq("").sum()
    null_label = df["label"].isna().sum()
    print(f"Empty sentences: {empty_count}")
    print(f"Missing labels: {null_label}")

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates after whitespace normalization."""
    df = df.copy()
    df["sentence_norm"] = df["sentence"].map(normalize_text)
    dup_count = df.duplicated("sentence_norm").sum()
    print(f"Duplicates after normalization: {dup_count}")
    df = df.drop_duplicates("sentence_norm").reset_index(drop=True)
    df = df.drop(columns=["sentence_norm"])
    print(f"Data after deduplication: {len(df):,} sentences")
    return df


def print_label_distribution(df: pd.DataFrame, name: str = "") -> None:
    dist = df["label"].value_counts().reindex(LABEL_ORDER)
    total = len(df)
    prefix = f"[{name}] " if name else ""
    for label in LABEL_ORDER:
        count = dist[label]
        pct = count / total * 100
        print(f"  {prefix}{label.capitalize()}: {count:,} ({pct:.2f}%)")


def metrics(y_true, y_pred, name: str, train_time: float | None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    wf1 = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0
    )[2]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": p,
        "Macro Recall": r,
        "Macro-F1": f1,
        "Weighted-F1": wf1,
        "Training Time": train_time,
    }


def main():
    print("=" * 60)
    print("FINANCIAL SENTIMENT ANALYSIS — TRAIN & EXPORT")
    print("=" * 60)

    # --- Bước 1: Tải và kiểm tra dữ liệu ---
    df = load_dataset()
    df = deduplicate(df)

    print("\nLabel distribution (all data):")
    print_label_distribution(df)

    # --- Bước 2: Chia dữ liệu ---
    # test_size=0.15 → ~518 test
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["label"]
    )
    # Từ train còn lại, tách validation: 518 / (3448 - 518) = 0.1764706
    train_df, val_df = train_test_split(
        train_df, test_size=0.1764706, random_state=SEED, stratify=train_df["label"]
    )
    print(f"\nTrain: {len(train_df):,} sentences")
    print(f"Validation: {len(val_df):,} sentences")
    print(f"Test: {len(test_df):,} sentences")

    # --- Bước 3: Huấn luyện các baseline trên validation ---
    # Pipeline nhận raw text → TF-IDF tự lowercase
    candidates = {
        "Dummy Classifier": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("classifier", DummyClassifier(strategy="most_frequent", random_state=SEED)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("classifier", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("classifier", LinearSVC(random_state=SEED)),
        ]),
    }

    print("\n--- Validation Results ---")
    val_rows = []
    fitted = {}
    for name, pipe in candidates.items():
        start = time.perf_counter()
        pipe.fit(train_df["sentence"], train_df["label"])
        elapsed = time.perf_counter() - start
        pred = pipe.predict(val_df["sentence"])
        row = metrics(val_df["label"], pred, name, elapsed)
        val_rows.append(row)
        fitted[name] = pipe
        print(f"  {name}: Accuracy={row['Accuracy']:.4f}, Macro-F1={row['Macro-F1']:.4f} ({elapsed:.2f}s)")

    val_metrics = pd.DataFrame(val_rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    best_name = val_metrics.iloc[0]["Model"]
    print(f"\nBest model (validation Macro-F1): {best_name}")

    # --- Bước 4: Đánh giá test đúng một lần ---
    best_model = fitted[best_name]
    test_pred = best_model.predict(test_df["sentence"])
    test_result = metrics(test_df["label"], test_pred, best_name, None)

    print(f"\n--- Test Results ({best_name}) ---")
    print(f"  Accuracy: {test_result['Accuracy']:.4f}")
    print(f"  Macro Precision: {test_result['Macro Precision']:.4f}")
    print(f"  Macro Recall: {test_result['Macro Recall']:.4f}")
    print(f"  Macro-F1: {test_result['Macro-F1']:.4f}")
    print(f"  Weighted-F1: {test_result['Weighted-F1']:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(test_df["label"], test_pred, labels=LABEL_ORDER, digits=4))

    # --- Bước 5: Huấn luyện model deploy trên train + validation ---
    deploy_train = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Training samples for deployment: {len(deploy_train):,} (train + validation)")

    # Tạo pipeline mới (dùng Logistic Regression theo notebook)
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    final_pipeline.fit(deploy_train["sentence"], deploy_train["label"])

    # --- Bước 6: Lưu model ---
    model_path = MODELS_DIR / "financial_sentiment_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"\nSaved model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # --- Bước 7: Verify load lại ---
    loaded = joblib.load(model_path)
    test_sentence = "Revenue increased significantly during the year."
    loaded_pred = loaded.predict([test_sentence])[0]
    print(f"Verify load: predict('{test_sentence}') = {loaded_pred}")

    # --- Bước 8: Lưu metadata ---
    metadata = {
        "project_name": "Financial Sentiment Analysis",
        "model_name": best_name,
        "vectorizer": "TF-IDF",
        "ngram_range": [1, 2],
        "labels": LABEL_ORDER,
        "label_mapping": {label: idx for idx, label in enumerate(LABEL_ORDER)},
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": round(float(val_metrics.iloc[0]["Macro-F1"]), 4),
        "test_accuracy": round(float(test_result["Accuracy"]), 4),
        "test_macro_f1": round(float(test_result["Macro-F1"]), 4),
        "test_macro_precision": round(float(test_result["Macro Precision"]), 4),
        "test_macro_recall": round(float(test_result["Macro Recall"]), 4),
        "test_weighted_f1": round(float(test_result["Weighted-F1"]), 4),
        "dataset_original_samples": int(len(load_dataset())),
        "dataset_after_deduplication": int(len(df)),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "training_samples": int(len(deploy_train)),
        "test_samples": int(len(test_df)),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "created_from_script": "scripts/train_export_model.py",
        "created_from_notebook": "Financial_Sentiment_Analysis_Application.ipynb",
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("DONE. Model is ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
