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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
LABEL_ORDER = ["negative", "neutral", "positive"]
DATA_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def normalize_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng trong văn bản."""
    return re.sub(r"\s+", " ", str(text).strip())


def download_dataset() -> Path:
    """Tải tập Financial PhraseBank nếu chưa có trong bộ nhớ đệm."""
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
    """Đọc cấu hình Sentences_75Agree của Financial PhraseBank."""
    zip_path = download_dataset()

    with zipfile.ZipFile(zip_path) as zf:
        matching_files = [
            name
            for name in zf.namelist()
            if name.endswith("Sentences_75Agree.txt")
        ]

        if not matching_files:
            raise FileNotFoundError(
                "Không tìm thấy Sentences_75Agree.txt trong file ZIP."
            )

        rows: list[dict[str, str]] = []

        with zf.open(matching_files[0]) as file:
            for raw_line in file:
                line = raw_line.decode("latin-1").strip()

                if not line or "@" not in line:
                    continue

                sentence, label = line.rsplit("@", 1)
                label = label.strip().lower()

                if label in LABEL_ORDER:
                    rows.append(
                        {
                            "sentence": sentence.strip(),
                            "label": label,
                        }
                    )

    df = pd.DataFrame(rows)
    print(f"Original data: {len(df):,} sentences")

    empty_count = df["sentence"].str.strip().eq("").sum()
    missing_labels = df["label"].isna().sum()
    print(f"Empty sentences: {empty_count}")
    print(f"Missing labels: {missing_labels}")

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ câu trùng sau khi chuẩn hóa khoảng trắng."""
    cleaned = df.copy()
    cleaned["sentence_norm"] = cleaned["sentence"].map(normalize_text)

    duplicate_count = cleaned.duplicated("sentence_norm").sum()
    print(f"Duplicates after normalization: {duplicate_count}")

    cleaned = cleaned.drop_duplicates("sentence_norm").reset_index(drop=True)
    cleaned = cleaned.drop(columns=["sentence_norm"])

    print(f"Data after deduplication: {len(cleaned):,} sentences")
    return cleaned


def print_label_distribution(df: pd.DataFrame, name: str = "") -> None:
    """In phân bố ba nhãn."""
    distribution = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = len(df)
    prefix = f"[{name}] " if name else ""

    for label in LABEL_ORDER:
        count = int(distribution[label])
        percentage = count / total * 100 if total else 0.0
        print(
            f"  {prefix}{label.capitalize()}: "
            f"{count:,} ({percentage:.2f}%)"
        )


def metrics(
    y_true: pd.Series,
    y_pred,
    name: str,
    train_time: float | None,
) -> dict:
    """Tính các chỉ số đánh giá chính."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )[2]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Training Time": train_time,
    }


def main() -> None:
    print("=" * 60)
    print("FINANCIAL SENTIMENT ANALYSIS — TRAIN & EXPORT")
    print("=" * 60)

    # --- Bước 1: Tải và kiểm tra dữ liệu ---
    raw_df = load_dataset()
    original_samples = len(raw_df)
    df = deduplicate(raw_df)

    print("\nLabel distribution (all data):")
    print_label_distribution(df)

    # --- Bước 2: Chia dữ liệu ---
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1764706,
        random_state=SEED,
        stratify=train_df["label"],
    )

    print(f"\nTrain: {len(train_df):,} sentences")
    print(f"Validation: {len(val_df):,} sentences")
    print(f"Test: {len(test_df):,} sentences")

    # --- Bước 3: Huấn luyện và chọn mô hình trên validation ---
    dummy_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=SEED,
                ),
            ),
        ]
    )

    nb_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )

    nb_grid = GridSearchCV(
        estimator=nb_pipeline,
        param_grid={
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )

    lr_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )

    lr_grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={
            "classifier__C": [0.1, 0.5, 1.0, 3.0, 10.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )

    svm_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )

    svm_grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid={
            "classifier__C": [0.1, 0.5, 1.0, 3.0, 10.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )

    candidates = {
        "Dummy Classifier": dummy_pipeline,
        "Naive Bayes": nb_grid,
        "Logistic Regression": lr_grid,
        "Linear SVM": svm_grid,
    }

    print("\n--- Validation Results ---")

    validation_rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    best_parameters: dict[str, dict] = {}

    for name, estimator in candidates.items():
        start_time = time.perf_counter()

        estimator.fit(
            train_df["sentence"],
            train_df["label"],
        )

        elapsed = time.perf_counter() - start_time

        if isinstance(estimator, GridSearchCV):
            fitted_model = estimator.best_estimator_
            best_parameters[name] = estimator.best_params_
        else:
            fitted_model = estimator
            best_parameters[name] = {}

        validation_prediction = fitted_model.predict(val_df["sentence"])

        row = metrics(
            val_df["label"],
            validation_prediction,
            name,
            elapsed,
        )

        validation_rows.append(row)
        fitted_models[name] = fitted_model

        print(
            f"  {name}: "
            f"Accuracy={row['Accuracy']:.4f}, "
            f"Macro-F1={row['Macro-F1']:.4f} "
            f"({elapsed:.2f}s)"
        )

        if best_parameters[name]:
            print(f"    Best parameters: {best_parameters[name]}")

    validation_metrics = (
        pd.DataFrame(validation_rows)
        .sort_values("Macro-F1", ascending=False)
        .reset_index(drop=True)
    )

    best_name = str(validation_metrics.iloc[0]["Model"])
    best_model = fitted_models[best_name]

    print("\nValidation ranking:")
    print(
        validation_metrics[
            [
                "Model",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro-F1",
                "Weighted-F1",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model (validation Macro-F1): {best_name}")

    # --- Bước 4: Đánh giá test đúng một lần ---
    test_prediction = best_model.predict(test_df["sentence"])
    test_result = metrics(
        test_df["label"],
        test_prediction,
        best_name,
        None,
    )

    print(f"\n--- Test Results ({best_name}) ---")
    print(f"  Accuracy: {test_result['Accuracy']:.4f}")
    print(f"  Macro Precision: {test_result['Macro Precision']:.4f}")
    print(f"  Macro Recall: {test_result['Macro Recall']:.4f}")
    print(f"  Macro-F1: {test_result['Macro-F1']:.4f}")
    print(f"  Weighted-F1: {test_result['Weighted-F1']:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_df["label"],
            test_prediction,
            labels=LABEL_ORDER,
            digits=4,
            zero_division=0,
        )
    )

    # --- Bước 5: Huấn luyện lại mô hình tốt nhất trên train + validation ---
    deployment_train = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    print(
        f"Training samples for deployment: "
        f"{len(deployment_train):,} (train + validation)"
    )

    final_pipeline = clone(best_model)
    final_pipeline.fit(
        deployment_train["sentence"],
        deployment_train["label"],
    )

    print(f"Deployment model: {best_name}")

    if best_parameters.get(best_name):
        print(f"Deployment parameters: {best_parameters[best_name]}")

    # --- Bước 6: Lưu mô hình ---
    model_path = MODELS_DIR / "financial_sentiment_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    print(
        f"\nSaved model: {model_path} "
        f"({model_path.stat().st_size / 1024:.1f} KB)"
    )

    # --- Bước 7: Kiểm tra tải lại mô hình ---
    loaded_model = joblib.load(model_path)
    test_sentence = "Revenue increased significantly during the year."
    loaded_prediction = loaded_model.predict([test_sentence])[0]

    print(
        f"Verify load: predict('{test_sentence}') "
        f"= {loaded_prediction}"
    )

    # --- Bước 8: Lưu metadata ---
    metadata = {
        "project_name": "Financial Sentiment Analysis",
        "model_name": best_name,
        "best_parameters": best_parameters.get(best_name, {}),
        "vectorizer": "TF-IDF",
        "ngram_range": [1, 2],
        "labels": LABEL_ORDER,
        "label_mapping": {
            label: index
            for index, label in enumerate(LABEL_ORDER)
        },
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": round(
            float(validation_metrics.iloc[0]["Macro-F1"]),
            4,
        ),
        "test_accuracy": round(float(test_result["Accuracy"]), 4),
        "test_macro_f1": round(float(test_result["Macro-F1"]), 4),
        "test_macro_precision": round(
            float(test_result["Macro Precision"]),
            4,
        ),
        "test_macro_recall": round(
            float(test_result["Macro Recall"]),
            4,
        ),
        "test_weighted_f1": round(
            float(test_result["Weighted-F1"]),
            4,
        ),
        "dataset_original_samples": int(original_samples),
        "dataset_after_deduplication": int(len(df)),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "training_samples": int(len(deployment_train)),
        "test_samples": int(len(test_df)),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "created_from_script": "scripts/train_export_model.py",
        "created_from_notebook": (
            "notebook/Financial_Sentiment_Analysis_Fixed_collab.ipynb"
        ),
    }

    metadata_path = MODELS_DIR / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved metadata: {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("DONE. Model is ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
