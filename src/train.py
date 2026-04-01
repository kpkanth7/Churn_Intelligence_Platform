import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import OUTPUTS_DIR, PROCESSED_DATA_PATH, RANDOM_STATE, build_dataset
from inference import build_curated_cases, predict_dataframe, save_sample_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "training_metrics.json"


def evaluate_model(model, X_test, y_test):
    # i keep the evaluation compact here because this file is the reproducible training entrypoint.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": matrix.tolist(),
    }


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # i stay with a plain sklearn pipeline here because it is easier to explain and maintain.
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # i always rebuild the processed data first so training starts from the same clean source of truth.
    df = build_dataset(save=True)

    y = df["Churn"].astype(int)
    X = df.drop(columns=["customerID", "Churn", "complaints"], errors="ignore")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    model_metrics = {}
    fitted_models = {}
    # i train both baselines every run so model selection stays explicit and repeatable.
    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        model_metrics[model_name] = evaluate_model(model, X_test, y_test)

    best_name = max(
        model_metrics,
        key=lambda name: (
            model_metrics[name]["f1"],
            model_metrics[name]["roc_auc"],
        ),
    )
    best_model = fitted_models[best_name]

    joblib.dump(best_model, BEST_MODEL_PATH)

    metadata = {
        "selected_model": best_name,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "processed_data_path": str(PROCESSED_DATA_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "target_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().items()
        },
        "synthetic_complaint_enrichment": True,
        "random_state": RANDOM_STATE,
    }

    save_json(METADATA_PATH, metadata)
    save_json(
        METRICS_PATH,
        {
            "selected_model": best_name,
            "models": model_metrics,
        },
    )

    # i save a business-facing holdout report because raw probabilities alone are not enough for this project.
    prediction_report = predict_dataframe(X_test.reset_index(drop=True), model=best_model, metadata=metadata)
    prediction_report.insert(0, "actual_churn", y_test.reset_index(drop=True).astype(int))
    prediction_report.to_csv(PROCESSED_DATA_PATH.parent / "test_predictions.csv", index=False)
    save_sample_prediction(prediction_report)
    build_curated_cases()

    print("\nModel comparison")
    for model_name, scores in model_metrics.items():
        print(f"\n{model_name}")
        for metric_name, metric_value in scores.items():
            print(f"{metric_name}: {metric_value}")

    print(f"\nSelected model: {best_name}")
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Best model saved to {BEST_MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Sample prediction saved to {OUTPUTS_DIR / 'sample_prediction.json'}")


if __name__ == "__main__":
    main()
