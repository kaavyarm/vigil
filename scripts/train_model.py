from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


DATA_PATH = Path("data/features/training_dataset.csv")
MODEL_DIR = Path("models")

TARGET_COL = "In-hospital_death"

LEAKAGE_AND_ID_COLS = [
    "RecordID",
    "SAPS-I",
    "SOFA",
    "Length_of_stay",
    "Survival",
    TARGET_COL,
]


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    print(f"\n==============================")
    print(name)
    print("==============================")
    print(f"AUROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    y = df[TARGET_COL]
    X = df.drop(columns=LEAKAGE_AND_ID_COLS)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Logistic Regression and Random Forest need missing values filled.
    # Use train medians for BOTH train and test to avoid data leakage.
    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    logistic_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    print("\nTraining Logistic Regression...")
    logistic_model.fit(X_train_filled, y_train)
    evaluate_model("Logistic Regression", logistic_model, X_test_filled, y_test)

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    random_forest_model.fit(X_train_filled, y_train)
    evaluate_model("Random Forest", random_forest_model, X_test_filled, y_test)

    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Initial XGBoost...")
    xgboost_model.fit(X_train, y_train)
    evaluate_model("Initial Vigil XGBoost", xgboost_model, X_test, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning 5-fold stratified cross-validation for XGBoost...")
    cv_scores = cross_val_score(
        xgboost_model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("\n==============================")
    print("XGBoost Cross-Validation")
    print("==============================")
    print(f"AUROC scores: {cv_scores}")
    print(f"Mean AUROC: {cv_scores.mean():.4f}")
    print(f"Std AUROC: {cv_scores.std():.4f}")

    print("\n==============================")
    print("XGBoost Hyperparameter Search")
    print("==============================")

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_grid,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    print("Running randomized search...")
    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)
    print(f"\nBest CV AUROC: {search.best_score_:.4f}")

    best_xgboost_model = search.best_estimator_

    evaluate_model(
        "Tuned Vigil XGBoost",
        best_xgboost_model,
        X_test,
        y_test,
    )

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(xgboost_model, MODEL_DIR / "vigil_xgboost_initial.joblib")
    joblib.dump(best_xgboost_model, MODEL_DIR / "vigil_xgboost_tuned.joblib")
    joblib.dump(X_train.columns.tolist(), MODEL_DIR / "feature_columns.joblib")
    joblib.dump(train_medians, MODEL_DIR / "train_medians.joblib")

    print("\nSaved initial model, tuned model, and feature metadata.")
    print("\n==============================")
    print("Clinical Score Baselines")
    print("==============================")
    print(f"SAPS-I AUROC: {roc_auc_score(y_test, df.loc[X_test.index, 'SAPS-I']):.4f}")
    print(f"SOFA AUROC: {roc_auc_score(y_test, df.loc[X_test.index, 'SOFA']):.4f}")


if __name__ == "__main__":
    main()