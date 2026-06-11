"""
test_improvements.py
Demo nhanh: class imbalance + SHAP + sensitivity.

Chạy:
    python test_improvements.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from imbalance_handler import try_apply_smote, compute_balanced_class_weights
from sensitivity_analysis import generate_default_sensitivity_outputs


def main() -> None:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    print("Creating demo imbalanced dataset...")
    X, y = make_classification(
        n_samples=900,
        n_features=20,
        n_informative=12,
        n_redundant=2,
        n_classes=3,
        weights=[0.82, 0.15, 0.03],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("Class weights:", compute_balanced_class_weights(y_train))
    X_res, y_res, report = try_apply_smote(X_train, y_train, verbose=True)
    print(report)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=80, class_weight="balanced_subsample", random_state=42, n_jobs=1)),
    ])
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))

    try:
        from model_interpretation import explain_classifier_shap
        shap_info = explain_classifier_shap(model, X_test, class_idx=2, max_samples=60)
        print("SHAP:", shap_info)
    except Exception as exc:
        print("SHAP demo skipped:", exc)

    print(generate_default_sensitivity_outputs(67.67))
    print("Done. Check outputs/figures and outputs/tables.")


if __name__ == "__main__":
    main()
