"""
generate_shap_explanations.py
Tạo SHAP plots từ model package đã train sẵn mà không cần huấn luyện lại toàn bộ.

Chạy:
    python generate_shap_explanations.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from carbon_utils import RANDOM_STATE, FEATURE_COLS

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight", dpi=140)
    plt.close()


def get_preprocessed_frame(pipe: Any, X: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    if hasattr(pipe, "base_estimator"):
        pipe = pipe.base_estimator
    preprocessor = pipe.named_steps.get("preprocessor")
    model = pipe.named_steps.get("model")
    X_trans = preprocessor.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
    return model, pd.DataFrame(X_trans, columns=feature_names)


def normalise_shap_values(raw_values: Any) -> list[np.ndarray]:
    if isinstance(raw_values, list):
        return [np.asarray(v) for v in raw_values]
    arr = np.asarray(raw_values)
    if arr.ndim == 3:
        return [arr[:, :, i] for i in range(arr.shape[2])]
    return [arr]


def main() -> None:
    import shap  # type: ignore

    model_path = Path("ecopredict_model_package.joblib")
    if not model_path.exists():
        model_path = Path("outputs/models/ecopredict_model_package.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Không tìm thấy ecopredict_model_package.joblib")

    package = joblib.load(model_path)
    clf = package["classifier"]
    meta = package.get("metadata", {})
    feature_cols = meta.get("feature_cols", FEATURE_COLS)

    if "test_data" in package and isinstance(package["test_data"], pd.DataFrame):
        df = package["test_data"].copy()
    else:
        df = package["reference_data"].copy()

    df = df.dropna(subset=[c for c in feature_cols if c in df.columns], how="all")
    X = df[feature_cols].copy()
    sample_n = min(160, len(X))
    if len(X) > sample_n:
        X = X.sample(sample_n, random_state=RANDOM_STATE)

    model, X_enc = get_preprocessed_frame(clf, X)
    if not hasattr(model, "feature_importances_"):
        raise TypeError("Best classifier không phải tree model phù hợp SHAP TreeExplainer")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_enc)
    shap_values_list = normalise_shap_values(raw)
    class_idx = min(2, len(shap_values_list) - 1)
    sv = shap_values_list[class_idx]

    abs_stack = np.stack([np.abs(v) for v in shap_values_list], axis=2) if len(shap_values_list) > 1 else np.abs(sv)[:, :, None]
    mean_abs = abs_stack.mean(axis=(0, 2))
    imp = pd.DataFrame({"feature": X_enc.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(TAB / "shap_feature_importance.csv", index=False)

    top = imp.head(15).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#047857")
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Ranking - mô hình phân loại")
    savefig("model_shap_summary_bar.png")

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_enc, max_display=20, show=False)
    plt.title("SHAP Beeswarm - lớp phát thải cao/tham chiếu")
    savefig("model_shap_beeswarm.png")

    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            base_value = np.asarray(expected).ravel()[class_idx]
        else:
            base_value = expected
        explanation = shap.Explanation(
            values=sv[0], base_values=base_value,
            data=X_enc.iloc[0].values, feature_names=X_enc.columns.tolist(),
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        savefig("model_shap_waterfall_first_sample.png")
    except Exception as exc:
        print("Waterfall skipped:", exc)

    try:
        top_feature = str(imp.iloc[0]["feature"])
        plt.figure(figsize=(8, 5.5))
        shap.dependence_plot(top_feature, sv, X_enc, show=False)
        plt.title(f"SHAP Dependence - {top_feature}")
        savefig("model_shap_dependence_top_feature.png")
    except Exception as exc:
        print("Dependence skipped:", exc)

    print(f"SHAP done. sample_n={sample_n}; encoded_features={X_enc.shape[1]}")


if __name__ == "__main__":
    main()
