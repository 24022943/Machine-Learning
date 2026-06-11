"""
model_interpretation.py
SHAP/XAI helpers cho EcoPredict Carbon.

Có thể dùng độc lập hoặc gọi trong train_advanced_models.py/generate_shap_explanations.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_preprocessed_frame(pipe: Any, X: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Trích model và ma trận features sau preprocessing từ sklearn/imblearn pipeline."""
    if hasattr(pipe, "base_estimator"):
        pipe = pipe.base_estimator
    preprocessor = pipe.named_steps.get("preprocessor")
    model = pipe.named_steps.get("model")
    if preprocessor is None or model is None:
        raise ValueError("Pipeline cần có named_steps: preprocessor và model")
    X_trans = preprocessor.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
    return model, pd.DataFrame(X_trans, columns=feature_names)


def normalise_shap_values(raw_values: Any) -> list[np.ndarray]:
    """Chuẩn hóa SHAP output giữa các phiên bản shap."""
    if isinstance(raw_values, list):
        return [np.asarray(v) for v in raw_values]
    arr = np.asarray(raw_values)
    if arr.ndim == 3:
        return [arr[:, :, i] for i in range(arr.shape[2])]
    return [arr]


def explain_classifier_shap(
    pipe: Any,
    X: pd.DataFrame,
    class_idx: int = 2,
    max_samples: int = 120,
    output_dir: str | Path = "outputs/figures",
    table_dir: str | Path = "outputs/tables",
) -> dict[str, Any]:
    """Sinh SHAP summary bar, beeswarm, waterfall và dependence plot.

    Hàm ưu tiên class_idx=2 tương ứng lớp phát thải Cao. Nếu số lớp ít hơn,
    tự fallback về lớp cuối cùng.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"Không import được shap: {exc}"}

    output_dir = Path(output_dir)
    table_dir = Path(table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if X.empty:
        return {"status": "skipped", "reason": "X rỗng"}

    X_sample = X.sample(min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X.copy()
    try:
        model, X_enc = get_preprocessed_frame(pipe, X_sample)
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_enc)
        shap_values = normalise_shap_values(raw)
        class_idx = min(int(class_idx), len(shap_values) - 1)
        sv = np.asarray(shap_values[class_idx])

        # Global mean abs SHAP across classes.
        if len(shap_values) > 1:
            mean_abs = np.stack([np.abs(np.asarray(v)) for v in shap_values], axis=2).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({"feature": X_enc.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(table_dir / "shap_feature_importance.csv", index=False)

        top = imp.head(18).sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(9, 6))
        plt.barh(top["feature"], top["mean_abs_shap"], color="#047857")
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Ranking")
        plt.tight_layout()
        plt.savefig(output_dir / "model_shap_summary_bar.png", dpi=180, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_enc, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "model_shap_beeswarm.png", dpi=180, bbox_inches="tight")
        plt.close()

        try:
            expected = explainer.expected_value
            base_value = np.asarray(expected).ravel()[class_idx] if isinstance(expected, (list, tuple, np.ndarray)) else expected
            explanation = shap.Explanation(values=sv[0], base_values=base_value, data=X_enc.iloc[0].values, feature_names=X_enc.columns.tolist())
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "model_shap_waterfall_first_sample.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            waterfall_warning = str(exc)
        else:
            waterfall_warning = None

        try:
            top_feature = str(imp.iloc[0]["feature"])
            plt.figure(figsize=(9, 5.6))
            shap.dependence_plot(top_feature, sv, X_enc, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "model_shap_dependence_top_feature.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            dependence_warning = str(exc)
        else:
            dependence_warning = None

        return {
            "status": "ok",
            "samples": int(len(X_sample)),
            "encoded_features": int(X_enc.shape[1]),
            "class_index_used": int(class_idx),
            "waterfall_warning": waterfall_warning,
            "dependence_warning": dependence_warning,
        }
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
