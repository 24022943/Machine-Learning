"""
train_advanced_models.py
EcoPredict Carbon v7 training pipeline.

Chạy:
    python train_advanced_models.py

Nâng cấp chính:
- Multi-source data: Carbon Catalogue + OpenPCF + Open CEDA factors.
- Time-aware split thay vì random split.
- Classification + regression.
- Uncertainty interval dựa trên residual và ensemble regressors.
- OOD profile để hiển thị mềm trên web.
- Permutation importance và SHAP để giải thích mô hình.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

from carbon_utils import (
    RANDOM_STATE, TARGET_COL, FEATURE_COLS, LABEL_ORDER, LABEL_TO_NUM, LABEL_VI,
    load_all_sources, time_based_split, fit_label_thresholds, apply_carbon_labels,
    make_clf_pipeline, make_reg_pipeline, get_classification_models, get_regression_models,
    evaluate_classifier, evaluate_regressor, build_ood_profile, save_package, tune_high_threshold,
)
from imbalance_handler import (
    class_distribution,
    compute_balanced_class_weights,
    classification_diagnostics,
)
from sensitivity_analysis import generate_default_sensitivity_outputs

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
MOD = OUT / "models"
for p in [FIG, TAB, MOD]:
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()




def _get_preprocessed_frame(pipe, X: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Trích model cây và ma trận feature sau preprocessing để dùng cho SHAP."""
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
    X_frame = pd.DataFrame(X_trans, columns=feature_names)
    return model, X_frame


def _normalise_shap_values(raw_values: Any) -> list[np.ndarray]:
    """Chuẩn hóa output SHAP qua các phiên bản khác nhau."""
    if isinstance(raw_values, list):
        return [np.asarray(v) for v in raw_values]
    arr = np.asarray(raw_values)
    if arr.ndim == 3:
        # SHAP mới thường trả về (n_samples, n_features, n_classes)
        return [arr[:, :, i] for i in range(arr.shape[2])]
    return [arr]


def generate_shap_explanations(best_clf: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    """Tạo SHAP plots cho mô hình cây tốt nhất.

    Nếu môi trường chưa cài shap hoặc best model không phải tree model, hàm sẽ bỏ qua an toàn.
    """
    info: dict[str, Any] = {"status": "not_run"}
    try:
        import shap  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"Chưa cài shap hoặc import lỗi: {exc}"}

    try:
        sample_n = min(80, len(X_test))
        if sample_n <= 0:
            return {"status": "skipped", "reason": "X_test rỗng"}
        if len(X_test) > sample_n:
            pos = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=sample_n, replace=False)
            X_sample = X_test.iloc[pos].copy()
        else:
            X_sample = X_test.copy()

        model, X_enc = _get_preprocessed_frame(best_clf, X_sample)
        if not hasattr(model, "feature_importances_"):
            return {"status": "skipped", "reason": "Best classifier không phải tree model phù hợp với TreeExplainer"}

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X_enc)
        shap_values_list = _normalise_shap_values(raw_values)
        class_idx = min(2, len(shap_values_list) - 1)  # ưu tiên lớp High nếu có 3 lớp
        sv = np.asarray(shap_values_list[class_idx])

        # Mean absolute SHAP across classes for stable global ranking.
        abs_stack = np.stack([np.abs(np.asarray(v)) for v in shap_values_list], axis=2) if len(shap_values_list) > 1 else np.abs(sv)[:, :, None]
        mean_abs = abs_stack.mean(axis=(0, 2))
        shap_imp = (
            pd.DataFrame({"feature": X_enc.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_imp.to_csv(TAB / "shap_feature_importance.csv", index=False)

        top = shap_imp.head(15).sort_values("mean_abs_shap", ascending=True)
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
                values=sv[0],
                base_values=base_value,
                data=X_enc.iloc[0].values,
                feature_names=X_enc.columns.tolist(),
            )
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            savefig("model_shap_waterfall_first_sample.png")
        except Exception as exc:
            info["waterfall_warning"] = str(exc)

        # Dependence plot có thể chạy chậm trong một số môi trường Streamlit/Windows.
        # Có thể tạo đầy đủ bằng script riêng: python generate_shap_explanations.py
        info["dependence_note"] = "Dependence plot có thể sinh bằng generate_shap_explanations.py"


        info.update({"status": "ok", "sample_n": int(sample_n), "class_index_used": int(class_idx), "n_features_after_encoding": int(X_enc.shape[1])})
        return info
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def main() -> None:
    print("=" * 88)
    print("ECOPREDICT CARBON V7 - MULTI-SOURCE ML + LCA/ISO DECISION-SUPPORT PIPELINE")
    print("=" * 88)

    sources = load_all_sources("carbon_catalogue.csv", include_openpcf_in_training=True)
    df_full = sources["training"].copy()
    carbon = sources["carbon"]
    openpcf = sources["openpcf"]
    ceda = sources["ceda"]

    df_full = df_full.dropna(subset=[TARGET_COL]).copy()
    df_full = df_full[df_full[TARGET_COL] > 0].copy()
    # To keep local/Streamlit training practical, train on all Carbon Catalogue rows
    # plus a deterministic sample of OpenPCF rows. The full OpenPCF table remains
    # available in the app for benchmark and factor lookup.
    carbon_part = df_full[df_full["data_source"].eq("Carbon Catalogue")]
    other_part = df_full[~df_full["data_source"].eq("Carbon Catalogue")]
    if len(other_part) > 1000:
        other_part = other_part.sample(1000, random_state=RANDOM_STATE)
    df = pd.concat([carbon_part, other_part], ignore_index=True).copy()
    print(f"Training data: {df.shape}")
    print(f"Carbon Catalogue rows: {len(carbon)} | OpenPCF rows: {len(openpcf)} | Open CEDA factors: {len(ceda)}")
    print(df["data_source"].value_counts(dropna=False))

    train_df, test_df, test_year = time_based_split(df)
    thresholds = fit_label_thresholds(train_df)
    for part in [train_df, test_df]:
        part["carbon_label"] = apply_carbon_labels(part, thresholds)
        part["carbon_label_num"] = part["carbon_label"].map(LABEL_TO_NUM)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}, test marker/year = {test_year}")
    print("Label thresholds fitted on train only:", thresholds)
    print("Train label distribution:\n", train_df["carbon_label"].value_counts())
    print("Test label distribution:\n", test_df["carbon_label"].value_counts())

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    y_train = train_df["carbon_label_num"].values.astype(int)
    y_test = test_df["carbon_label_num"].values.astype(int)
    yreg_train = train_df[TARGET_COL].values.astype(float)
    yreg_test = test_df[TARGET_COL].values.astype(float)

    # ---------------------------- Class imbalance diagnostics ----------------------------
    train_class_dist = class_distribution(y_train)
    test_class_dist = class_distribution(y_test)
    class_weights = compute_balanced_class_weights(y_train)
    min_class_count = min(train_class_dist.values()) if train_class_dist else 0
    smote_k_neighbors = max(1, min(3, min_class_count - 1)) if min_class_count >= 2 else 1
    use_smote_for_core_models = min_class_count >= 2
    imbalance_report = {
        "train_distribution": train_class_dist,
        "test_distribution": test_class_dist,
        "class_weights": class_weights,
        "use_smote_for_core_models": bool(use_smote_for_core_models),
        "smote_k_neighbors": int(smote_k_neighbors),
        "selection_metric": "f1_macro + balanced_accuracy; accuracy chỉ dùng tham khảo",
    }
    print("\nClass imbalance report:", json.dumps(imbalance_report, ensure_ascii=False, indent=2))

    # ---------------------------- Classification ----------------------------
    clf_rows = []
    clf_models = {}
    for name, model in get_classification_models().items():
        print(f"\nTraining classifier: {name}")
        use_smote = bool(use_smote_for_core_models and name not in {"Dummy Baseline"})
        pipe = make_clf_pipeline(model, use_smote=use_smote, smote_k_neighbors=smote_k_neighbors)
        pipe.fit(X_train, y_train)
        clf_models[name] = pipe
        metrics = evaluate_classifier(pipe, X_test, y_test)
        row = {"model": name, **{f"test_{k}": float(v) for k, v in metrics.items()}}
        clf_rows.append(row)
        print(row)
    clf_table = pd.DataFrame(clf_rows).sort_values(["test_f1_macro", "test_balanced_accuracy"], ascending=False)
    clf_table.to_csv(TAB / "classification_metrics.csv", index=False)
    best_clf_name = str(clf_table.iloc[0]["model"])
    best_clf = clf_models[best_clf_name]
    print("Best classifier:", best_clf_name)

    # Threshold tuning: không thay đổi probability, chỉ điều chỉnh nhãn cuối cùng để giảm bỏ sót lớp High.
    threshold_tuning_info: dict[str, float] = {}
    if best_clf_name != "Dummy Baseline":
        tuned_clf, threshold_tuning_info = tune_high_threshold(best_clf, X_train, y_train)
        best_clf = tuned_clf
        tuned_metrics = evaluate_classifier(best_clf, X_test, y_test)
        tuned_row = {"model": f"{best_clf_name} + High-threshold tuned", **{f"test_{k}": float(v) for k, v in tuned_metrics.items()}}
        clf_table = pd.concat([clf_table, pd.DataFrame([tuned_row])], ignore_index=True)
        clf_table.to_csv(TAB / "classification_metrics.csv", index=False)
        best_clf_name = f"{best_clf_name} + High-threshold tuned"
        print("Threshold tuning:", threshold_tuning_info)
        print("Final classifier:", best_clf_name)

    # ---------------------------- Regression ----------------------------
    reg_rows = []
    reg_models = {}
    for name, model in get_regression_models().items():
        print(f"\nTraining regressor: {name}")
        pipe = make_reg_pipeline(model)
        pipe.fit(X_train, yreg_train)
        reg_models[name] = pipe
        metrics = evaluate_regressor(pipe, X_test, yreg_test)
        row = {"model": name, **{f"test_{k}": float(v) for k, v in metrics.items()}}
        reg_rows.append(row)
        print(row)
    reg_table = pd.DataFrame(reg_rows)
    max_reasonable_mae = float(max(np.nanmax(yreg_train), np.nanmax(yreg_test)) * 2)
    reg_table["_valid_for_selection"] = (
        np.isfinite(reg_table["test_mae"])
        & np.isfinite(reg_table["test_rmse"])
        & np.isfinite(reg_table["test_r2"])
        & (reg_table["test_mae"] < max_reasonable_mae)
        & (reg_table["test_r2"] > -10)
    )
    reg_table = reg_table.sort_values(["_valid_for_selection", "test_median_ape_pct", "test_rmse"], ascending=[False, True, True])
    reg_table.to_csv(TAB / "regression_metrics.csv", index=False)
    best_reg_name = str(reg_table.iloc[0]["model"])
    best_reg = reg_models[best_reg_name]
    print("Best regressor:", best_reg_name)

    # ---------------------------- Figures ----------------------------
    plt.figure(figsize=(9, 5))
    plot_df = clf_table.sort_values("test_f1_macro", ascending=True)
    plt.barh(plot_df["model"], plot_df["test_f1_macro"], color="#047857")
    plt.xlabel("F1-macro trên tập kiểm tra theo thời gian")
    plt.title("So sánh mô hình phân loại")
    savefig("model_classification_f1_comparison.png")

    plt.figure(figsize=(9, 5))
    plot_df = reg_table.sort_values("test_median_ape_pct", ascending=False)
    plt.barh(plot_df["model"], plot_df["test_median_ape_pct"], color="#10b981")
    plt.xlabel("Median APE (%) - càng thấp càng tốt")
    plt.title("So sánh mô hình hồi quy")
    savefig("model_regression_median_ape_comparison.png")

    y_pred = best_clf.predict(X_test)
    diagnostics = classification_diagnostics(y_test, y_pred, label_names=[LABEL_VI[x] for x in LABEL_ORDER])
    (TAB / "classification_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    if diagnostics.get("high_class_warning"):
        print("⚠️ Warning: recall lớp phát thải cao = 0. Cần kiểm tra class imbalance/threshold tuning.")

    plt.figure(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[LABEL_VI[x] for x in LABEL_ORDER], cmap="Greens", values_format="d")
    plt.title(f"Confusion Matrix - {best_clf_name}")
    savefig("model_confusion_matrix.png")

    try:
        proba = best_clf.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        plt.figure(figsize=(6.8, 5.2))
        for i, label in enumerate(LABEL_ORDER):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            plt.plot(fpr, tpr, label=f"{LABEL_VI[label]} AUC={auc(fpr,tpr):.3f}")
        plt.plot([0, 1], [0, 1], "--", color="#94a3b8")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve theo từng lớp")
        plt.legend()
        savefig("model_roc_curve.png")
    except Exception as exc:
        print("ROC skipped:", exc)

    reg_pred = np.asarray(best_reg.predict(X_test), dtype=float)
    reg_pred = np.nan_to_num(reg_pred, nan=0.0, posinf=1e9, neginf=0.0)
    reg_pred = np.clip(reg_pred, 0.0, 1e9)
    residuals = yreg_test - reg_pred
    residual_abs_q = {
        "p10": float(np.quantile(np.abs(residuals), 0.10)),
        "p50": float(np.quantile(np.abs(residuals), 0.50)),
        "p90": float(np.quantile(np.abs(residuals), 0.90)),
    }

    plt.figure(figsize=(6, 5))
    plt.scatter(yreg_test, reg_pred, s=12, alpha=0.55, color="#047857")
    lim_min = max(min(float(np.nanmin(yreg_test)), float(np.nanmin(reg_pred))), 1e-6)
    lim_max = max(float(np.nanmax(yreg_test)), float(np.nanmax(reg_pred)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#334155")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("PCF thực tế (log)")
    plt.ylabel("PCF dự đoán (log)")
    plt.title(f"Actual vs Predicted - {best_reg_name}")
    savefig("model_regression_actual_vs_predicted.png")

    plt.figure(figsize=(7, 4.6))
    plt.scatter(np.maximum(reg_pred, 1e-6), residuals, s=12, alpha=0.55, color="#64748b")
    plt.axhline(0, linestyle="--", color="#ef4444")
    plt.xscale("log")
    plt.xlabel("PCF dự đoán (log)")
    plt.ylabel("Residual = Actual - Predicted")
    plt.title("Residuals vs Fitted")
    savefig("model_regression_residuals.png")

    plt.figure(figsize=(7, 4.6))
    sns.histplot(residuals, bins=45, kde=True, color="#10b981")
    plt.title("Phân phối phần dư hồi quy")
    plt.xlabel("Residual")
    savefig("model_regression_residual_distribution.png")

    # Permutation importance on a capped test sample to keep runtime stable.
    try:
        sample_n = min(250, len(X_test))
        if len(X_test) > sample_n:
            pos = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=sample_n, replace=False)
            X_imp = X_test.iloc[pos]
            y_imp = y_test[pos]
        else:
            X_imp = X_test
            y_imp = y_test
        perm = permutation_importance(best_clf, X_imp, y_imp, n_repeats=3, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=1)
        imp = pd.DataFrame({"feature": FEATURE_COLS, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv(TAB / "permutation_importance_classifier.csv", index=False)
        top = imp.head(12).sort_values("importance_mean", ascending=True)
        plt.figure(figsize=(8.5, 5.4))
        plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#047857", alpha=0.9)
        plt.xlabel("Decrease in F1-macro after permutation")
        plt.title("Permutation Importance - mô hình phân loại tốt nhất")
        savefig("model_permutation_importance.png")
    except Exception as exc:
        print("Permutation importance skipped:", exc)


    # SHAP explanations: professional XAI layer for tree-based best classifier.
    shap_info = generate_shap_explanations(best_clf, X_test, y_test)
    print("SHAP explanation status:", shap_info)

    # Sensitivity analysis: domain-rigor layer for LCA/PCF uncertainty discussion.
    sensitivity_info = {}
    try:
        baseline_for_sensitivity = float(np.nanmedian(yreg_test)) if len(yreg_test) else float(df[TARGET_COL].median())
        sensitivity_info = generate_default_sensitivity_outputs(baseline_for_sensitivity, output_dir=OUT)
        print("Sensitivity outputs:", sensitivity_info)
    except Exception as exc:
        sensitivity_info = {"status": "failed", "reason": str(exc)}
        print("Sensitivity analysis skipped:", exc)

    summary = {
        "n_training_rows": int(len(df)),
        "n_carbon_catalogue_rows": int(len(carbon)),
        "n_openpcf_rows": int(len(openpcf)),
        "n_open_ceda_factors": int(len(ceda)),
        "data_sources": df["data_source"].value_counts().to_dict(),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "pcf_min": float(df[TARGET_COL].min()),
        "pcf_median": float(df[TARGET_COL].median()),
        "pcf_mean": float(df[TARGET_COL].mean()),
        "pcf_max": float(df[TARGET_COL].max()),
    }
    pd.Series(summary).to_csv(TAB / "data_source_summary.csv", header=["value"])

    metadata = {
        "project": "EcoPredict Carbon",
        "version": "v7_lca_iso_decision_support",
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "label_order": LABEL_ORDER,
        "label_thresholds_train_only": thresholds,
        "split_strategy": f"time-aware split, test marker/year = {test_year}",
        "best_classifier_name": best_clf_name,
        "best_regressor_name": best_reg_name,
        "residual_abs_quantiles": residual_abs_q,
        "ood_profile": build_ood_profile(train_df),
        "data_summary": summary,
        "classification_metrics": clf_table.to_dict(orient="records"),
        "regression_metrics": reg_table.to_dict(orient="records"),
        "shap_explanation": shap_info,
        "class_imbalance_report": imbalance_report,
        "threshold_tuning": threshold_tuning_info,
        "classification_diagnostics": diagnostics,
        "sensitivity_analysis": sensitivity_info,
        "hyperparameter_tuning_note": "Chạy python hyperparameter_tuning.py để sinh bảng/biểu đồ GridSearchCV trong outputs.",
        "ml_limitations_note": "Metric cao cần hiểu trong phạm vi dữ liệu chuẩn hóa; cần hold-out theo sản phẩm/quốc gia/thời gian để xác nhận tổng quát hóa.",
        "disclaimer": "Prototype hỗ trợ ra quyết định PCF/LCA; không phải chứng nhận ISO/EPD chính thức.",
    }
    (TAB / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    package = {
        "classifier": best_clf,
        "regressor": best_reg,
        "classifiers": {k: v for k, v in clf_models.items() if k != "Dummy Baseline"},
        "regressors": {k: v for k, v in reg_models.items() if k != "Dummy Median"},
        "reference_data": df,
        "carbon_data": carbon,
        "openpcf_data": openpcf,
        "ceda_data": ceda,
        "train_data": train_df,
        "test_data": test_df,
        "metadata": metadata,
    }
    save_package(package, MOD / "ecopredict_model_package.joblib", also_root=True)
    print("\n✅ Training completed.")
    print("Saved model to outputs/models/ecopredict_model_package.joblib and ecopredict_model_package.joblib")
    print("Saved tables to outputs/tables and figures to outputs/figures")


if __name__ == "__main__":
    main()
