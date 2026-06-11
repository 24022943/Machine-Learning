"""
hyperparameter_tuning.py
EcoPredict Carbon - GridSearchCV tuning for final report.

Chạy:
    python hyperparameter_tuning.py

Output:
    outputs/tables/hyperparameter_tuning_results.csv
    outputs/figures/hyperparameter_tuning_comparison.png
"""
from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

from carbon_utils import (
    RANDOM_STATE, TARGET_COL, FEATURE_COLS,
    load_all_sources, time_based_split, fit_label_thresholds, apply_carbon_labels,
    make_clf_pipeline, make_reg_pipeline, evaluate_classifier, evaluate_regressor,
)
from imbalance_handler import class_distribution

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 80)
    print("ECOPREDICT CARBON - HYPERPARAMETER TUNING")
    print("=" * 80)

    sources = load_all_sources("carbon_catalogue.csv", include_openpcf_in_training=True)
    df_full = sources["training"].dropna(subset=[TARGET_COL]).copy()
    df_full = df_full[df_full[TARGET_COL] > 0].copy()

    carbon_part = df_full[df_full["data_source"].eq("Carbon Catalogue")]
    other_part = df_full[~df_full["data_source"].eq("Carbon Catalogue")]
    if len(carbon_part) > 400:
        carbon_part = carbon_part.sample(400, random_state=RANDOM_STATE)
    if len(other_part) > 800:
        other_part = other_part.sample(800, random_state=RANDOM_STATE)
    df = pd.concat([carbon_part, other_part], ignore_index=True)

    train_df, test_df, test_marker = time_based_split(df)
    thresholds = fit_label_thresholds(train_df)
    for part in [train_df, test_df]:
        part["carbon_label"] = apply_carbon_labels(part, thresholds)
        part["carbon_label_num"] = part["carbon_label"].map({"Low": 0, "Medium": 1, "High": 2})

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    y_train = train_df["carbon_label_num"].values.astype(int)
    y_test = test_df["carbon_label_num"].values.astype(int)
    yreg_train = train_df[TARGET_COL].values.astype(float)
    yreg_test = test_df[TARGET_COL].values.astype(float)

    rows: list[dict[str, object]] = []

    train_dist = class_distribution(y_train)
    min_class_count = min(train_dist.values()) if train_dist else 0
    smote_k = max(1, min(3, min_class_count - 1)) if min_class_count >= 2 else 1
    use_smote = min_class_count >= 2

    default_clf = make_clf_pipeline(RandomForestClassifier(
        n_estimators=80, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1,
    ), use_smote=use_smote, smote_k_neighbors=smote_k)
    default_clf.fit(X_train, y_train)
    rows.append({"task": "classification", "model": "Random Forest default", **evaluate_classifier(default_clf, X_test, y_test)})

    clf_grid = {
        "model__n_estimators": [80, 140],
        "model__max_depth": [8, 14, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt"],
    }
    clf_pipe = make_clf_pipeline(RandomForestClassifier(
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1,
    ), use_smote=use_smote, smote_k_neighbors=smote_k)
    clf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    clf_search = GridSearchCV(
        clf_pipe, clf_grid, cv=clf_cv, scoring="f1_macro", n_jobs=1, verbose=1,
        return_train_score=True,
    )
    clf_search.fit(X_train, y_train)
    rows.append({
        "task": "classification", "model": "Random Forest tuned",
        "best_params": str(clf_search.best_params_), "cv_best_score": float(clf_search.best_score_),
        **evaluate_classifier(clf_search.best_estimator_, X_test, y_test),
    })

    default_reg = make_reg_pipeline(TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=80, min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        func=np.log1p, inverse_func=np.expm1,
    ))
    default_reg.fit(X_train, yreg_train)
    rows.append({"task": "regression", "model": "Random Forest regressor default", **evaluate_regressor(default_reg, X_test, yreg_test)})

    reg_grid = {
        "model__regressor__n_estimators": [60, 100],
        "model__regressor__max_depth": [10],
        "model__regressor__min_samples_leaf": [1, 2],
        "model__regressor__max_features": ["sqrt"],
    }
    reg_pipe = make_reg_pipeline(TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        func=np.log1p, inverse_func=np.expm1,
    ))
    reg_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    reg_search = GridSearchCV(
        reg_pipe, reg_grid, cv=reg_cv, scoring="neg_mean_absolute_error", n_jobs=1, verbose=1,
        return_train_score=True,
    )
    reg_search.fit(X_train, yreg_train)
    rows.append({
        "task": "regression", "model": "Random Forest regressor tuned",
        "best_params": str(reg_search.best_params_), "cv_best_score": float(reg_search.best_score_),
        **evaluate_regressor(reg_search.best_estimator_, X_test, yreg_test),
    })

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "hyperparameter_tuning_results.csv", index=False)

    plot_df = out.copy()
    plot_df["score_for_plot"] = np.where(
        plot_df["task"].eq("classification"),
        plot_df.get("f1_macro", np.nan),
        -plot_df.get("median_ape_pct", np.nan),
    )
    plt.figure(figsize=(9, 5))
    labels = plot_df["model"].astype(str).str.replace(" Random Forest", "RF", regex=False)
    plt.barh(labels, plot_df["score_for_plot"], color="#047857")
    plt.title("Default vs GridSearchCV tuned models")
    plt.xlabel("F1-macro hoặc -Median APE")
    plt.tight_layout()
    plt.savefig(FIG / "hyperparameter_tuning_comparison.png", bbox_inches="tight")
    plt.close()

    print(out.to_string(index=False))
    print("Saved tuning outputs to outputs/tables and outputs/figures")


if __name__ == "__main__":
    main()
