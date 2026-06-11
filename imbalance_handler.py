"""
imbalance_handler.py
Utilities xử lý mất cân bằng lớp cho EcoPredict Carbon.

Mục tiêu:
- Không để lớp phát thải cao bị bỏ sót chỉ vì số mẫu ít.
- Ưu tiên F1-macro, Balanced Accuracy và Recall lớp High thay vì chỉ Accuracy.
- Hỗ trợ SMOTE nếu cài imbalanced-learn; nếu chưa cài thì fallback an toàn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42


@dataclass
class ImbalanceReport:
    strategy: str
    before_distribution: dict[int, int]
    after_distribution: dict[int, int]
    note: str


def class_distribution(y: np.ndarray) -> dict[int, int]:
    """Trả về phân phối lớp dạng {class_id: count}."""
    y = np.asarray(y).astype(int)
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def compute_balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """Tính class_weight theo công thức balanced của scikit-learn.

    Không normalize về tổng 1 vì scikit-learn kỳ vọng weight là penalty tương đối.
    Lớp ít mẫu sẽ có weight lớn hơn để mô hình phạt sai nặng hơn.
    """
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_stratified_kfold(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """CV giữ tỷ lệ lớp trong từng fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _safe_smote_k_neighbors(y: np.ndarray, requested: int = 3) -> int:
    """Chọn k_neighbors an toàn cho SMOTE dựa trên lớp ít mẫu nhất."""
    counts = np.array(list(class_distribution(y).values()), dtype=int)
    if len(counts) == 0:
        return 1
    min_count = int(counts.min())
    # SMOTE cần n_minority > k_neighbors.
    return max(1, min(int(requested), min_count - 1))


def try_apply_smote(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
    k_neighbors: int = 3,
    sampling_strategy: str | dict[Any, int] = "not majority",
    verbose: bool = True,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray, ImbalanceReport]:
    """Áp dụng SMOTE nếu có imbalanced-learn.

    Hàm này dùng được cho dữ liệu đã tiền xử lý toàn số. Với dữ liệu raw có categorical,
    nên dùng imblearn Pipeline: preprocessor -> SMOTE -> model.
    """
    y_train = np.asarray(y_train).astype(int)
    before = class_distribution(y_train)
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except Exception as exc:
        report = ImbalanceReport(
            strategy="none",
            before_distribution=before,
            after_distribution=before,
            note=f"imbalanced-learn chưa cài hoặc import lỗi: {exc}. Fallback dùng class_weight.",
        )
        if verbose:
            print(report)
        return X_train, y_train, report

    k = _safe_smote_k_neighbors(y_train, requested=k_neighbors)
    if k < 1 or min(before.values()) < 2:
        report = ImbalanceReport(
            strategy="class_weight_only",
            before_distribution=before,
            after_distribution=before,
            note="Lớp thiểu số quá ít mẫu để SMOTE an toàn. Fallback dùng class_weight.",
        )
        if verbose:
            print(report)
        return X_train, y_train, report

    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        after = class_distribution(y_res)
        report = ImbalanceReport(
            strategy=f"SMOTE(k_neighbors={k}, sampling_strategy={sampling_strategy})",
            before_distribution=before,
            after_distribution=after,
            note=f"Generated {len(y_res) - len(y_train)} synthetic minority samples.",
        )
        if verbose:
            print("\n" + "=" * 72)
            print("CLASS IMBALANCE HANDLING")
            print("Before:", before)
            print("After: ", after)
            print(report.note)
            print("=" * 72 + "\n")
        if isinstance(X_train, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X_train.columns)
        return X_res, np.asarray(y_res).astype(int), report
    except Exception as exc:
        report = ImbalanceReport(
            strategy="class_weight_only",
            before_distribution=before,
            after_distribution=before,
            note=f"SMOTE failed: {exc}. Fallback dùng class_weight.",
        )
        if verbose:
            print(report)
        return X_train, y_train, report


def classification_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Tạo diagnostics nhấn mạnh lớp minority/High."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if label_names is None:
        label_names = ["Low", "Medium", "High"]
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names[: len(np.unique(np.concatenate([y_true, y_pred])))],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    high_recall = 0.0
    if cm.shape == (3, 3) and cm[2].sum() > 0:
        high_recall = float(cm[2, 2] / cm[2].sum())
    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "high_class_recall": high_recall,
        "high_class_warning": high_recall == 0.0,
    }
