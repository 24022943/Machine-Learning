"""
carbon_utils.py
Core utilities for EcoPredict Carbon Streamlit package.

Định hướng hệ thống:
- PCF là lõi: dự báo Product Carbon Footprint theo sản phẩm.
- LCA là nền khoa học: tính PCF bottom-up = Σ(activity data × emission factor).
- ML là lớp hỗ trợ: dự báo, hiệu chỉnh sai số, benchmark, uncertainty, OOD và giải thích yếu tố.
- ISO/EPD là định hướng minh bạch, chưa phải chứng nhận chính thức.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math
import re
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

RANDOM_STATE = 42
DATA_PATH = Path("carbon_catalogue.csv")
OPEN_PCF_PATHS = [Path("data/open_pcf_terralytiq.csv"), Path("Open-PCFs-by-Terralytiq.csv")]
OPEN_CEDA_PATHS = [Path("data/open_ceda_2024.csv"), Path("Open CEDA 2024 by Watershed(Open CEDA).csv")]
MODEL_PATH = Path("outputs/models/ecopredict_model_package.joblib")
ROOT_MODEL_PATH = Path("ecopredict_model_package.joblib")


class ThresholdTunedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper điều chỉnh ngưỡng ra quyết định cho bài toán mất cân bằng lớp.

    predict_proba giữ nguyên xác suất của base estimator. predict() ưu tiên nhận diện
    lớp High nếu xác suất High vượt `high_threshold`. Mục tiêu là giảm rủi ro bỏ sót
    sản phẩm phát thải cao trong demo/research prototype.
    """

    def __init__(self, base_estimator: Any, high_threshold: float = 0.20, medium_threshold: float | None = None):
        self.base_estimator = base_estimator
        self.high_threshold = float(high_threshold)
        self.medium_threshold = None if medium_threshold is None else float(medium_threshold)
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ThresholdTunedClassifier":
        self.base_estimator.fit(X, y)
        if hasattr(self.base_estimator, "classes_"):
            self.classes_ = self.base_estimator.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base_estimator.predict_proba(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        pred = np.asarray(np.argmax(proba, axis=1), dtype=int)
        if proba.shape[1] >= 3:
            high_mask = proba[:, 2] >= self.high_threshold
            pred[high_mask] = 2
        if self.medium_threshold is not None and proba.shape[1] >= 2:
            med_mask = (pred != 2) & (proba[:, 1] >= self.medium_threshold)
            pred[med_mask] = 1
        return pred


def tune_high_threshold(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    candidate_thresholds: Iterable[float] | None = None,
) -> tuple[ThresholdTunedClassifier, dict[str, float]]:
    """Tìm ngưỡng High theo F1-macro trên validation/train nội bộ.

    Không dùng test set để tune. Nếu không có đủ xác suất 3 lớp, trả về wrapper mặc định.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.03, 0.50, 20)
    y_val = np.asarray(y_val).astype(int)
    try:
        proba = np.asarray(estimator.predict_proba(X_val), dtype=float)
    except Exception:
        wrapper = ThresholdTunedClassifier(estimator, high_threshold=0.20)
        return wrapper, {"high_threshold": 0.20, "train_f1_macro": np.nan, "train_high_recall": np.nan}
    if proba.ndim != 2 or proba.shape[1] < 3:
        wrapper = ThresholdTunedClassifier(estimator, high_threshold=0.20)
        return wrapper, {"high_threshold": 0.20, "train_f1_macro": np.nan, "train_high_recall": np.nan}

    best = {"high_threshold": 0.20, "train_f1_macro": -1.0, "train_high_recall": 0.0}
    for thr in candidate_thresholds:
        pred = np.argmax(proba, axis=1).astype(int)
        pred[proba[:, 2] >= float(thr)] = 2
        score = float(f1_score(y_val, pred, average="macro", zero_division=0))
        high_mask = y_val == 2
        high_recall = float((pred[high_mask] == 2).mean()) if high_mask.any() else 0.0
        # Ưu tiên F1-macro, sau đó recall High để tránh bỏ sót lớp phát thải cao.
        if (score > best["train_f1_macro"]) or (np.isclose(score, best["train_f1_macro"]) and high_recall > best["train_high_recall"]):
            best = {"high_threshold": float(thr), "train_f1_macro": score, "train_high_recall": high_recall}
    return ThresholdTunedClassifier(estimator, high_threshold=best["high_threshold"]), best
TARGET_COL = "pcf_kg_co2e"
LABEL_ORDER = ["Low", "Medium", "High"]
LABEL_TO_NUM = {v: i for i, v in enumerate(LABEL_ORDER)}
NUM_TO_LABEL = {i: v for v, i in LABEL_TO_NUM.items()}
LABEL_VI = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}

RAW_TO_CANONICAL = {
    "Year of reporting": "year",
    "*Stage-level CO2e available": "stage_level_available",
    "Product name (and functional unit)": "product_name",
    "Product detail": "product_detail",
    "Company": "company",
    "Country (where company is incorporated)": "country",
    "Company's GICS Industry Group": "industry_group",
    "Company's GICS Industry": "industry",
    "*Company's sector": "company_sector",
    "Product weight (kg)": "product_weight_kg",
    "*Source for product weight": "weight_source",
    "Product's carbon footprint (PCF, kg CO2e)": TARGET_COL,
    "*Carbon intensity": "carbon_intensity",
    "Protocol used for PCF": "protocol",
    "Relative change in PCF vs previous": "relative_change_pcf",
    "Company-reported reason for change": "change_reason_text",
    "*Change reason category": "change_reason_category",
    "*%Upstream estimated from %Operations": "upstream_estimated_from_operations",
    "*Upstream CO2e (fraction of total PCF)": "upstream_frac",
    "*Operations CO2e (fraction of total PCF)": "operations_frac",
    "*Downstream CO2e (fraction of total PCF)": "downstream_frac",
    "*Transport CO2e (fraction of total PCF)": "transport_frac",
    "*EndOfLife CO2e (fraction of total PCF)": "end_of_life_frac",
    "*Adjustments to raw data (if any)": "raw_adjustments",
}

NUMERIC_FEATURES = [
    "year", "year_offset", "future_year_gap",
    "product_weight_kg", "product_weight_log",
    "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac",
    "lifecycle_fraction_sum", "lifecycle_balance_std", "lifecycle_max_share", "lifecycle_min_share",
    "upstream_x_weight_log", "operations_x_weight_log", "downstream_x_weight_log", "transport_x_weight_log",
    "product_name_length", "product_detail_length",
    "is_weight_estimated", "has_stage_data", "has_transport_data", "has_eol_data",
    "openpcf_factor_kgco2e_per_kg", "ceda_factor_kgco2e_per_usd",
    "lca_proxy_pcf", "ceda_proxy_pcf",
    "renewable_energy_pct", "material_reduction_pct", "transport_improvement_pct",
]
CATEGORICAL_FEATURES = [
    "country", "industry_group", "industry", "company_sector",
    "stage_level_available", "weight_source", "protocol_simple",
    "dominant_stage", "weight_category", "data_source", "system_boundary", "functional_unit_type",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_NAME_VI = {
    "year": "Năm báo cáo/kịch bản",
    "year_offset": "Khoảng cách so với năm đầu dữ liệu",
    "future_year_gap": "Khoảng cách năm kịch bản tương lai",
    "product_weight_kg": "Khối lượng sản phẩm",
    "product_weight_log": "Khối lượng sản phẩm (log)",
    "upstream_frac": "Tỷ trọng đầu vào",
    "operations_frac": "Tỷ trọng sản xuất/vận hành",
    "downstream_frac": "Tỷ trọng đầu ra",
    "transport_frac": "Tỷ trọng vận chuyển",
    "end_of_life_frac": "Tỷ trọng cuối vòng đời",
    "lifecycle_fraction_sum": "Tổng tỷ trọng vòng đời",
    "lifecycle_balance_std": "Độ lệch giữa các giai đoạn vòng đời",
    "lifecycle_max_share": "Tỷ trọng vòng đời lớn nhất",
    "lifecycle_min_share": "Tỷ trọng vòng đời nhỏ nhất",
    "upstream_x_weight_log": "Tương tác đầu vào và khối lượng",
    "operations_x_weight_log": "Tương tác vận hành và khối lượng",
    "downstream_x_weight_log": "Tương tác đầu ra và khối lượng",
    "transport_x_weight_log": "Tương tác vận chuyển và khối lượng",
    "product_name_length": "Độ dài tên sản phẩm",
    "product_detail_length": "Độ dài mô tả sản phẩm",
    "is_weight_estimated": "Khối lượng có tính ước lượng",
    "has_stage_data": "Có dữ liệu theo giai đoạn",
    "has_transport_data": "Có dữ liệu vận chuyển",
    "has_eol_data": "Có dữ liệu cuối vòng đời",
    "openpcf_factor_kgco2e_per_kg": "Hệ số OpenPCF theo vật liệu/sản phẩm",
    "ceda_factor_kgco2e_per_usd": "Hệ số Open CEDA theo quốc gia/ngành",
    "lca_proxy_pcf": "PCF proxy từ LCA bottom-up",
    "ceda_proxy_pcf": "PCF proxy từ Open CEDA",
    "renewable_energy_pct": "Kịch bản tăng điện tái tạo",
    "material_reduction_pct": "Kịch bản giảm vật liệu",
    "transport_improvement_pct": "Kịch bản cải thiện vận chuyển",
    "country": "Quốc gia/khu vực",
    "industry_group": "Nhóm ngành",
    "industry": "Ngành/sản phẩm",
    "company_sector": "Sector công ty",
    "stage_level_available": "Có dữ liệu stage-level",
    "weight_source": "Nguồn khối lượng",
    "protocol_simple": "Chuẩn PCF",
    "dominant_stage": "Giai đoạn phát thải chính",
    "weight_category": "Nhóm khối lượng",
    "data_source": "Nguồn dữ liệu",
    "system_boundary": "Ranh giới hệ thống",
    "functional_unit_type": "Loại đơn vị chức năng",
}

COUNTRY_ALIASES = {
    "USA": "United States Of America",
    "United States": "United States Of America",
    "United States of America": "United States Of America",
    "UK": "United Kingdom",
    "Viet Nam": "Vietnam",
    "Czech Republic": "Czechia",
}

DEMO_EMISSION_FACTORS = pd.DataFrame([
    {"stage": "Vật liệu", "activity_group": "Material", "activity_name": "Steel", "unit": "kg", "amount": 0.0, "emission_factor": 1.90, "source": "Demo EF", "quality": "Medium"},
    {"stage": "Vật liệu", "activity_group": "Material", "activity_name": "Aluminium", "unit": "kg", "amount": 0.0, "emission_factor": 8.60, "source": "Demo EF", "quality": "Medium"},
    {"stage": "Vật liệu", "activity_group": "Material", "activity_name": "Plastic", "unit": "kg", "amount": 0.0, "emission_factor": 2.50, "source": "Demo EF", "quality": "Medium"},
    {"stage": "Năng lượng", "activity_group": "Energy", "activity_name": "Electricity", "unit": "kWh", "amount": 0.0, "emission_factor": 0.45, "source": "Demo EF", "quality": "Medium"},
    {"stage": "Vận chuyển", "activity_group": "Transport", "activity_name": "Truck freight", "unit": "ton-km", "amount": 0.0, "emission_factor": 0.12, "source": "Demo EF", "quality": "Low"},
    {"stage": "Bao bì", "activity_group": "Packaging", "activity_name": "Paper/Cardboard", "unit": "kg", "amount": 0.0, "emission_factor": 0.90, "source": "Demo EF", "quality": "Medium"},
    {"stage": "Cuối vòng đời", "activity_group": "End-of-life", "activity_name": "Waste treatment", "unit": "kg", "amount": 0.0, "emission_factor": 0.25, "source": "Demo EF", "quality": "Low"},
])


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def read_csv_safely(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    last_error: Exception | None = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error if last_error else RuntimeError(f"Không đọc được file {path}")


def parse_numeric(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "not reported", "unknown", "-", "nan"}:
        return np.nan
    s = s.replace("\u00a0", "").replace(" ", "")
    # Decimal comma when there is no dot, thousands comma when dot exists or group size > 3.
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 3:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_fraction(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v / 100.0 if v > 1.5 else v
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "not reported", "unknown", "-", "nan"}:
        return np.nan
    if "included" in s.lower() or "not reported" in s.lower():
        return np.nan
    pct = "%" in s
    v = parse_numeric(s.replace("%", ""))
    if pd.isna(v):
        return np.nan
    return v / 100.0 if pct or v > 1.5 else v


def clean_text_value(x: Any, default: str = "Unknown") -> str:
    if pd.isna(x):
        return default
    s = str(x).strip()
    if not s or s.lower() in {"n/a", "na", "none", "nan", "not reported"}:
        return default
    return s


def normalize_country(x: Any) -> str:
    s = clean_text_value(x, "Unknown")
    return COUNTRY_ALIASES.get(s, s)


def simplify_protocol(x: Any) -> str:
    s = clean_text_value(x, "Unknown")
    sl = s.lower()
    if "openpcf" in sl or "terralytiq" in sl:
        return "OpenPCF"
    if "ghg" in sl:
        return "GHG Protocol"
    if "iso" in sl:
        return "ISO"
    if "pas" in sl:
        return "PAS 2050"
    if "unknown" in sl or "not" in sl:
        return "Unknown"
    return "Other"


def infer_industry_group(text: Any) -> str:
    s = clean_text_value(text, "").lower()
    rules = [
        ("alum|steel|iron|copper|nickel|cement|concrete|glass|ceramic|bauxite|metal", "Materials"),
        ("battery|circuit|magnet|electro|coil|semiconductor", "Technology Hardware & Equipment"),
        ("poly|plastic|resin|benzene|methanol|ammonia|acid|phenol|chemical|propane|butane", "Chemicals"),
        ("cotton|fabric|textile|polyester|nylon", "Textiles & Apparel"),
        ("paper|corrugated|cardboard|box|packaging", "Packaging"),
        ("diesel|oil|gas|fuel|petroleum", "Energy"),
        ("food|milk|grain|beef|poultry|crop", "Food, Beverage & Tobacco"),
        ("transport|truck|freight|shipping", "Transportation"),
    ]
    for pat, group in rules:
        if re.search(pat, s):
            return group
    return "Manufacturing"


def weight_category(w: float) -> str:
    if pd.isna(w) or w <= 0:
        return "Unknown"
    if w < 1:
        return "Very light (<1 kg)"
    if w < 10:
        return "Light (1-10 kg)"
    if w < 100:
        return "Medium (10-100 kg)"
    if w < 1000:
        return "Heavy (100-1000 kg)"
    return "Very heavy (>1000 kg)"


def load_carbon_catalogue(path: str | Path = DATA_PATH) -> pd.DataFrame:
    raw = read_csv_safely(path)
    df = raw.rename(columns={c: RAW_TO_CANONICAL.get(c, c) for c in raw.columns}).copy()

    required = ["year", TARGET_COL, "product_weight_kg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File Carbon Catalogue thiếu cột bắt buộc: {missing}")

    for col in ["year", "product_weight_kg", TARGET_COL, "carbon_intensity"]:
        if col in df.columns:
            df[col] = df[col].map(parse_numeric)
    for col in ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        df[col] = df[col].map(parse_fraction) if col in df.columns else np.nan

    text_cols = [
        "stage_level_available", "product_name", "product_detail", "company", "country",
        "industry_group", "industry", "company_sector", "weight_source", "protocol",
        "upstream_estimated_from_operations", "change_reason_category", "relative_change_pcf",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].map(clean_text_value)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df[df[TARGET_COL] > 0].copy()
    df["product_weight_kg"] = df["product_weight_kg"].fillna(df["product_weight_kg"].median()).clip(lower=0.0001)
    df["year"] = df["year"].fillna(df["year"].median()).astype(int)
    df["country"] = df["country"].map(normalize_country)
    df["protocol_simple"] = df["protocol"].map(simplify_protocol)
    df["data_source"] = "Carbon Catalogue"
    df["functional_unit_type"] = "Product reported unit"
    df["system_boundary"] = "Reported PCF boundary"
    return df.reset_index(drop=True)


def load_open_pcf(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else first_existing(OPEN_PCF_PATHS)
    if p is None or not p.exists():
        return pd.DataFrame()
    raw = read_csv_safely(p, header=None, dtype=str, engine="python")
    # In Terralytiq CSV, row 26 = common name, row 27 = HS code, row 28 = product description.
    if raw.shape[0] < 30:
        return pd.DataFrame()
    common = raw.iloc[26].copy()
    hs = raw.iloc[27].copy()
    desc = raw.iloc[28].copy()
    data_rows = raw.iloc[29:].copy()
    records: list[dict[str, Any]] = []
    for _, row in data_rows.iterrows():
        country = clean_text_value(row.iloc[1], "Unknown")
        if country == "Unknown":
            continue
        for j in range(2, raw.shape[1]):
            name = clean_text_value(common.iloc[j], "")
            if not name:
                continue
            value = parse_numeric(row.iloc[j])
            if pd.isna(value) or value <= 0:
                continue
            description = clean_text_value(desc.iloc[j], name)
            records.append({
                "country": normalize_country(country),
                "product_name": name,
                "product_detail": description,
                "hs_code": clean_text_value(hs.iloc[j], "Unknown"),
                "openpcf_factor_kgco2e_per_kg": float(value),
                "industry_group": infer_industry_group(name + " " + description),
                "industry": name.title(),
                "company_sector": infer_industry_group(name + " " + description),
                "data_source": "OpenPCF by Terralytiq",
                "year": 2025,
            })
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def load_open_ceda(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else first_existing(OPEN_CEDA_PATHS)
    if p is None or not p.exists():
        return pd.DataFrame()
    raw = read_csv_safely(p, sep=";", header=None, dtype=str, engine="python")
    header_row = None
    for i in range(raw.shape[0]):
        vals = [clean_text_value(v, "") for v in raw.iloc[i].tolist()]
        if "Country Code" in vals and "Country" in vals:
            header_row = i
            break
    if header_row is None or header_row == 0:
        return pd.DataFrame()
    sector_names = raw.iloc[header_row - 1].copy()
    codes = raw.iloc[header_row].copy()
    records: list[dict[str, Any]] = []
    for i in range(header_row + 1, raw.shape[0]):
        row = raw.iloc[i]
        country_code = clean_text_value(row.iloc[1], "")
        country = clean_text_value(row.iloc[2], "")
        unit = clean_text_value(row.iloc[3], "kgCO2e/US Dollar")
        if not country_code or not country:
            continue
        for j in range(4, raw.shape[1]):
            sector = clean_text_value(sector_names.iloc[j], "")
            code = clean_text_value(codes.iloc[j], "")
            if not sector:
                continue
            val = parse_numeric(row.iloc[j])
            if pd.isna(val) or val < 0:
                continue
            records.append({
                "country_code": country_code,
                "country": normalize_country(country),
                "unit": unit,
                "ceda_sector_code": code,
                "ceda_sector": sector,
                "ceda_factor_kgco2e_per_usd": float(val),
                "data_source": "Open CEDA 2024 by Watershed",
                "year": 2024,
            })
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def country_factor_lookup(ceda: pd.DataFrame) -> dict[str, float]:
    if ceda.empty:
        return {}
    return ceda.groupby("country")["ceda_factor_kgco2e_per_usd"].median().to_dict()


def product_factor_lookup(openpcf: pd.DataFrame) -> dict[str, float]:
    if openpcf.empty:
        return {}
    return openpcf.groupby("product_name")["openpcf_factor_kgco2e_per_kg"].median().to_dict()


def make_openpcf_training_rows(openpcf: pd.DataFrame, ceda: pd.DataFrame | None = None) -> pd.DataFrame:
    if openpcf.empty:
        return pd.DataFrame()
    out = openpcf.copy()
    out["product_weight_kg"] = 1.0
    out[TARGET_COL] = out["openpcf_factor_kgco2e_per_kg"].clip(lower=0.000001)
    out["company"] = "Terralytiq OpenPCF"
    out["stage_level_available"] = "Estimated"
    out["weight_source"] = "Functional unit 1 kg"
    out["protocol"] = "OpenPCF"
    out["protocol_simple"] = "OpenPCF"
    out["functional_unit_type"] = "1 kg material/product"
    out["system_boundary"] = "Cradle-to-gate factor"
    out["upstream_frac"] = 0.72
    out["operations_frac"] = 0.18
    out["downstream_frac"] = 0.10
    out["transport_frac"] = 0.0
    out["end_of_life_frac"] = 0.0
    if ceda is not None and not ceda.empty:
        lookup = country_factor_lookup(ceda)
        global_median = float(ceda["ceda_factor_kgco2e_per_usd"].median())
        out["ceda_factor_kgco2e_per_usd"] = out["country"].map(lookup).fillna(global_median)
    else:
        out["ceda_factor_kgco2e_per_usd"] = np.nan
    return out


def enrich_with_reference_factors(df: pd.DataFrame, openpcf: pd.DataFrame, ceda: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not openpcf.empty:
        country_mat = openpcf.groupby("country")["openpcf_factor_kgco2e_per_kg"].median().to_dict()
        industry_mat = openpcf.groupby("industry_group")["openpcf_factor_kgco2e_per_kg"].median().to_dict()
        global_mat = float(openpcf["openpcf_factor_kgco2e_per_kg"].median())
        out["openpcf_factor_kgco2e_per_kg"] = out.get("openpcf_factor_kgco2e_per_kg", np.nan)
        missing = out["openpcf_factor_kgco2e_per_kg"].isna()
        out.loc[missing, "openpcf_factor_kgco2e_per_kg"] = out.loc[missing, "industry_group"].map(industry_mat)
        missing = out["openpcf_factor_kgco2e_per_kg"].isna()
        out.loc[missing, "openpcf_factor_kgco2e_per_kg"] = out.loc[missing, "country"].map(country_mat)
        out["openpcf_factor_kgco2e_per_kg"] = out["openpcf_factor_kgco2e_per_kg"].fillna(global_mat)
    else:
        out["openpcf_factor_kgco2e_per_kg"] = out.get("openpcf_factor_kgco2e_per_kg", np.nan).fillna(1.0)

    if not ceda.empty:
        lookup = country_factor_lookup(ceda)
        global_ceda = float(ceda["ceda_factor_kgco2e_per_usd"].median())
        out["ceda_factor_kgco2e_per_usd"] = out.get("ceda_factor_kgco2e_per_usd", np.nan)
        out["ceda_factor_kgco2e_per_usd"] = out["ceda_factor_kgco2e_per_usd"].fillna(out["country"].map(lookup)).fillna(global_ceda)
    else:
        out["ceda_factor_kgco2e_per_usd"] = out.get("ceda_factor_kgco2e_per_usd", np.nan).fillna(0.5)
    return out


def load_all_sources(
    carbon_path: str | Path = DATA_PATH,
    include_openpcf_in_training: bool = True,
) -> dict[str, pd.DataFrame]:
    carbon = load_carbon_catalogue(carbon_path)
    openpcf = load_open_pcf()
    ceda = load_open_ceda()
    carbon = enrich_with_reference_factors(carbon, openpcf, ceda)
    pieces = [carbon]
    if include_openpcf_in_training and not openpcf.empty:
        open_train = make_openpcf_training_rows(openpcf, ceda)
        open_train = enrich_with_reference_factors(open_train, openpcf, ceda)
        pieces.append(open_train)
    training = pd.concat(pieces, ignore_index=True, sort=False)
    training = add_model_features(training)
    return {"carbon": carbon, "openpcf": openpcf, "ceda": ceda, "training": training}


def add_model_features(df: pd.DataFrame, reference_min_year: int | None = None, max_train_year: int | None = None) -> pd.DataFrame:
    out = df.copy()
    # Required text defaults.
    for c in ["product_name", "product_detail", "country", "industry_group", "industry", "company_sector", "stage_level_available", "weight_source", "protocol", "protocol_simple", "data_source", "system_boundary", "functional_unit_type"]:
        if c not in out.columns:
            out[c] = "Unknown"
        out[c] = out[c].map(clean_text_value)
    out["country"] = out["country"].map(normalize_country)
    out["protocol_simple"] = out["protocol_simple"].map(simplify_protocol)

    # Numeric defaults.
    for col in ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    main = ["upstream_frac", "operations_frac", "downstream_frac"]
    s = out[main].sum(axis=1).replace(0, np.nan)
    out.loc[s.notna(), main] = out.loc[s.notna(), main].div(s[s.notna()], axis=0)

    if "product_weight_kg" not in out.columns:
        out["product_weight_kg"] = 1.0
    out["product_weight_kg"] = pd.to_numeric(out["product_weight_kg"], errors="coerce").fillna(1.0).clip(lower=0.0001)
    out["product_weight_log"] = np.log1p(out["product_weight_kg"])
    if "year" not in out.columns:
        out["year"] = 2025
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(2025).astype(int)
    min_year = int(reference_min_year if reference_min_year is not None else out["year"].min())
    max_year = int(max_train_year if max_train_year is not None else out["year"].max())
    out["year_offset"] = out["year"] - min_year
    out["future_year_gap"] = np.maximum(out["year"] - max_year, 0)

    stages = out[["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]]
    out["lifecycle_fraction_sum"] = stages.sum(axis=1)
    out["lifecycle_balance_std"] = stages.std(axis=1)
    out["lifecycle_max_share"] = stages.max(axis=1)
    out["lifecycle_min_share"] = stages.min(axis=1)
    names = ["Upstream", "Operations", "Downstream", "Transport", "End-of-life"]
    out["dominant_stage"] = stages.values.argmax(axis=1)
    out["dominant_stage"] = out["dominant_stage"].map(lambda i: names[int(i)] if pd.notna(i) else "Unknown")

    out["upstream_x_weight_log"] = out["upstream_frac"] * out["product_weight_log"]
    out["operations_x_weight_log"] = out["operations_frac"] * out["product_weight_log"]
    out["downstream_x_weight_log"] = out["downstream_frac"] * out["product_weight_log"]
    out["transport_x_weight_log"] = out["transport_frac"] * out["product_weight_log"]
    out["product_name_length"] = out["product_name"].astype(str).str.len()
    out["product_detail_length"] = out["product_detail"].astype(str).str.len()
    out["is_weight_estimated"] = out["weight_source"].astype(str).str.lower().str.contains("estim|external|calculated|proxy", regex=True).astype(int)
    out["has_stage_data"] = out["stage_level_available"].astype(str).str.lower().str.contains("yes|estimated|available").astype(int)
    out["has_transport_data"] = (out["transport_frac"] > 0).astype(int)
    out["has_eol_data"] = (out["end_of_life_frac"] > 0).astype(int)
    out["weight_category"] = out["product_weight_kg"].map(weight_category)

    if "openpcf_factor_kgco2e_per_kg" not in out.columns:
        out["openpcf_factor_kgco2e_per_kg"] = 1.0
    if "ceda_factor_kgco2e_per_usd" not in out.columns:
        out["ceda_factor_kgco2e_per_usd"] = 0.5
    if "lca_proxy_pcf" not in out.columns:
        out["lca_proxy_pcf"] = np.nan
    out["openpcf_factor_kgco2e_per_kg"] = pd.to_numeric(out["openpcf_factor_kgco2e_per_kg"], errors="coerce").fillna(1.0).clip(lower=0.0)
    out["ceda_factor_kgco2e_per_usd"] = pd.to_numeric(out["ceda_factor_kgco2e_per_usd"], errors="coerce").fillna(0.5).clip(lower=0.0)
    out["lca_proxy_pcf"] = pd.to_numeric(out["lca_proxy_pcf"], errors="coerce")
    out["lca_proxy_pcf"] = out["lca_proxy_pcf"].fillna(out["product_weight_kg"] * out["openpcf_factor_kgco2e_per_kg"])
    if "ceda_proxy_pcf" not in out.columns:
        out["ceda_proxy_pcf"] = np.nan
    out["ceda_proxy_pcf"] = pd.to_numeric(out["ceda_proxy_pcf"], errors="coerce")
    out["ceda_proxy_pcf"] = out["ceda_proxy_pcf"].fillna(out["product_weight_kg"] * out["ceda_factor_kgco2e_per_usd"])
    for c in ["renewable_energy_pct", "material_reduction_pct", "transport_improvement_pct"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].map(clean_text_value) if c in out.columns else "Unknown"
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def time_based_split(df: pd.DataFrame, year_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Time-aware split.

    If the latest year dominates because OpenPCF is a large 2025 release, the function
    still keeps the chronological spirit but puts the first 80% of the latest-year rows
    into train and the remaining 20% into test. This avoids a situation where the model
    trains only on the 866-row Carbon Catalogue and cannot learn the new factor-based
    data structure.
    """
    d = df.sort_values([year_col, "data_source", "country", "industry"]).reset_index(drop=True).copy()
    years = sorted(d[year_col].dropna().unique())
    if len(years) >= 3:
        latest = int(years[-1])
        older = d[d[year_col] < latest].copy()
        latest_df = d[d[year_col] == latest].copy()
        if len(latest_df) > 0.55 * len(d) and len(latest_df) >= 200:
            cut_latest = int(len(latest_df) * 0.8)
            train = pd.concat([older, latest_df.iloc[:cut_latest]], ignore_index=True)
            test = latest_df.iloc[cut_latest:].copy()
            return train, test, latest
        if len(older) >= 100 and len(latest_df) >= 50:
            return older, latest_df, latest
    cut = int(len(d) * 0.8)
    train, test = d.iloc[:cut].copy(), d.iloc[cut:].copy()
    return train, test, int(test[year_col].min())


def fit_label_thresholds(train_df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, float]:
    q25 = float(train_df[target_col].quantile(0.25))
    q75 = float(train_df[target_col].quantile(0.75))
    if q75 <= q25:
        q75 = q25 * 1.5 + 1e-6
    return {"q25": q25, "q75": q75}


def apply_carbon_labels(df: pd.DataFrame, thresholds: dict[str, float], target_col: str = TARGET_COL) -> pd.Series:
    q25, q75 = thresholds["q25"], thresholds["q75"]
    labels = np.where(df[target_col] <= q25, "Low", np.where(df[target_col] <= q75, "Medium", "High"))
    return pd.Series(labels, index=df.index)


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot_encoder())])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def get_classification_models() -> dict[str, Any]:
    """Các mô hình phân loại Low/Medium/High.

    Dummy và Logistic dùng làm baseline; Random Forest/Extra Trees dùng làm mô hình cây tổ hợp chính.
    """
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=40,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=40,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def get_regression_models() -> dict[str, Any]:
    """Các mô hình hồi quy dự báo PCF kg CO2e.

    Target được log-transform để xử lý phân phối PCF lệch phải.
    """
    return {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Ridge log-target": TransformedTargetRegressor(
            regressor=Ridge(alpha=2.0),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest Regressor": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=40,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Extra Trees Regressor": TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(
                n_estimators=40,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }

def make_clf_pipeline(model: Any, use_smote: bool = False, smote_k_neighbors: int = 3) -> Pipeline:
    """Tạo pipeline phân loại.

    Nếu use_smote=True và đã cài imbalanced-learn, pipeline sẽ là:
    preprocessor -> SMOTE -> model.

    SMOTE chỉ chạy trong fit(); khi predict/predict_proba trên app, bước SMOTE tự
    được bỏ qua đúng cơ chế của imblearn Pipeline. Nếu chưa cài imbalanced-learn,
    hàm fallback về sklearn Pipeline và vẫn dùng class_weight trong model.
    """
    preprocessor = make_preprocessor()
    if use_smote:
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
            from imblearn.over_sampling import SMOTE  # type: ignore

            return ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=int(smote_k_neighbors), sampling_strategy="not majority")),
                ("model", model),
            ])
        except Exception:
            # Fallback an toàn để project vẫn chạy nếu imbalanced-learn chưa cài.
            pass
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def make_reg_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y_num: np.ndarray) -> dict[str, float]:
    pred = np.asarray(model.predict(X)).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_num, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_num, pred)),
        "f1_macro": float(f1_score(y_num, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_num, pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_num, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_num, pred, average="macro", zero_division=0)),
    }
    # Per-class recall/F1 giúp nhìn rõ việc mô hình có bỏ sót lớp phát thải cao hay không.
    recall_per_class = recall_score(y_num, pred, labels=[0, 1, 2], average=None, zero_division=0)
    f1_per_class = f1_score(y_num, pred, labels=[0, 1, 2], average=None, zero_division=0)
    for idx, label in enumerate(["low", "medium", "high"]):
        out[f"recall_{label}"] = float(recall_per_class[idx])
        out[f"f1_{label}"] = float(f1_per_class[idx])
    out["high_class_warning"] = float(out.get("recall_high", 0.0) == 0.0)
    try:
        proba = model.predict_proba(X)
        out["roc_auc_ovr"] = float(roc_auc_score(y_num, proba, multi_class="ovr"))
    except Exception:
        out["roc_auc_ovr"] = np.nan
    return out


def safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def median_ape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.median(np.abs((y_true - y_pred) / denom)) * 100)


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Phát hiện outlier bằng IQR.

    Đây là hàm hỗ trợ EDA/quality check, không tự động xóa dữ liệu vì PCF thường lệch phải
    và các giá trị cực trị có thể là thông tin quan trọng về ngành/sản phẩm.
    """
    if column not in df.columns or df.empty:
        return pd.DataFrame(columns=df.columns)
    values = pd.to_numeric(df[column], errors="coerce")
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return pd.DataFrame(columns=df.columns)
    lower = q1 - float(threshold) * iqr
    upper = q3 + float(threshold) * iqr
    return df.loc[(values < lower) | (values > upper)].copy()


def evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(model.predict(X), dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)
    pred = np.clip(pred, 0.0, 1e9)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "mape_pct": float(safe_mape(y, pred)),
        "median_ape_pct": float(median_ape(y, pred)),
    }


def build_ood_profile(train_df: pd.DataFrame) -> dict[str, Any]:
    numeric_stats = {}
    for c in NUMERIC_FEATURES:
        s = pd.to_numeric(train_df[c], errors="coerce")
        std = float(s.std()) if float(s.std()) > 1e-9 else 1.0
        numeric_stats[c] = {"mean": float(s.mean()), "std": std, "min": float(s.min()), "max": float(s.max())}
    cat_levels = {c: sorted(train_df[c].astype(str).dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}

    year_series = pd.to_numeric(train_df.get("year"), errors="coerce").dropna().astype(int)
    observed_years = sorted(year_series.unique().tolist()) if len(year_series) else []
    year_counts = {int(k): int(v) for k, v in year_series.value_counts().to_dict().items()} if len(year_series) else {}

    return {
        "numeric_stats": numeric_stats,
        "cat_levels": cat_levels,
        "min_year": int(year_series.min()) if len(year_series) else int(train_df["year"].min()),
        "max_year": int(year_series.max()) if len(year_series) else int(train_df["year"].max()),
        "observed_years": observed_years,
        "year_counts": year_counts,
    }


def check_ood(input_df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """Đánh giá mức tin cậy kịch bản theo cách mềm nhưng không làm mọi năm tương lai đều "Cao".

    Ý nghĩa đúng:
    - Đây KHÔNG phải nhãn mà mô hình ML học trực tiếp.
    - Đây là lớp kiểm tra phạm vi dữ liệu: input có gần dữ liệu tham chiếu hay không.
    - ML dạng bảng dự báo PCF theo đặc trưng sản phẩm/ngành/quốc gia/factor.
    - Phần tương lai được xem là mô phỏng kịch bản; với năm càng xa vùng dữ liệu gốc, mức tin cậy phải giảm.
    """
    detail: list[str] = []
    max_z = 0.0
    score = 0.88
    score_cap = 0.98

    scenario_control_features = {
        "renewable_energy_pct",
        "material_reduction_pct",
        "transport_improvement_pct",
    }
    binary_indicator_features = {
        "is_weight_estimated",
        "has_stage_data",
        "has_transport_data",
        "has_eol_data",
    }
    soft_scenario_categories = {
        "stage_level_available",
        "weight_source",
        "data_source",
        "system_boundary",
        "functional_unit_type",
    }

    min_year = int(profile.get("min_year", profile.get("numeric_stats", {}).get("year", {}).get("min", 2013)))
    max_year = int(profile.get("max_year", profile.get("numeric_stats", {}).get("year", {}).get("max", 2025)))
    observed_years = [int(y) for y in profile.get("observed_years", []) if pd.notna(y)]

    # 1) Xử lý năm riêng.
    # Dữ liệu gốc chủ yếu là Carbon Catalogue 2013–2017 và OpenPCF/OpenCEDA mới hơn nhưng không liên tục theo thời gian.
    # Vì vậy 2026–2050 là mô phỏng kịch bản tương lai, không được mặc định là "Cao".
    if "year" in input_df.columns:
        year_val = int(pd.to_numeric(input_df["year"], errors="coerce").iloc[0])

        if year_val < min_year:
            score -= 0.25
            score_cap = min(score_cap, 0.58)
            detail.append(f"Năm {year_val} sớm hơn vùng dữ liệu tham chiếu chính ({min_year}–{max_year}).")

        elif year_val <= max_year:
            # Nếu nằm trong khoảng min-max nhưng không gần năm quan sát nào thì xem là nội suy theo khoảng trống dữ liệu.
            if observed_years:
                nearest_gap = min(abs(year_val - y) for y in observed_years)
                if nearest_gap == 0:
                    detail.append(f"Năm {year_val} thuộc vùng dữ liệu đã có mẫu tham chiếu.")
                elif nearest_gap <= 1:
                    score -= 0.04
                    score_cap = min(score_cap, 0.84)
                    detail.append(f"Năm {year_val} gần năm có dữ liệu tham chiếu; kết quả phù hợp cho phân tích sơ bộ.")
                else:
                    score -= 0.14
                    score_cap = min(score_cap, 0.74)
                    detail.append(f"Năm {year_val} nằm trong khoảng dữ liệu nhưng không phải năm quan sát trực tiếp; kết quả nên xem là nội suy tham khảo.")

        else:
            gap = year_val - max_year
            if gap <= 2:
                score -= 0.10
                score_cap = min(score_cap, 0.82)
                detail.append(f"Năm {year_val} là kịch bản ngắn hạn sau vùng dữ liệu gốc; có thể dùng để tham khảo.")
            elif gap <= 5:
                score -= 0.18
                score_cap = min(score_cap, 0.74)
                detail.append(f"Năm {year_val} là dự báo tương lai gần; nên hiểu là mô phỏng theo giả định, không phải dữ liệu đã học trực tiếp.")
            elif gap <= 10:
                score -= 0.30
                score_cap = min(score_cap, 0.64)
                detail.append(f"Năm {year_val} nằm khá xa vùng dữ liệu gốc; kết quả phù hợp để so sánh kịch bản, cần đối chiếu thêm.")
            elif gap <= 15:
                score -= 0.40
                score_cap = min(score_cap, 0.56)
                detail.append(f"Năm {year_val} là kịch bản dài hạn; kết quả chủ yếu phục vụ mô phỏng xu hướng.")
            else:
                score -= 0.52
                score_cap = min(score_cap, 0.46)
                detail.append(f"Năm {year_val} rất xa vùng dữ liệu gốc; kết quả cần được diễn giải thận trọng.")

    # 2) Kiểm tra numeric nhưng bỏ qua biến kịch bản và biến nhị phân dễ gây z-score giả.
    for c, st in profile.get("numeric_stats", {}).items():
        if c not in input_df.columns or c in scenario_control_features or c in binary_indicator_features:
            continue
        if c in {"year", "year_offset", "future_year_gap"}:
            continue

        v = float(pd.to_numeric(input_df[c], errors="coerce").iloc[0])
        train_min = float(st.get("min", np.nan))
        train_max = float(st.get("max", np.nan))
        train_mean = float(st.get("mean", 0.0))
        train_std = max(float(st.get("std", 1.0)), 1e-9)
        z = abs((v - train_mean) / train_std)
        max_z = max(max_z, z)

        if np.isfinite(train_min) and np.isfinite(train_max) and (v < train_min or v > train_max):
            if c in {"product_weight_kg", "product_weight_log", "openpcf_factor_kgco2e_per_kg", "lca_proxy_pcf", "ceda_factor_kgco2e_per_usd"}:
                score -= 0.14
                score_cap = min(score_cap, 0.70)
            else:
                score -= 0.07
            detail.append(f"{FEATURE_NAME_VI.get(c,c)} nằm ngoài vùng tham chiếu: {v:.2f} so với [{train_min:.2f}, {train_max:.2f}].")
        elif z > 6 and c in {"product_weight_kg", "openpcf_factor_kgco2e_per_kg", "lca_proxy_pcf"}:
            score -= 0.06
            score_cap = min(score_cap, 0.80)
            detail.append(f"{FEATURE_NAME_VI.get(c,c)} khác xa trung tâm dữ liệu tham chiếu, nhưng vẫn nằm trong vùng đã học.")

    # 3) Kiểm tra category chính. Bỏ qua category kỹ thuật do UI tạo ra.
    unseen = []
    for c, levels in profile.get("cat_levels", {}).items():
        if c not in input_df.columns:
            continue
        v = str(input_df[c].iloc[0])
        if c in soft_scenario_categories:
            if v in set(levels) or any(token in v.lower() for token in ["user", "scenario", "kịch bản"]):
                continue
        if v not in set(levels):
            unseen.append(FEATURE_NAME_VI.get(c, c))

    if unseen:
        score -= min(0.24, 0.08 * len(unseen))
        score_cap = min(score_cap, 0.72)
        detail.append("Một số giá trị phân loại chưa có nhiều dữ liệu tham chiếu: " + ", ".join(unseen[:5]))

    score = min(score, score_cap)
    score = float(np.clip(score, 0.05, 0.98))

    if score >= 0.75:
        confidence = "Cao"
        message = "Dữ liệu đầu vào gần vùng dữ liệu đã có mẫu tham chiếu; kết quả phù hợp cho phân tích sơ bộ."
    elif score >= 0.50:
        confidence = "Trung bình"
        message = "Kết quả phù hợp để tham khảo, nhưng nên đối chiếu thêm với inventory thực tế hoặc giả định kịch bản."
    else:
        confidence = "Thận trọng"
        message = "Kết quả là mô phỏng dài hạn/ngoài vùng dữ liệu gốc, cần diễn giải thận trọng và không thay thế kiểm kê LCA chính thức."

    return {
        "detail": detail,
        "max_z": max_z,
        "unseen_count": len(unseen),
        "confidence": confidence,
        "score": score,
        "message": message,
    }

def calculate_lca_bottom_up(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory is None or len(inventory) == 0:
        return {"total_pcf": 0.0, "by_group": pd.DataFrame(), "detail": pd.DataFrame()}
    inv = inventory.copy()
    for col in ["amount", "emission_factor"]:
        inv[col] = pd.to_numeric(inv.get(col, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    inv["co2e"] = inv["amount"] * inv["emission_factor"]
    inv["activity_group"] = inv.get("activity_group", "Other").astype(str).fillna("Other")
    by_group = inv.groupby("activity_group", as_index=False)["co2e"].sum().sort_values("co2e", ascending=False)
    return {"total_pcf": float(inv["co2e"].sum()), "by_group": by_group, "detail": inv}


def hybrid_pcf_estimate(ml_pcf: float, lca_pcf: float, lca_weight: float = 0.45) -> float:
    if lca_pcf and lca_pcf > 0:
        return float((1 - lca_weight) * ml_pcf + lca_weight * lca_pcf)
    return float(ml_pcf)


def scenario_projection(
    base_pcf: float,
    start_year: int,
    target_year: int,
    renewable_gain_pct: float = 0,
    material_reduction_pct: float = 0,
    logistics_gain_pct: float = 0,
    supplier_factor_pct: float = 0,
) -> pd.DataFrame:
    target_year = max(int(target_year), int(start_year))
    years = sorted(set([start_year, min(start_year + 1, target_year), min(start_year + 5, target_year), min(start_year + 10, target_year), target_year]))
    years = [y for y in years if y >= start_year]
    rows = []
    horizon_total = max(target_year - start_year, 1)
    # Conservative levers: renewable mainly affects operation; material affects upstream; logistics affects transport.
    max_reduction = (renewable_gain_pct * 0.28 + material_reduction_pct * 0.48 + logistics_gain_pct * 0.16 + supplier_factor_pct * 0.08) / 100.0
    max_reduction = min(max_reduction, 0.85)
    for y in years:
        progress = (y - start_year) / horizon_total
        gradual = 1 - max_reduction * progress
        baseline = base_pcf * (1 + 0.002 * max(y - start_year, 0))
        rows.append({"year": y, "baseline_pcf": baseline, "scenario_pcf": max(baseline * gradual, 0), "reduction_pct": (1 - gradual) * 100})
    return pd.DataFrame(rows)


def build_input_row(reference_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for c in ["year", "product_weight_kg", "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac", "openpcf_factor_kgco2e_per_kg", "ceda_factor_kgco2e_per_usd", "lca_proxy_pcf", "renewable_energy_pct", "material_reduction_pct", "transport_improvement_pct"]:
        if c in kwargs:
            row[c] = kwargs[c]
        elif c in reference_df.columns:
            row[c] = float(pd.to_numeric(reference_df[c], errors="coerce").median())
        else:
            row[c] = 0.0
    for c in ["product_name", "product_detail", "country", "industry_group", "industry", "company_sector", "stage_level_available", "weight_source", "protocol", "protocol_simple", "data_source", "system_boundary", "functional_unit_type"]:
        if c in kwargs:
            row[c] = kwargs[c]
        elif c in reference_df.columns and not reference_df[c].mode().empty:
            row[c] = reference_df[c].mode().iloc[0]
        else:
            row[c] = "Unknown"
    df = pd.DataFrame([row])
    return add_model_features(df, reference_min_year=int(reference_df["year"].min()), max_train_year=int(reference_df["year"].max()))


def predict_with_package(package: dict[str, Any], input_row: pd.DataFrame) -> dict[str, Any]:
    feature_cols = package["metadata"]["feature_cols"]
    clf = package["classifier"]
    reg = package["regressor"]
    all_regs = package.get("regressors", {})
    X = input_row[feature_cols]
    pred_num = int(clf.predict(X)[0])
    pred_label = NUM_TO_LABEL.get(pred_num, str(pred_num))
    try:
        proba = clf.predict_proba(X)[0]
    except Exception:
        proba = np.eye(3)[pred_num]
    raw_pcf = np.asarray(reg.predict(X), dtype=float)
    raw_pcf = np.nan_to_num(raw_pcf, nan=0.0, posinf=1e9, neginf=0.0)
    raw_pcf = np.clip(raw_pcf, 0.0, 1e9)
    pcf = float(max(raw_pcf[0], 0))
    ensemble = []
    for m in all_regs.values():
        try:
            raw_m = np.asarray(m.predict(X), dtype=float)
            raw_m = np.nan_to_num(raw_m, nan=0.0, posinf=1e9, neginf=0.0)
            raw_m = np.clip(raw_m, 0.0, 1e9)
            ensemble.append(float(max(raw_m[0], 0)))
        except Exception:
            pass
    residual_q = package["metadata"].get("residual_abs_quantiles", {"p50": 0, "p90": 0})
    spread = float(residual_q.get("p90", 0))
    if len(ensemble) >= 3:
        q10_model = float(np.quantile(ensemble, 0.10))
        q90_model = float(np.quantile(ensemble, 0.90))
        q10 = max(min(q10_model, pcf) - spread * 0.25, 0)
        q90 = max(q90_model, pcf) + spread * 0.25
    else:
        q10 = max(pcf - spread, 0)
        q90 = pcf + spread
    return {"label": pred_label, "label_vi": LABEL_VI.get(pred_label, pred_label), "proba": proba, "pcf": pcf, "p10": q10, "p90": q90, "ensemble_predictions": ensemble}


def get_local_factor_impact(package: dict[str, Any], input_row: pd.DataFrame, n_top: int = 6) -> pd.DataFrame:
    clf = package["classifier"]
    meta = package["metadata"]
    feature_cols = meta["feature_cols"]
    X0 = input_row[feature_cols].copy()
    try:
        base_proba = clf.predict_proba(X0)[0]
    except Exception:
        return pd.DataFrame()
    pred_idx = int(np.argmax(base_proba))
    impacts: list[dict[str, Any]] = []
    for c in NUMERIC_FEATURES:
        if c not in X0.columns:
            continue
        Xt = X0.copy()
        v = float(Xt[c].iloc[0]) if pd.notna(Xt[c].iloc[0]) else 0.0
        delta = max(abs(v) * 0.10, 0.05)
        Xt[c] = v + delta
        try:
            newp = clf.predict_proba(Xt)[0][pred_idx]
            impacts.append({"feature": c, "feature_vi": FEATURE_NAME_VI.get(c, c), "impact": float(newp - base_proba[pred_idx])})
        except Exception:
            pass
    cat_levels = meta.get("ood_profile", {}).get("cat_levels", {})
    for c in CATEGORICAL_FEATURES:
        if c not in X0.columns:
            continue
        cur = str(X0[c].iloc[0])
        alts = [v for v in cat_levels.get(c, []) if v != cur]
        if not alts:
            continue
        Xt = X0.copy()
        Xt[c] = alts[0]
        try:
            newp = clf.predict_proba(Xt)[0][pred_idx]
            impacts.append({"feature": c, "feature_vi": FEATURE_NAME_VI.get(c, c), "impact": float(newp - base_proba[pred_idx])})
        except Exception:
            pass
    out = pd.DataFrame(impacts)
    if out.empty:
        return out
    out["abs_impact"] = out["impact"].abs()
    out["direction_vi"] = np.where(out["impact"] >= 0, "Tăng xu hướng nhãn dự báo", "Giảm xu hướng nhãn dự báo")
    return out.sort_values("abs_impact", ascending=False).head(n_top).reset_index(drop=True)


def save_package(package: dict[str, Any], path: str | Path = MODEL_PATH, also_root: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)
    if also_root:
        joblib.dump(package, ROOT_MODEL_PATH)


def load_package(path: str | Path | None = None) -> dict[str, Any]:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [ROOT_MODEL_PATH, MODEL_PATH, Path("outputs/models/ecopredict_model_package.joblib")]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Không tìm thấy ecopredict_model_package.joblib. Hãy chạy train_advanced_models.py trước.")




def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "-"
    if abs(xf) >= 1000:
        return f"{xf:,.0f}"
    return f"{xf:,.{digits}f}"
