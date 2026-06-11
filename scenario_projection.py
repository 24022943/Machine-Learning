"""
scenario_projection.py
EcoPredict Carbon - Scenario-based parametric projection module.

Mục đích:
- Sử dụng mô hình kịch bản tham số minh bạch cho giai đoạn 2025–2050.
- Không yêu cầu chuỗi thời gian dài.
- PCF tương lai = PCF nền × tổ hợp factor theo từng giai đoạn vòng đời.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    upstream_2050: float
    operations_2050: float
    downstream_2050: float
    transport_2050: float
    eol_2050: float
    description: str


SCENARIOS: Dict[str, ScenarioConfig] = {
    "baseline": ScenarioConfig(
        name="Baseline - chính sách hiện tại",
        upstream_2050=0.85,
        operations_2050=0.60,
        downstream_2050=0.95,
        transport_2050=0.75,
        eol_2050=0.90,
        description="Chính sách hiện tại tiếp tục; cải thiện hiệu suất và năng lượng ở mức vừa phải.",
    ),
    "net_zero": ScenarioConfig(
        name="Net Zero 2050",
        upstream_2050=0.70,
        operations_2050=0.25,
        downstream_2050=0.80,
        transport_2050=0.45,
        eol_2050=0.65,
        description="Khử carbon nhanh, tăng năng lượng tái tạo, cải thiện hiệu suất, logistics xanh và tái chế.",
    ),
    "pessimistic": ScenarioConfig(
        name="Pessimistic - chuyển dịch chậm",
        upstream_2050=0.95,
        operations_2050=0.85,
        downstream_2050=1.00,
        transport_2050=0.95,
        eol_2050=1.00,
        description="Chuyển dịch chậm; ít cải thiện về vật liệu, năng lượng, vận chuyển và xử lý cuối vòng đời.",
    ),
}


def normalize_lifecycle_shares(
    upstream: float = 40,
    operations: float = 30,
    downstream: float = 15,
    transport: float = 10,
    end_of_life: float = 5,
) -> Dict[str, float]:
    """Chuẩn hóa tỷ trọng vòng đời về tổng 1.0."""
    values = np.array([upstream, operations, downstream, transport, end_of_life], dtype=float)
    total = float(values.sum())
    if total <= 0:
        values = np.array([40, 30, 15, 10, 5], dtype=float)
        total = float(values.sum())
    values = values / total
    return {
        "upstream": float(values[0]),
        "operations": float(values[1]),
        "downstream": float(values[2]),
        "transport": float(values[3]),
        "end_of_life": float(values[4]),
    }


def build_custom_2050_factors(
    renewable_gain_pct: float = 20,
    material_reduction_pct: float = 10,
    logistics_gain_pct: float = 10,
    supplier_factor_pct: float = 5,
) -> Dict[str, float]:
    """
    Chuyển các slider kịch bản người dùng nhập thành hệ số 2050.

    Hệ số càng nhỏ nghĩa là PCF ở giai đoạn đó càng giảm.
    Ví dụ: operations_2050 = 0.55 nghĩa là đến 2050 phát thải operations còn 55% so với baseline.
    """
    renewable = np.clip(float(renewable_gain_pct), 0, 100) / 100
    material = np.clip(float(material_reduction_pct), 0, 80) / 100
    logistics = np.clip(float(logistics_gain_pct), 0, 80) / 100
    supplier = np.clip(float(supplier_factor_pct), 0, 80) / 100

    # Các công thức dưới đây là giả định minh bạch cho prototype, không phải hệ số chứng nhận.
    upstream = 1.0 - 0.70 * material - 0.25 * supplier
    operations = 1.0 - 0.65 * renewable - 0.20 * material
    downstream = 1.0 - 0.25 * material
    transport = 1.0 - 0.70 * logistics - 0.15 * supplier
    eol = 1.0 - 0.25 * material - 0.25 * supplier

    return {
        "upstream_2050": float(np.clip(upstream, 0.35, 1.10)),
        "operations_2050": float(np.clip(operations, 0.15, 1.10)),
        "downstream_2050": float(np.clip(downstream, 0.55, 1.10)),
        "transport_2050": float(np.clip(transport, 0.25, 1.10)),
        "eol_2050": float(np.clip(eol, 0.50, 1.10)),
    }


def _get_config(scenario_key: str, custom_2050_factors: Optional[Dict[str, float]] = None) -> ScenarioConfig:
    if scenario_key == "custom":
        f = custom_2050_factors or build_custom_2050_factors()
        return ScenarioConfig(
            name="Custom - theo slider người dùng",
            upstream_2050=float(f["upstream_2050"]),
            operations_2050=float(f["operations_2050"]),
            downstream_2050=float(f["downstream_2050"]),
            transport_2050=float(f["transport_2050"]),
            eol_2050=float(f["eol_2050"]),
            description="Kịch bản tùy chỉnh từ slider: điện tái tạo, giảm vật liệu, cải thiện vận chuyển và supplier/geography.",
        )
    return SCENARIOS.get(scenario_key, SCENARIOS["baseline"])


def interpolate_factor(year: int, baseline_year: int, target_year: int, target_factor: float) -> float:
    """Nội suy tuyến tính từ 1.0 tại baseline_year đến target_factor tại target_year."""
    year = int(year)
    baseline_year = int(baseline_year)
    target_year = int(target_year)
    if target_year <= baseline_year or year <= baseline_year:
        return 1.0
    progress = (year - baseline_year) / (target_year - baseline_year)
    progress = max(0.0, min(1.0, progress))
    return float(1.0 + progress * (float(target_factor) - 1.0))


def uncertainty_rate(year: int) -> float:
    """Khoảng bất định tăng theo độ xa của năm mô phỏng."""
    if year <= 2030:
        return 0.15
    if year <= 2040:
        return 0.20
    return 0.30


def project_pcf_for_year(
    baseline_pcf: float,
    year: int,
    baseline_year: int,
    scenario_key: str,
    lifecycle_shares: Dict[str, float],
    target_year: int = 2050,
    custom_2050_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float | str]:
    """Tính PCF mô phỏng cho một năm theo kịch bản tham số."""
    config = _get_config(scenario_key, custom_2050_factors)

    upstream_factor = interpolate_factor(year, baseline_year, target_year, config.upstream_2050)
    operations_factor = interpolate_factor(year, baseline_year, target_year, config.operations_2050)
    downstream_factor = interpolate_factor(year, baseline_year, target_year, config.downstream_2050)
    transport_factor = interpolate_factor(year, baseline_year, target_year, config.transport_2050)
    eol_factor = interpolate_factor(year, baseline_year, target_year, config.eol_2050)

    combined_factor = (
        lifecycle_shares.get("upstream", 0.40) * upstream_factor
        + lifecycle_shares.get("operations", 0.30) * operations_factor
        + lifecycle_shares.get("downstream", 0.15) * downstream_factor
        + lifecycle_shares.get("transport", 0.10) * transport_factor
        + lifecycle_shares.get("end_of_life", 0.05) * eol_factor
    )

    projected_pcf = max(float(baseline_pcf) * float(combined_factor), 0.0)
    u = uncertainty_rate(int(year))

    return {
        "year": int(year),
        "scenario": config.name,
        "baseline_pcf": float(baseline_pcf),
        "projected_pcf": float(projected_pcf),
        "lower": float(projected_pcf * (1 - u)),
        "upper": float(projected_pcf * (1 + u)),
        "uncertainty_pct": float(u * 100),
        "combined_factor": float(combined_factor),
        "upstream_factor": float(upstream_factor),
        "operations_factor": float(operations_factor),
        "downstream_factor": float(downstream_factor),
        "transport_factor": float(transport_factor),
        "eol_factor": float(eol_factor),
    }


def build_projection_pathway(
    baseline_pcf: float,
    baseline_year: int,
    target_year: int,
    scenario_key: str,
    lifecycle_shares: Dict[str, float],
    custom_2050_factors: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Tạo bảng pathway từ baseline_year đến target_year."""
    baseline_year = int(baseline_year)
    target_year = int(max(target_year, baseline_year))
    years: List[int] = list(range(baseline_year, target_year + 1))
    rows = [
        project_pcf_for_year(
            baseline_pcf=baseline_pcf,
            year=year,
            baseline_year=baseline_year,
            target_year=target_year,
            scenario_key=scenario_key,
            lifecycle_shares=lifecycle_shares,
            custom_2050_factors=custom_2050_factors,
        )
        for year in years
    ]
    return pd.DataFrame(rows)


def build_assumptions_table(custom_2050_factors: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Bảng giả định để hiển thị trong Streamlit."""
    rows = []
    for key, value in SCENARIOS.items():
        rows.append(
            {
                "Kịch bản": value.name,
                "Upstream 2050": value.upstream_2050,
                "Operations 2050": value.operations_2050,
                "Downstream 2050": value.downstream_2050,
                "Transport 2050": value.transport_2050,
                "End-of-life 2050": value.eol_2050,
                "Diễn giải": value.description,
            }
        )
    if custom_2050_factors is not None:
        rows.append(
            {
                "Kịch bản": "Custom - theo slider người dùng",
                "Upstream 2050": custom_2050_factors["upstream_2050"],
                "Operations 2050": custom_2050_factors["operations_2050"],
                "Downstream 2050": custom_2050_factors["downstream_2050"],
                "Transport 2050": custom_2050_factors["transport_2050"],
                "End-of-life 2050": custom_2050_factors["eol_2050"],
                "Diễn giải": "Sinh từ các slider kịch bản trong sidebar.",
            }
        )
    return pd.DataFrame(rows)
