"""
sensitivity_analysis.py
Phân tích độ nhạy cho EcoPredict Carbon.

Trả lời câu hỏi:
- Nếu emission factor của vật liệu/năng lượng/vận chuyển thay đổi ±20%, PCF đổi bao nhiêu?
- Driver nào cần ưu tiên kiểm soát dữ liệu hoặc cải tiến phát thải?
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_FACTORS: dict[str, tuple[float, float]] = {
    "Material factor": (1.90, 0.40),
    "Energy factor": (0.45, 0.30),
    "Downstream/use factor": (1.00, 0.15),
    "Transport factor": (0.12, 0.10),
    "End-of-life factor": (0.25, 0.05),
}


def one_way_sensitivity(
    baseline_pcf: float,
    factor_name: str,
    factor_baseline: float,
    lifecycle_share: float,
    variations: np.ndarray | None = None,
) -> pd.DataFrame:
    """Độ nhạy một chiều theo một factor."""
    if variations is None:
        variations = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20], dtype=float)
    baseline_pcf = float(max(baseline_pcf, 0.0))
    lifecycle_share = float(max(min(lifecycle_share, 1.0), 0.0))
    rows: list[dict[str, float | str]] = []
    for var in variations:
        factor_value = float(factor_baseline) * (1 + float(var) / 100)
        delta_pcf = baseline_pcf * lifecycle_share * (float(var) / 100)
        pcf_new = baseline_pcf + delta_pcf
        rows.append({
            "factor": factor_name,
            "variation_pct": float(var),
            "factor_value": factor_value,
            "pcf_new": pcf_new,
            "pcf_change": delta_pcf,
            "pcf_change_pct": (delta_pcf / baseline_pcf * 100) if baseline_pcf > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def tornado_chart(
    baseline_pcf: float,
    factors: Dict[str, Tuple[float, float]] | None = None,
    variation_pct: float = 20.0,
    output_file: str | Path = "outputs/figures/sensitivity_tornado.png",
) -> pd.DataFrame:
    """Vẽ tornado chart so sánh tác động của các factor."""
    if factors is None:
        factors = DEFAULT_FACTORS
    baseline_pcf = float(max(baseline_pcf, 0.0))
    rows: list[dict[str, float | str]] = []
    for name, (factor_value, share) in factors.items():
        share = float(max(min(share, 1.0), 0.0))
        pcf_low = baseline_pcf * (1 - variation_pct / 100 * share)
        pcf_high = baseline_pcf * (1 + variation_pct / 100 * share)
        rows.append({
            "factor": name,
            "factor_baseline": float(factor_value),
            "lifecycle_share": share,
            "pcf_low": pcf_low,
            "pcf_high": pcf_high,
            "impact_low": pcf_low - baseline_pcf,
            "impact_high": pcf_high - baseline_pcf,
            "range": pcf_high - pcf_low,
        })
    df = pd.DataFrame(rows).sort_values("range", ascending=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(df))
    ax.barh(y, df["impact_low"], left=baseline_pcf, color="#f97316", alpha=0.85, label=f"-{variation_pct:.0f}% factor")
    ax.barh(y, df["impact_high"], left=baseline_pcf, color="#047857", alpha=0.85, label=f"+{variation_pct:.0f}% factor")
    ax.axvline(baseline_pcf, color="#111827", linestyle="--", linewidth=1.8, label="Baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(df["factor"])
    ax.set_xlabel("PCF (kg CO₂e)")
    ax.set_title("Sensitivity Tornado Chart - ± factor variation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return df.reset_index(drop=True)


def two_way_sensitivity(
    baseline_pcf: float,
    factor1_name: str,
    factor1_share: float,
    factor2_name: str,
    factor2_share: float,
    variations: np.ndarray | None = None,
    output_file: str | Path = "outputs/figures/sensitivity_heatmap_2way.png",
) -> pd.DataFrame:
    """Heatmap độ nhạy khi hai factor cùng thay đổi."""
    if variations is None:
        variations = np.array([-20, -10, 0, 10, 20], dtype=float)
    baseline_pcf = float(max(baseline_pcf, 0.0))
    matrix = np.zeros((len(variations), len(variations)))
    rows = []
    for i, v1 in enumerate(variations):
        for j, v2 in enumerate(variations):
            pcf_new = baseline_pcf * (1 + factor1_share * v1 / 100 + factor2_share * v2 / 100)
            matrix[i, j] = pcf_new
            rows.append({"factor1_change_pct": v1, "factor2_change_pct": v2, "pcf_new": pcf_new})

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(variations)))
    ax.set_yticks(np.arange(len(variations)))
    ax.set_xticklabels([f"{v:+.0f}%" for v in variations])
    ax.set_yticklabels([f"{v:+.0f}%" for v in variations])
    ax.set_xlabel(f"{factor2_name} change")
    ax.set_ylabel(f"{factor1_name} change")
    ax.set_title("Two-way sensitivity heatmap")
    for i in range(len(variations)):
        for j in range(len(variations)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PCF (kg CO₂e)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return pd.DataFrame(rows)


def scenario_sensitivity(
    baseline_pcf: float,
    lifecycle_stages: Dict[str, float] | None = None,
    scenarios: Dict[str, Dict[str, float]] | None = None,
    output_file: str | Path = "outputs/figures/sensitivity_scenarios.png",
) -> pd.DataFrame:
    """So sánh PCF theo các bộ factor kịch bản."""
    if lifecycle_stages is None:
        lifecycle_stages = {"upstream": 0.40, "operations": 0.30, "downstream": 0.15, "transport": 0.10, "eol": 0.05}
    if scenarios is None:
        scenarios = {
            "Current baseline": {"upstream": 1.00, "operations": 1.00, "downstream": 1.00, "transport": 1.00, "eol": 1.00},
            "Net Zero 2050": {"upstream": 0.70, "operations": 0.25, "downstream": 0.80, "transport": 0.45, "eol": 0.65},
            "Pessimistic": {"upstream": 0.95, "operations": 0.85, "downstream": 1.00, "transport": 0.95, "eol": 1.00},
        }
    baseline_pcf = float(max(baseline_pcf, 0.0))
    rows = []
    for name, factors in scenarios.items():
        combined_factor = sum(lifecycle_stages.get(stage, 0.0) * factors.get(stage, 1.0) for stage in lifecycle_stages)
        pcf = baseline_pcf * combined_factor
        rows.append({"scenario": name, "combined_factor": combined_factor, "pcf": pcf, "delta": pcf - baseline_pcf, "delta_pct": (pcf / baseline_pcf - 1) * 100 if baseline_pcf > 0 else 0.0})
    df = pd.DataFrame(rows).sort_values("pcf")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#047857" if v <= baseline_pcf else "#f97316" for v in df["pcf"]]
    ax.bar(df["scenario"], df["pcf"], color=colors, alpha=0.9)
    ax.axhline(baseline_pcf, linestyle="--", color="#111827", label=f"Baseline {baseline_pcf:.2f}")
    ax.set_ylabel("PCF (kg CO₂e)")
    ax.set_title("Scenario sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return df.reset_index(drop=True)


def generate_default_sensitivity_outputs(baseline_pcf: float, output_dir: str | Path = "outputs") -> dict[str, str]:
    """Sinh nhanh toàn bộ ảnh/bảng sensitivity mặc định."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    tab_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    tornado = tornado_chart(baseline_pcf, output_file=fig_dir / "sensitivity_tornado.png")
    tornado.to_csv(tab_dir / "sensitivity_tornado.csv", index=False)

    heat = two_way_sensitivity(
        baseline_pcf,
        factor1_name="Material factor",
        factor1_share=0.40,
        factor2_name="Energy factor",
        factor2_share=0.30,
        output_file=fig_dir / "sensitivity_heatmap_2way.png",
    )
    heat.to_csv(tab_dir / "sensitivity_heatmap_2way.csv", index=False)

    scenarios = scenario_sensitivity(baseline_pcf, output_file=fig_dir / "sensitivity_scenarios.png")
    scenarios.to_csv(tab_dir / "sensitivity_scenarios.csv", index=False)

    return {
        "tornado": str(fig_dir / "sensitivity_tornado.png"),
        "heatmap": str(fig_dir / "sensitivity_heatmap_2way.png"),
        "scenario": str(fig_dir / "sensitivity_scenarios.png"),
    }


if __name__ == "__main__":
    outputs = generate_default_sensitivity_outputs(67.67)
    print("Generated sensitivity outputs:", outputs)
