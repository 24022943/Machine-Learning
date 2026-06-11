"""
carbon_eda.py
EDA for EcoPredict Carbon v7.

Chạy:
    python carbon_eda.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from carbon_utils import load_all_sources, TARGET_COL

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def main() -> None:
    src = load_all_sources("carbon_catalogue.csv", include_openpcf_in_training=True)
    df = src["training"]
    carbon = src["carbon"]
    openpcf = src["openpcf"]
    ceda = src["ceda"]

    summary = {
        "training_rows": int(len(df)),
        "carbon_catalogue_rows": int(len(carbon)),
        "openpcf_rows": int(len(openpcf)),
        "open_ceda_factors": int(len(ceda)),
        "pcf_min": float(df[TARGET_COL].min()),
        "pcf_median": float(df[TARGET_COL].median()),
        "pcf_mean": float(df[TARGET_COL].mean()),
        "pcf_max": float(df[TARGET_COL].max()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n_countries": int(df["country"].nunique()),
        "n_industry_groups": int(df["industry_group"].nunique()),
    }
    pd.Series(summary).to_csv(TAB / "eda_summary.csv", header=["value"])
    (TAB / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df["data_source"].value_counts().to_csv(TAB / "data_source_counts.csv")

    print("Data shape:", df.shape)
    print("Sources:", df["data_source"].value_counts().to_dict())

    # 1. Raw PCF histogram: demonstrates right-skew.
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df[TARGET_COL], bins=80, color="#10b981")
    plt.title("Phân phối PCF thang tuyến tính - lệch phải mạnh")
    plt.xlabel("PCF (kg CO2e)")
    savefig("eda_pcf_hist_linear.png")

    # 2. Log PCF distribution.
    plt.figure(figsize=(8, 4.5))
    sns.histplot(np.log1p(df[TARGET_COL]), bins=60, kde=True, color="#047857")
    plt.title("Phân phối log(PCF + 1) - phù hợp hơn cho dữ liệu lệch phải")
    plt.xlabel("log(PCF + 1)")
    savefig("eda_pcf_hist_log.png")

    # 3. Distribution by data source.
    plt.figure(figsize=(8.5, 4.8))
    sns.boxplot(data=df, y="data_source", x=np.log1p(df[TARGET_COL]), color="#a7f3d0")
    plt.title("PCF theo nguồn dữ liệu (log-scale)")
    plt.xlabel("log(PCF + 1)")
    plt.ylabel("")
    savefig("eda_pcf_by_data_source.png")

    # 4. Top industry groups.
    counts = df["industry_group"].value_counts().head(12)
    counts.to_csv(TAB / "top_industry_groups.csv")
    plt.figure(figsize=(8.8, 5.4))
    sns.barplot(x=counts.values, y=counts.index, color="#10b981")
    plt.title("Top nhóm ngành theo số mẫu")
    plt.xlabel("Số mẫu")
    plt.ylabel("")
    savefig("eda_top_industry_groups.png")

    # 5. Time/source trend.
    time_stats = df.groupby(["year", "data_source"])[TARGET_COL].agg(["count", "median", "mean"]).reset_index()
    time_stats.to_csv(TAB / "pcf_by_year_source.csv", index=False)
    plt.figure(figsize=(8.2, 4.8))
    for src_name, g in time_stats.groupby("data_source"):
        plt.plot(g["year"], g["median"], marker="o", label=src_name)
    plt.yscale("log")
    plt.title("Xu hướng trung vị PCF theo năm và nguồn dữ liệu")
    plt.xlabel("Năm")
    plt.ylabel("Median PCF (kg CO2e, log)")
    plt.legend()
    savefig("eda_pcf_time_trend_by_source.png")

    # 6. Lifecycle stacked average.
    stages = ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]
    stage_mean = df[stages].mean().rename({
        "upstream_frac": "Upstream",
        "operations_frac": "Operations",
        "downstream_frac": "Downstream",
        "transport_frac": "Transport",
        "end_of_life_frac": "End-of-life",
    })
    stage_mean.to_csv(TAB / "lifecycle_stage_mean.csv")
    plt.figure(figsize=(8, 2.4))
    left = 0
    colors = ["#047857", "#10b981", "#86efac", "#64748b", "#cbd5e1"]
    for (name, val), color in zip(stage_mean.items(), colors):
        plt.barh(["Average"], [val * 100], left=left, color=color, label=name)
        left += val * 100
    plt.title("Tỷ trọng phát thải vòng đời trung bình")
    plt.xlabel("Tỷ trọng (%)")
    plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.35), loc="upper center")
    savefig("eda_lifecycle_stacked_bar.png")

    # 7. OpenPCF geography effect.
    if not openpcf.empty:
        top_products = openpcf["product_name"].value_counts().head(8).index
        geo = openpcf[openpcf["product_name"].isin(top_products)].groupby("product_name")["openpcf_factor_kgco2e_per_kg"].agg(["median", "min", "max", "count"]).sort_values("median", ascending=False)
        geo.to_csv(TAB / "openpcf_product_factor_summary.csv")
        plt.figure(figsize=(9, 5.2))
        geo_plot = geo.sort_values("median")
        plt.barh(geo_plot.index, geo_plot["median"], color="#047857")
        plt.xlabel("Median factor (kg CO2e/kg)")
        plt.title("OpenPCF: trung vị hệ số PCF theo sản phẩm/vật liệu")
        savefig("eda_openpcf_product_factors.png")

    # 8. Open CEDA country median.
    if not ceda.empty:
        cmed = ceda.groupby("country")["ceda_factor_kgco2e_per_usd"].median().sort_values(ascending=False).head(15)
        cmed.to_csv(TAB / "open_ceda_top_country_median.csv")
        plt.figure(figsize=(8.8, 5.2))
        sns.barplot(x=cmed.values, y=cmed.index, color="#64748b")
        plt.title("Open CEDA: Top quốc gia theo median kgCO2e/USD")
        plt.xlabel("kgCO2e / USD")
        plt.ylabel("")
        savefig("eda_open_ceda_country_median.png")

    # 9. Correlation heatmap.
    cols = ["year", "product_weight_log", "openpcf_factor_kgco2e_per_kg", "ceda_factor_kgco2e_per_usd", "lca_proxy_pcf", "upstream_frac", "operations_frac", "downstream_frac", TARGET_COL]
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(TAB / "correlation_matrix.csv")
    plt.figure(figsize=(9, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", center=0)
    plt.title("Heatmap tương quan giữa các biến chính")
    savefig("eda_correlation_heatmap.png")

    print("✅ EDA completed. Outputs saved to outputs/figures and outputs/tables.")


if __name__ == "__main__":
    main()
