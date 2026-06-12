"""
app.py

Chạy:
    python -m streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scenario_projection import (
    normalize_lifecycle_shares,
    build_custom_2050_factors,
    build_projection_pathway,
    build_assumptions_table,
)

from carbon_utils import (
    DATA_PATH, TARGET_COL, LABEL_ORDER, LABEL_VI, FEATURE_NAME_VI,
    DEMO_EMISSION_FACTORS,
    load_all_sources, load_package, build_input_row, predict_with_package,
    calculate_lca_bottom_up,
    check_ood, get_local_factor_impact, fmt_num,
)

st.set_page_config(
    page_title="EcoPredict Carbon",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

CUSTOM_CSS = """
<style>
:root{--primary:#047857;--accent:#10b981;--dark:#063b2d;--bg:#f4fbf7;--muted:#667085;--line:#dbe7e0;--ink:#0f172a;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f8fcfa 0%,#edf7f2 100%);} 
[data-testid="stSidebar"]{background:linear-gradient(180deg,#063b2d 0%,#0f5132 100%);} 
[data-testid="stSidebar"] *{color:#f7fff9 !important;}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="base-input"] input,
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] > div{color:#111827 !important;-webkit-text-fill-color:#111827 !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-baseweb="base-input"]{background:#ffffff !important;border-radius:14px !important;}
.hero-shell{position:relative;overflow:hidden;margin-bottom:24px;padding:34px 36px 30px 36px;border-radius:32px;background:linear-gradient(180deg,#f4fbf7 0%,#edf7f2 100%);border:1px solid #bce5cf;box-shadow:0 18px 42px rgba(15,81,50,.08);} 
.hero-shell::after{content:"";position:absolute;right:-70px;bottom:-100px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,.20),rgba(16,185,129,.10) 55%,transparent 56%);} 
.hero-badge{display:inline-flex;align-items:center;gap:10px;padding:12px 22px;border-radius:999px;background:#d9ece4;color:#0f5132;font-size:17px;font-weight:850;margin-bottom:24px;} 
.hero-title{position:relative;z-index:1;font-size:42px;line-height:1.16;font-weight:950;color:#0f172a;margin:0 0 16px 0;letter-spacing:-1px;max-width:1120px;} 
.hero-subtitle{position:relative;z-index:1;max-width:1100px;font-size:18px;line-height:1.7;color:#667085;margin-bottom:24px;} 
.hero-chip-row{display:flex;flex-wrap:wrap;gap:16px;position:relative;z-index:1;} .hero-chip{display:inline-flex;align-items:center;padding:13px 24px;border-radius:999px;background:#d6efe3;color:#0f5132;font-size:16px;font-weight:850;} 
.card{background:white;padding:24px;border-radius:26px;box-shadow:0 12px 32px rgba(15,81,50,.07);border:1px solid #e6efe9;margin-bottom:18px;} 
.section-title{font-size:25px;font-weight:950;color:#0f172a;margin-bottom:8px;} .card-subtitle{font-size:15px;color:#667085;margin-bottom:14px;line-height:1.6;} 
.kpi-card{background:white;border-radius:24px;padding:24px;box-shadow:0 12px 28px rgba(15,81,50,.07);border:1px solid #e8f2ed;min-height:138px;} 
.kpi-title{font-size:14px;color:#64748b;font-weight:850;margin-bottom:8px;} .kpi-value{font-size:38px;font-weight:950;margin:0;letter-spacing:-1px;color:#0f172a;} .kpi-note{font-size:13px;color:#64748b;margin-top:8px;line-height:1.45;} 
.kpi-green{background:radial-gradient(circle at 90% 20%,rgba(16,185,129,.30),transparent 30%),linear-gradient(135deg,#047857 0%,#0f5132 100%);color:white;border:none;} .kpi-green .kpi-title,.kpi-green .kpi-value,.kpi-green .kpi-note{color:white;} 
.info-box{padding:16px 18px;border-radius:18px;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;line-height:1.65;margin-bottom:10px;} 
.scope-box{padding:16px 18px;border-radius:18px;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;line-height:1.65;margin-bottom:10px;} 
.soft-warning{padding:16px 18px;border-radius:18px;background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;line-height:1.65;margin-bottom:10px;} 
.insight-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:8px 0 14px 0;} .insight-card{background:#f8fbfa;border:1px solid #e5ece8;border-radius:18px;padding:16px 18px;} .insight-label{font-size:13px;color:#64748b;font-weight:850;margin-bottom:6px;} .insight-value{font-size:24px;color:#111827;font-weight:950;line-height:1.15;} .insight-note{font-size:13px;color:#64748b;margin-top:4px;line-height:1.45;} 
.badge{display:inline-block;padding:8px 12px;border-radius:999px;font-size:13px;font-weight:850;background:#dcfce7;color:#166534;margin-right:6px;margin-bottom:6px;} 
.small-muted{font-size:13px;color:#64748b;line-height:1.5;} 
@media(max-width:900px){.hero-title{font-size:32px}.insight-grid{grid-template-columns:1fr}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-badge'>🌱 ECOPREDICT CARBON</div>
            <div class='hero-title'>ECOPREDICT CARBON – HỆ THỐNG DỰ BÁO PHÁT THẢI CARBON CỦA SẢN PHẨM</div>
            <div class='hero-subtitle'>Product Carbon Footprint prediction and emission-level classification based on Carbon Catalogue, OpenPCF and Open CEDA data</div>
            <div class='hero-chip-row'>
                <div class='hero-chip'>PCF Core</div>
                <div class='hero-chip'>LCA-oriented</div>
                <div class='hero-chip'>Machine Learning</div>
                <div class='hero-chip'>Scenario Projection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_sources_cached() -> dict[str, pd.DataFrame]:
    return load_all_sources(str(DATA_PATH), include_openpcf_in_training=True)


@st.cache_resource(show_spinner=False)
def load_model_cached() -> dict[str, Any] | None:
    for p in [Path("ecopredict_model_package.joblib"), Path("outputs/models/ecopredict_model_package.joblib")]:
        if p.exists():
            return load_package(p)
    return None


def bootstrap() -> tuple[dict[str, pd.DataFrame], dict[str, Any] | None]:
    return load_sources_cached(), load_model_cached()


def train_now_panel() -> None:
    st.markdown("<div class='soft-warning'><b>Chưa tìm thấy model package.</b><br>Hãy chạy <code>python train_advanced_models.py</code> trước, hoặc bấm nút dưới để huấn luyện ngay trên môi trường hiện tại.</div>", unsafe_allow_html=True)
    if st.button("Huấn luyện model ngay", type="primary"):
        with st.spinner("Đang huấn luyện mô hình, vui lòng chờ..."):
            result = subprocess.run([sys.executable, "train_advanced_models.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Huấn luyện xong. Hãy refresh app.")
                st.code(result.stdout[-3000:])
            else:
                st.error("Huấn luyện lỗi")
                st.code(result.stderr[-5000:])


def plot_probability_bar(proba: np.ndarray) -> go.Figure:
    labels = [LABEL_VI[x] for x in LABEL_ORDER]
    colors = ["#047857", "#f59e0b", "#ef4444"]
    fig = go.Figure(go.Bar(x=labels, y=proba * 100, marker_color=colors, text=[f"{p*100:.1f}%" for p in proba], textposition="outside", showlegend=False))
    fig.update_layout(height=310, margin=dict(l=30, r=30, t=10, b=40), yaxis_title="Xác suất (%)", xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), yaxis=dict(range=[0, max(100, float(proba.max()*120))], gridcolor="#e5e7eb"))
    return fig


def plot_lifecycle_stacked(vals: dict[str, float]) -> go.Figure:
    names = ["Upstream", "Operations", "Downstream", "Transport", "End-of-life"]
    xs = np.array([vals.get(k, 0.0) for k in ["upstream", "operations", "downstream", "transport", "eol"]], dtype=float)
    xs = xs / xs.sum() if xs.sum() > 0 else xs
    colors = ["#047857", "#10b981", "#86efac", "#64748b", "#cbd5e1"]
    fig = go.Figure()
    for name, val, color in zip(names, xs, colors):
        fig.add_trace(go.Bar(y=["Tỷ trọng"], x=[val * 100], orientation="h", name=name, marker_color=color, text=[f"{val*100:.1f}%"], textposition="inside"))
    fig.update_layout(barmode="stack", height=220, margin=dict(l=20, r=20, t=10, b=40), xaxis_title="Tỷ trọng (%)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), legend=dict(orientation="h", y=-0.3), xaxis=dict(range=[0, 100], gridcolor="#e5e7eb"))
    return fig


def plot_benchmark(values: dict[str, float]) -> go.Figure:
    labels = ["Sản phẩm của bạn", "Trung vị ngành", "Trung vị toàn bộ", "OpenPCF gần nhất"]
    xs = [values.get("product", 0), values.get("industry", 0), values.get("global", 0), values.get("openpcf", 0)]
    fig = go.Figure(go.Bar(y=labels, x=xs, orientation="h", marker_color=["#047857", "#64748b", "#cbd5e1", "#10b981"], text=[fmt_num(v) for v in xs], textposition="outside", showlegend=False))
    fig.update_layout(height=330, margin=dict(l=145, r=80, t=8, b=40), xaxis_title="kg CO₂e", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827", size=13), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(categoryorder="array", categoryarray=labels[::-1]))
    return fig


def plot_lca_breakdown(by_group: pd.DataFrame) -> go.Figure:
    if by_group.empty:
        fig = go.Figure()
        fig.update_layout(height=260, title="Chưa có inventory")
        return fig
    d = by_group.sort_values("co2e", ascending=True)
    fig = go.Figure(go.Bar(y=d["activity_group"], x=d["co2e"], orientation="h", marker_color="#047857", text=[fmt_num(v) for v in d["co2e"]], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=130, r=60, t=10, b=40), xaxis_title="kg CO₂e", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"))
    return fig


def plot_scenario(df_scn: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_scn["year"], y=df_scn["baseline_pcf"], mode="lines+markers", name="Baseline", line=dict(color="#64748b", width=3)))
    fig.add_trace(go.Scatter(x=df_scn["year"], y=df_scn["scenario_pcf"], mode="lines+markers", name="Kịch bản cải thiện", line=dict(color="#047857", width=4)))
    fig.update_layout(height=340, margin=dict(l=40, r=30, t=10, b=40), xaxis_title="Năm", yaxis_title="PCF dự kiến (kg CO₂e)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"), legend=dict(orientation="h", y=1.12))
    return fig


def plot_factor_impact(impact_df: pd.DataFrame) -> go.Figure:
    if impact_df.empty:
        return go.Figure()
    d = impact_df.copy().sort_values("impact", ascending=True)
    labels = d["feature_vi"].map(lambda s: s if len(str(s)) <= 32 else str(s)[:29] + "...")
    colors = np.where(d["impact"] >= 0, "#047857", "#f97316")
    max_abs = max(float(d["impact"].abs().max()), 1e-6)
    fig = go.Figure(go.Bar(y=labels, x=d["impact"], orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in d["impact"]], textposition="outside", cliponaxis=False, hovertemplate="%{customdata}<br>Tác động: %{x:.4f}<extra></extra>", customdata=d["feature_vi"], showlegend=False))
    fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    fig.update_layout(height=410, margin=dict(l=190, r=90, t=10, b=45), xaxis_title="Thay đổi xác suất nhãn dự báo khi tăng/đổi yếu tố", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827", size=13), xaxis=dict(range=[-max_abs*1.55, max_abs*1.55], gridcolor="#e5e7eb"), yaxis=dict(automargin=True))
    return fig


def get_reference_factor(openpcf: pd.DataFrame, product_name: str, country: str) -> float:
    if openpcf.empty:
        return 1.0
    filt = openpcf[openpcf["product_name"].eq(product_name)]
    if not filt.empty:
        ctry = filt[filt["country"].eq(country)]
        if not ctry.empty:
            return float(ctry["openpcf_factor_kgco2e_per_kg"].median())
        return float(filt["openpcf_factor_kgco2e_per_kg"].median())
    return float(openpcf["openpcf_factor_kgco2e_per_kg"].median())


def get_ceda_country_factor(ceda: pd.DataFrame, country: str) -> float:
    if ceda.empty:
        return 0.5
    c = ceda[ceda["country"].eq(country)]
    if not c.empty:
        return float(c["ceda_factor_kgco2e_per_usd"].median())
    return float(ceda["ceda_factor_kgco2e_per_usd"].median())


def benchmark_stats(ref: pd.DataFrame, openpcf: pd.DataFrame, pcf: float, industry_group: str, openpcf_factor: float, weight: float) -> dict[str, Any]:
    subset = ref[ref["industry_group"].eq(industry_group)]
    if subset.empty:
        subset = ref
    industry_median = float(subset[TARGET_COL].median())
    global_median = float(ref[TARGET_COL].median())
    openpcf_est = float(openpcf_factor * max(weight, 0.0001))
    percentile = float((subset[TARGET_COL] <= pcf).mean() * 100)
    ratio = pcf / max(industry_median, 1e-9)
    diff = pcf - industry_median
    if ratio < 0.8:
        conclusion = "PCF ước lượng thấp hơn trung vị ngành. Đây là tín hiệu tích cực, nhưng vẫn nên kiểm tra inventory và hệ số phát thải trước khi dùng cho công bố môi trường."
    elif ratio <= 1.2:
        conclusion = "PCF ước lượng nằm gần vùng trung vị ngành. Sản phẩm ở mức tương đương nhóm tham chiếu, nên xem phần yếu tố ảnh hưởng để tìm cơ hội tối ưu."
    else:
        conclusion = "PCF ước lượng cao hơn trung vị ngành. Nên rà soát vật liệu, năng lượng, vận chuyển và nhà cung cấp để xây dựng kịch bản giảm phát thải."
    return {"industry_median": industry_median, "global_median": global_median, "openpcf_est": openpcf_est, "percentile": percentile, "ratio": ratio, "diff": diff, "n": int(len(subset)), "conclusion": conclusion}


def current_hybrid_pcf(ml_pcf: float, bottom_up_pcf: float) -> float:
    """Kết hợp PCF ML với LCA bottom-up, không dùng mô hình chuỗi thời gian.

    Nếu người dùng có inventory bottom-up thì lấy 70% ML + 30% bottom-up để giữ ổn định.
    Nếu chưa có inventory thì dùng PCF ML làm kết quả hiện tại.
    """
    ml = max(float(ml_pcf), 0.0)
    bottom = max(float(bottom_up_pcf), 0.0)
    if bottom > 0:
        return float(0.70 * ml + 0.30 * bottom)
    return float(ml)


def norm_text(value: Any) -> str:
    """Chuẩn hóa text để lọc/gợi ý sản phẩm gần nhất."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().replace("&", " and ").split())


def text_similarity(a: Any, b: Any) -> float:
    """Điểm gần nhau giữa 2 chuỗi, dùng để gợi ý OpenPCF reference."""
    a_norm = norm_text(a)
    b_norm = norm_text(b)
    if not a_norm or not b_norm:
        return 0.0
    score = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = set(a_norm.replace("/", " ").replace("-", " ").split())
    b_tokens = set(b_norm.replace("/", " ").replace("-", " ").split())
    if a_tokens and b_tokens:
        token_overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
        score = max(score, token_overlap)
    if a_norm in b_norm or b_norm in a_norm:
        score = max(score, 0.92)
    return float(score)


def get_sector_options(ref: pd.DataFrame, openpcf: pd.DataFrame, industry_group: str, industry: str) -> list[str]:
    """Chọn ngành/sản phẩm xong thì chỉ hiện sector phù hợp nhất."""
    mask_exact = ref["industry_group"].astype(str).eq(str(industry_group)) & ref["industry"].astype(str).eq(str(industry))
    sectors = ref.loc[mask_exact, "company_sector"].dropna().astype(str).unique().tolist()

    if not sectors and not openpcf.empty:
        mask_open = openpcf["industry_group"].astype(str).eq(str(industry_group)) & openpcf["industry"].astype(str).eq(str(industry))
        sectors = openpcf.loc[mask_open, "company_sector"].dropna().astype(str).unique().tolist()

    if not sectors:
        mask_group = ref["industry_group"].astype(str).eq(str(industry_group))
        sectors = ref.loc[mask_group, "company_sector"].dropna().astype(str).unique().tolist()

    if not sectors:
        sectors = ref["company_sector"].dropna().astype(str).unique().tolist()

    sectors = sorted([x for x in set(sectors) if x and x.lower() not in {"unknown", "nan"}])
    return sectors or ["Unknown"]


def get_openpcf_reference_options(openpcf: pd.DataFrame, industry_group: str, industry: str, sector: str, limit: int = 80) -> tuple[list[str], str]:
    """Gợi ý OpenPCF reference gần nhất theo ngành/sản phẩm/sector đã chọn."""
    if openpcf.empty:
        return [industry], industry

    data = openpcf.copy()
    query = f"{industry} {industry_group} {sector}"

    # Ưu tiên các dòng cùng group/industry/sector trước, sau đó mới fuzzy matching toàn bộ.
    same_industry = data[data["industry"].astype(str).str.lower().eq(str(industry).lower())]
    same_group = data[data["industry_group"].astype(str).str.lower().eq(str(industry_group).lower())]
    same_sector = data[data["company_sector"].astype(str).str.lower().eq(str(sector).lower())]

    if not same_industry.empty:
        candidates = same_industry.copy()
    elif not same_group.empty:
        candidates = same_group.copy()
    elif not same_sector.empty:
        candidates = same_sector.copy()
    else:
        candidates = data.copy()

    candidates = candidates[["product_name", "industry", "industry_group", "company_sector", "product_detail"]].dropna(subset=["product_name"]).drop_duplicates()
    candidates["_score"] = candidates.apply(
        lambda r: max(
            text_similarity(r.get("product_name", ""), industry),
            text_similarity(r.get("industry", ""), industry),
            0.72 * text_similarity(r.get("product_detail", ""), query),
            0.65 * text_similarity(r.get("industry_group", ""), industry_group),
            0.60 * text_similarity(r.get("company_sector", ""), sector),
        ),
        axis=1,
    )

    candidates = candidates.sort_values(["_score", "product_name"], ascending=[False, True])
    ordered = candidates["product_name"].astype(str).drop_duplicates().head(limit).tolist()

    if not ordered:
        ordered = data["product_name"].dropna().astype(str).drop_duplicates().head(limit).tolist()

    suggested = ordered[0] if ordered else industry
    return ordered or [industry], suggested

def sidebar_inputs(sources: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ref = sources["training"]
    openpcf = sources["openpcf"]
    ceda = sources["ceda"]
    with st.sidebar:
        st.markdown("## 🌱 EcoPredict Carbon")
        st.caption("Nhập thông tin sản phẩm để ước lượng PCF, so sánh benchmark và mô phỏng kịch bản.")
        st.markdown("---")

        year = st.number_input("Năm báo cáo / năm kịch bản", min_value=1900, max_value=2050, value=2030, step=1)
        weight = st.number_input("Khối lượng sản phẩm (kg)", min_value=0.0001, value=float(ref["product_weight_kg"].median()), step=0.1, format="%.4f")

        countries = sorted(set(ref["country"].dropna().astype(str).tolist()) | set(ceda.get("country", pd.Series(dtype=str)).dropna().astype(str).tolist()))
        country = st.selectbox("Quốc gia/khu vực", countries if countries else ["Vietnam", "Australia", "United States Of America"])

        # 1) Chọn Nhóm ngành → tự lọc Ngành/sản phẩm
        groups = sorted(ref["industry_group"].dropna().astype(str).unique().tolist())
        industry_group = st.selectbox("Nhóm ngành", groups)

        inds = sorted(ref.loc[ref["industry_group"].astype(str).eq(str(industry_group)), "industry"].dropna().astype(str).unique().tolist())
        if not inds:
            inds = sorted(ref["industry"].dropna().astype(str).unique().tolist())[:200]
        industry = st.selectbox("Ngành/sản phẩm", inds, help="Danh sách này đã được lọc theo Nhóm ngành đã chọn.")

        # 2) Chọn Ngành/sản phẩm → tự lọc Sector phù hợp
        sector_options = get_sector_options(ref, openpcf, industry_group, industry)
        sector = st.selectbox("Sector", sector_options, help="Sector được lọc tự động theo Nhóm ngành và Ngành/sản phẩm.")

        protocols = sorted(ref["protocol_simple"].dropna().astype(str).unique().tolist())
        protocol = st.selectbox("Chuẩn/nguồn PCF", protocols)

        # 3) Chọn Ngành/sản phẩm → gợi ý OpenPCF reference gần nhất
        product_options, suggested_ref = get_openpcf_reference_options(openpcf, industry_group, industry, sector)
        suggested_idx = product_options.index(suggested_ref) if suggested_ref in product_options else 0
        ref_product = st.selectbox(
            "Vật liệu/sản phẩm tham chiếu OpenPCF",
            product_options,
            index=suggested_idx,
            help="Hệ thống gợi ý reference gần nhất theo ngành/sản phẩm, nhóm ngành và sector đã chọn.",
        )
        st.caption(f"Gợi ý gần nhất: {suggested_ref}")

        st.markdown("### Tỷ trọng vòng đời")
        st.caption("Mặc định theo cấu hình học thuật: Upstream 40%, Operations 30%, Downstream 15%, Transport 10%, End-of-life 5%.")
        upstream = st.slider("Upstream (%)", 0, 100, 40)
        operations = st.slider("Operations (%)", 0, 100, 30)
        downstream = st.slider("Downstream (%)", 0, 100, 15)
        transport = st.slider("Transport (%)", 0, 100, 10)
        eol = st.slider("End-of-life (%)", 0, 100, 5)
        lifecycle_total = upstream + operations + downstream + transport + eol
        if lifecycle_total == 100:
            st.success("Tổng tỷ trọng vòng đời = 100%")
        else:
            st.warning(f"Tổng hiện tại = {lifecycle_total}%. Hệ thống sẽ tự chuẩn hóa về 100% khi tính toán.")

        st.markdown("### Kịch bản cải thiện")
        renewable = st.slider("Tăng điện tái tạo (%)", 0, 100, 20)
        material_red = st.slider("Giảm vật liệu/khối lượng (%)", 0, 80, 10)
        logistics = st.slider("Cải thiện vận chuyển (%)", 0, 80, 10)
        supplier = st.slider("Tối ưu supplier/geography (%)", 0, 80, 5)
    return {
        "year": int(year), "weight": float(weight), "country": country, "industry_group": industry_group,
        "industry": industry, "sector": sector, "protocol": protocol, "ref_product": ref_product,
        "upstream": upstream, "operations": operations, "downstream": downstream, "transport": transport, "eol": eol,
        "renewable": renewable, "material_red": material_red, "logistics": logistics, "supplier": supplier,
    }

def prediction_page(sources: dict[str, pd.DataFrame], package: dict[str, Any]) -> None:
    render_hero()
    ref = sources["training"]
    openpcf = sources["openpcf"]
    ceda = sources["ceda"]
    inputs = sidebar_inputs(sources)

    lifecycle_total = max(inputs["upstream"] + inputs["operations"] + inputs["downstream"] + inputs["transport"] + inputs["eol"], 1)
    up_frac = inputs["upstream"] / lifecycle_total
    op_frac = inputs["operations"] / lifecycle_total
    down_frac = inputs["downstream"] / lifecycle_total
    transport_frac = inputs["transport"] / lifecycle_total
    eol_frac = inputs["eol"] / lifecycle_total
    openpcf_factor = get_reference_factor(openpcf, inputs["ref_product"], inputs["country"])
    ceda_factor = get_ceda_country_factor(ceda, inputs["country"])

    st.markdown("<div class='scope-box'><b>Mục tiêu hệ thống:</b> ước lượng PCF sản phẩm theo hướng LCA/ISO-oriented decision-support. Kết quả dùng để phân tích, benchmark và mô phỏng kịch bản; không phải chứng nhận ISO/EPD chính thức.</div>", unsafe_allow_html=True)

    # Organized bottom-up UI.
    with st.expander("Kiểm kê vòng đời bottom-up: vật liệu, năng lượng, vận chuyển, bao bì, cuối vòng đời", expanded=False):
        st.caption("Có thể chỉnh bảng dưới để tính PCF theo công thức PCF = Σ(activity data × emission factor). Nếu chưa có dữ liệu thật, hệ thống vẫn dùng ML + hệ số tham chiếu.")
        inv_default = DEMO_EMISSION_FACTORS.copy()
        inv_default.loc[inv_default["activity_group"].eq("Material"), "amount"] = inputs["weight"] * 0.35
        inv_default.loc[inv_default["activity_group"].eq("Energy"), "amount"] = inputs["weight"] * 2.0
        inv_default.loc[inv_default["activity_group"].eq("Transport"), "amount"] = inputs["weight"] * 0.15
        inv_default.loc[inv_default["activity_group"].eq("Packaging"), "amount"] = inputs["weight"] * 0.05
        inv_default.loc[inv_default["activity_group"].eq("End-of-life"), "amount"] = inputs["weight"] * 0.10
        inventory = st.data_editor(
            inv_default[["stage", "activity_group", "activity_name", "unit", "amount", "emission_factor", "source", "quality"]],
            num_rows="dynamic",
            use_container_width=True,
            key="inventory_editor",
        )
        lca = calculate_lca_bottom_up(inventory)
        l1, l2 = st.columns([1, 1])
        with l1:
            st.metric("PCF bottom-up", f"{fmt_num(lca['total_pcf'])} kg CO₂e")
        with l2:
            st.caption("Khi có inventory đáng tin cậy, PCF Hybrid sẽ ưu tiên kết hợp ML với LCA bottom-up để thực tế hơn.")
    if "lca" not in locals():
        lca = {"total_pcf": 0.0, "by_group": pd.DataFrame(), "detail": pd.DataFrame()}

    lca_proxy = lca["total_pcf"] if lca["total_pcf"] > 0 else inputs["weight"] * openpcf_factor
    input_row = build_input_row(
        ref,
        year=inputs["year"], product_weight_kg=inputs["weight"], country=inputs["country"],
        industry_group=inputs["industry_group"], industry=inputs["industry"], company_sector=inputs["sector"],
        protocol=inputs["protocol"], protocol_simple=inputs["protocol"], weight_source="Functional unit 1 kg",
        stage_level_available="Estimated", upstream_frac=up_frac, operations_frac=op_frac, downstream_frac=down_frac,
        transport_frac=transport_frac, end_of_life_frac=eol_frac, product_name=inputs["ref_product"],
        product_detail="User scenario input", data_source="OpenPCF by Terralytiq", system_boundary="Cradle-to-gate factor",
        functional_unit_type="1 kg material/product", openpcf_factor_kgco2e_per_kg=openpcf_factor,
        ceda_factor_kgco2e_per_usd=ceda_factor, lca_proxy_pcf=lca_proxy,
        renewable_energy_pct=inputs["renewable"], material_reduction_pct=inputs["material_red"], transport_improvement_pct=inputs["logistics"],
    )
    pred = predict_with_package(package, input_row)

    # Hybrid hiện tại: kết hợp ML và LCA bottom-up, không dùng mô hình chuỗi thời gian.
    hybrid = current_hybrid_pcf(pred["pcf"], lca["total_pcf"])
    ood = check_ood(input_row, package["metadata"].get("ood_profile", {}))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-title'>Phân loại carbon</div><div class='kpi-value'>{pred['label_vi']}</div><div class='kpi-note'>Mô hình: {package['metadata'].get('best_classifier_name','')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCF ML ước lượng</div><div class='kpi-value'>{fmt_num(pred['pcf'])}</div><div class='kpi-note'>kg CO₂e; khoảng tham khảo: {fmt_num(pred['p10'])}–{fmt_num(pred['p90'])}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCF Hybrid</div><div class='kpi-value'>{fmt_num(hybrid)}</div><div class='kpi-note'>Kết hợp ML và LCA bottom-up; không dùng mô hình chuỗi thời gian cho PCF hiện tại.</div></div>", unsafe_allow_html=True)
    with c4:
        confidence_note = ood.get("message", "Mức tin cậy được tính từ phạm vi dữ liệu tham chiếu và giả định đầu vào.")
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Mức tin cậy kịch bản</div><div class='kpi-value'>{ood['confidence']}</div><div class='kpi-note'>{confidence_note}</div></div>", unsafe_allow_html=True)

    with st.expander("Xem chi tiết kỹ thuật về phạm vi dữ liệu", expanded=False):
        st.write("Mục này dành cho người kiểm định hoặc người phát triển. Người dùng nghiệp vụ chỉ cần xem mức tin cậy kịch bản ở thẻ kết quả.")
        if ood["detail"]:
            st.write(ood["detail"])
        else:
            st.write("Input hiện nằm trong vùng dữ liệu tham chiếu chính của mô hình.")

    st.markdown("<div class='card'><div class='section-title'>Diễn giải kết quả</div><div class='info-box'>Hệ thống dùng PCF làm lõi, LCA làm khung tính toán, còn Machine Learning hỗ trợ ước lượng, benchmark, phát hiện bất thường và giải thích yếu tố ảnh hưởng. Kết quả phù hợp cho phân tích sơ bộ và ra quyết định cải thiện phát thải carbon.</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Xác suất phân loại</div><div class='card-subtitle'>Xác suất mô hình xếp sản phẩm vào các mức phát thải.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_probability_bar(pred["proba"]), use_container_width=True, config=PLOTLY_CONFIG, key="probability_bar")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Tỷ trọng vòng đời</div><div class='card-subtitle'>Stacked bar hiển thị cơ cấu vòng đời sau khi chuẩn hóa tổng tỷ trọng về 100%.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_lifecycle_stacked({"upstream": up_frac, "operations": op_frac, "downstream": down_frac, "transport": transport_frac, "eol": eol_frac}), use_container_width=True, config=PLOTLY_CONFIG, key="lifecycle_stacked")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>So sánh PCF với ngành và dữ liệu tham chiếu</div><div class='card-subtitle'>Benchmark giúp đặt kết quả vào bối cảnh ngành; không thay thế kiểm kê LCA chính thức.</div>", unsafe_allow_html=True)
    stats = benchmark_stats(ref, openpcf, hybrid, inputs["industry_group"], openpcf_factor, inputs["weight"])
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(plot_benchmark({"product": hybrid, "industry": stats["industry_median"], "global": stats["global_median"], "openpcf": stats["openpcf_est"]}), use_container_width=True, config=PLOTLY_CONFIG, key="benchmark_chart")
    with right:
        st.markdown(f"""
        <div class='insight-grid'>
          <div class='insight-card'><div class='insight-label'>Vị trí trong nhóm ngành</div><div class='insight-value'>{stats['percentile']:.1f}%</div><div class='insight-note'>Tỷ lệ mẫu cùng ngành có PCF thấp hơn hoặc bằng sản phẩm này.</div></div>
          <div class='insight-card'><div class='insight-label'>Tỷ lệ so với trung vị ngành</div><div class='insight-value'>{stats['ratio']:.2f}x</div><div class='insight-note'>Nhỏ hơn 1 là thấp hơn benchmark ngành.</div></div>
          <div class='insight-card'><div class='insight-label'>Số mẫu tham chiếu</div><div class='insight-value'>{stats['n']}</div><div class='insight-note'>Số mẫu cùng nhóm ngành trong dữ liệu hợp nhất.</div></div>
          <div class='insight-card'><div class='insight-label'>OpenPCF proxy</div><div class='insight-value'>{fmt_num(stats['openpcf_est'])}</div><div class='insight-note'>Khối lượng × hệ số OpenPCF gần nhất.</div></div>
        </div>
        <div class='info-box'>{stats['conclusion']}</div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>PCF bottom-up breakdown</div><div class='card-subtitle'>Phân rã PCF theo các nhóm hoạt động người dùng nhập trong inventory.</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_lca_breakdown(lca["by_group"]), use_container_width=True, config=PLOTLY_CONFIG, key="lca_breakdown")
    if not lca["detail"].empty:
        st.dataframe(lca["detail"][["stage", "activity_group", "activity_name", "unit", "amount", "emission_factor", "co2e", "quality"]], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Mô phỏng kịch bản phát thải theo thời gian</div><div class='card-subtitle'>Mô hình kịch bản tham số: PCF tương lai được điều chỉnh theo các driver vòng đời như vật liệu, năng lượng, vận chuyển và cuối vòng đời. Đây là mô phỏng tham khảo, không phải dự báo tuyệt đối.</div>", unsafe_allow_html=True)

    scenario_labels = {
        "baseline": "Baseline - chính sách hiện tại",
        "net_zero": "Net Zero 2050",
        "pessimistic": "Pessimistic - chuyển dịch chậm",
        "custom": "Custom - theo slider người dùng",
    }
    scenario_key = st.selectbox(
        "Kịch bản mô phỏng",
        options=list(scenario_labels.keys()),
        format_func=lambda x: scenario_labels[x],
        index=1,
        key="scenario_projection_key",
    )
    default_target_year = min(max(int(inputs["year"]), 2030), 2050)
    target_year = st.number_input(
        "Năm mục tiêu mô phỏng",
        min_value=2025,
        max_value=2050,
        value=default_target_year,
        step=1,
        key="scenario_projection_target_year",
    )

    lifecycle_shares = normalize_lifecycle_shares(
        upstream=inputs["upstream"],
        operations=inputs["operations"],
        downstream=inputs["downstream"],
        transport=inputs["transport"],
        end_of_life=inputs["eol"],
    )
    custom_2050 = build_custom_2050_factors(
        renewable_gain_pct=inputs["renewable"],
        material_reduction_pct=inputs["material_red"],
        logistics_gain_pct=inputs["logistics"],
        supplier_factor_pct=inputs["supplier"],
    )

    baseline_year_for_scenario = 2025
    pathway_df = build_projection_pathway(
        baseline_pcf=float(hybrid),
        baseline_year=baseline_year_for_scenario,
        target_year=int(target_year),
        scenario_key=scenario_key,
        lifecycle_shares=lifecycle_shares,
        custom_2050_factors=custom_2050 if scenario_key == "custom" else None,
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        last_value = float(pathway_df["projected_pcf"].iloc[-1]) if not pathway_df.empty else float(hybrid)
        st.markdown(f"<div class='insight-card'><div class='insight-label'>PCF năm mục tiêu</div><div class='insight-value'>{fmt_num(last_value)}</div><div class='insight-note'>kg CO₂e theo kịch bản đã chọn.</div></div>", unsafe_allow_html=True)
    with sc2:
        reduction_pct = (1 - last_value / max(float(hybrid), 1e-9)) * 100
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Mức thay đổi so với baseline</div><div class='insight-value'>{reduction_pct:+.1f}%</div><div class='insight-note'>Âm là tăng phát thải; dương là giảm phát thải.</div></div>", unsafe_allow_html=True)
    with sc3:
        u_pct = float(pathway_df["uncertainty_pct"].iloc[-1]) if not pathway_df.empty else 15.0
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Khoảng bất định</div><div class='insight-value'>±{u_pct:.0f}%</div><div class='insight-note'>Tăng dần khi năm mô phỏng xa hơn.</div></div>", unsafe_allow_html=True)

    fig_scenario = go.Figure()
    fig_scenario.add_trace(
        go.Scatter(
            x=pathway_df["year"],
            y=pathway_df["projected_pcf"],
            mode="lines+markers",
            name="PCF mô phỏng",
            line=dict(color="#047857", width=4),
        )
    )
    fig_scenario.add_trace(
        go.Scatter(
            x=list(pathway_df["year"]) + list(pathway_df["year"])[::-1],
            y=list(pathway_df["upper"]) + list(pathway_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(16,185,129,0.18)",
            line=dict(color="rgba(255,255,255,0)", width=0),
            name="Khoảng bất định",
            hoverinfo="skip",
        )
    )
    fig_scenario.update_layout(
        height=380,
        margin=dict(l=40, r=30, t=20, b=40),
        xaxis_title="Năm",
        yaxis_title="PCF dự kiến (kg CO₂e)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827"),
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_scenario, use_container_width=True, config=PLOTLY_CONFIG, key="scenario_projection_chart")

    st.markdown("""
    <div class='info-box'>
    <b>Về bản chất:</b> đây là mô phỏng kịch bản tham số, không phải dự báo tuyệt đối chắc chắn. 
    Hệ thống lấy PCF nền từ ML/LCA hybrid, sau đó điều chỉnh theo driver vòng đời. 
    Khoảng bất định tham khảo: 2025–2030 ±15%, 2031–2040 ±20%, 2041–2050 ±30%.
    </div>
    """, unsafe_allow_html=True)

    show_cols = [
        "year", "scenario", "baseline_pcf", "projected_pcf", "lower", "upper",
        "uncertainty_pct", "combined_factor", "upstream_factor", "operations_factor",
        "transport_factor", "eol_factor",
    ]
    pathway_show = pathway_df[show_cols].copy()
    for col in pathway_show.select_dtypes(include=["float", "float64", "float32"]).columns:
        pathway_show[col] = pathway_show[col].round(3)
    st.dataframe(pathway_show, use_container_width=True, hide_index=True)

    with st.expander("Xem giả định kịch bản 2050", expanded=False):
        st.dataframe(build_assumptions_table(custom_2050_factors=custom_2050), use_container_width=True, hide_index=True)
        st.caption("Các hệ số 2050 là giả định minh bạch cho prototype nghiên cứu. Ví dụ operations_2050 = 0.25 nghĩa là phát thải giai đoạn vận hành còn 25% so với baseline.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Giải thích yếu tố ảnh hưởng</div><div class='card-subtitle'>Hiển thị top 6 yếu tố cục bộ để tránh chồng chữ; bảng bên dưới giữ tên đầy đủ.</div>", unsafe_allow_html=True)
    impact = get_local_factor_impact(package, input_row, n_top=6)
    if impact.empty:
        st.info("Không tạo được giải thích cục bộ cho dự báo này.")
    else:
        top_names = ", ".join(impact["feature_vi"].head(3).tolist())
        st.markdown(f"<div class='info-box'>Các yếu tố ảnh hưởng mạnh nhất trong dự báo hiện tại gồm: {top_names}. Thanh xanh làm tăng xác suất nhãn hiện tại, thanh cam kéo dự báo theo chiều ngược lại.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_factor_impact(impact), use_container_width=True, config=PLOTLY_CONFIG, key="factor_impact")
        st.dataframe(impact[["feature_vi", "impact", "direction_vi"]].rename(columns={"feature_vi": "Tên yếu tố", "impact": "Mức tác động", "direction_vi": "Chiều tác động"}), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def data_page(sources: dict[str, pd.DataFrame]) -> None:
    render_hero()
    df = sources["training"]
    carbon = sources["carbon"]
    openpcf = sources["openpcf"]
    ceda = sources["ceda"]
    st.markdown("<div class='card'><div class='section-title'>Tổng quan dữ liệu hợp nhất</div><div class='card-subtitle'>Carbon Catalogue làm dữ liệu lịch sử, OpenPCF bổ sung PCF activity-based, Open CEDA bổ sung hệ số phát thải theo quốc gia/ngành.</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    vals = [len(carbon), len(openpcf), len(ceda), df["country"].nunique(), df[TARGET_COL].median()]
    labs = ["Carbon Catalogue", "OpenPCF", "Open CEDA factors", "Quốc gia/khu vực", "Median PCF"]
    for c, lab, val in zip(cols, labs, vals):
        with c:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>{lab}</div><div class='kpi-value'>{fmt_num(val)}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><div class='section-title'>Phân phối PCF theo thang log</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x=np.log1p(df[TARGET_COL]), color="data_source", nbins=60, labels={"x": "log(PCF + 1)", "data_source": "Nguồn"})
        fig.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_{abs(hash(str(fig.to_dict()))) % 10_000_000}")
        st.markdown("<div class='small-muted'>PCF lệch phải mạnh nên histogram tuyến tính sẽ dồn dữ liệu về một phía. Log-scale giúp nhìn rõ phân phối hơn.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='section-title'>Top nhóm ngành theo số mẫu</div>", unsafe_allow_html=True)
        counts = df["industry_group"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(y=counts.index, x=counts.values, orientation="h", marker_color="#10b981", text=counts.values, textposition="outside"))
        fig.update_layout(height=370, margin=dict(l=180, r=40, t=5, b=35), xaxis_title="Số mẫu", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_{abs(hash(str(fig.to_dict()))) % 10_000_000}")
        st.markdown("</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='card'><div class='section-title'>Xu hướng PCF theo nguồn dữ liệu</div>", unsafe_allow_html=True)
        t = df.groupby(["year", "data_source"])[TARGET_COL].median().reset_index()
        fig = px.line(t, x="year", y=TARGET_COL, color="data_source", markers=True)
        fig.update_layout(height=360, yaxis_type="log", yaxis_title="Median PCF (log)", xaxis_title="Năm", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_{abs(hash(str(fig.to_dict()))) % 10_000_000}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><div class='section-title'>OpenPCF: top hệ số theo sản phẩm</div>", unsafe_allow_html=True)
        if not openpcf.empty:
            top = openpcf.groupby("product_name")["openpcf_factor_kgco2e_per_kg"].median().sort_values(ascending=False).head(10).sort_values()
            fig = go.Figure(go.Bar(y=top.index, x=top.values, orientation="h", marker_color="#047857"))
            fig.update_layout(height=360, margin=dict(l=180, r=40, t=5, b=35), xaxis_title="kg CO₂e/kg", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_{abs(hash(str(fig.to_dict()))) % 10_000_000}")
        else:
            st.info("Chưa có file OpenPCF.")
        st.markdown("</div>", unsafe_allow_html=True)


def lca_iso_page(sources: dict[str, pd.DataFrame]) -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>LCA/ISO-oriented decision-support layer</div><div class='card-subtitle'>Phần này giúp hệ thống tiến gần hơn đến tư duy ISO 14040/14044/14067: có goal & scope, inventory, data quality và audit trail. Đây vẫn là hỗ trợ quyết định, không phải chứng nhận chính thức.</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'><b>Định vị đúng:</b> PCF là lõi làm bài; LCA là nền học thuật mở rộng; ISO là định hướng minh bạch; EPD là đích ứng dụng doanh nghiệp trong tương lai.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Goal & Scope</div>", unsafe_allow_html=True)
        st.text_input("Mục tiêu nghiên cứu", "Ước tính PCF, benchmark ngành và so sánh kịch bản giảm phát thải")
        st.text_input("Functional unit", "1 sản phẩm / 1 kg vật liệu / 1 đơn vị chức năng")
        st.selectbox("System boundary", ["Cradle-to-gate", "Cradle-to-grave", "Gate-to-gate", "Reported boundary"])
        st.selectbox("Allocation method", ["Mass allocation", "Economic allocation", "Energy allocation", "Not applicable"])
        st.text_area("Cut-off criteria", "Ghi rõ tiêu chí loại trừ các dòng dữ liệu nhỏ hoặc thiếu độ tin cậy.")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Data Quality Rating</div>", unsafe_allow_html=True)
        temporal = st.slider("Temporal representativeness", 1, 5, 3)
        geo = st.slider("Geographical representativeness", 1, 5, 3)
        tech = st.slider("Technological representativeness", 1, 5, 3)
        complete = st.slider("Completeness", 1, 5, 3)
        reliab = st.slider("Reliability", 1, 5, 3)
        score = np.mean([temporal, geo, tech, complete, reliab])
        st.metric("Điểm chất lượng dữ liệu", f"{score:.1f}/5")
        st.caption("Điểm này giúp người dùng hiểu kết quả có độ chắc chắn đến đâu, thay vì chỉ nhìn một con số PCF duy nhất.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='section-title'>Audit trail cần có khi triển khai thật</div>", unsafe_allow_html=True)
    st.markdown("""
    - Input sản phẩm, inventory, emission factors, nguồn factor và năm dữ liệu.
    - Phiên bản model, phiên bản dữ liệu, ngày chạy và người thực hiện.
    - Kết quả PCF, khoảng bất định, benchmark, OOD score và giải thích yếu tố.
    - Trạng thái dự án: draft / review / approved.
    - Nhận xét reviewer nếu hướng tới EPD-ready hoặc công bố môi trường.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


def ml_page(package: dict[str, Any] | None) -> None:
    render_hero()
    if package is None:
        train_now_panel()
        return
    meta = package["metadata"]
    st.markdown("<div class='card'><div class='section-title'>Thông tin mô hình ML</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <span class='badge'>Best classifier: {meta.get('best_classifier_name')}</span>
    <span class='badge'>Best regressor: {meta.get('best_regressor_name')}</span>
    <span class='badge'>{meta.get('split_strategy')}</span>
    <span class='badge'>Uncertainty P10–P90</span>
    <span class='badge'>Permutation + SHAP</span>
    <span class='badge'>Scenario-based projection</span>
    """, unsafe_allow_html=True)
    st.markdown("<div class='scope-box'>ML trong hệ thống không thay thế LCA. ML hỗ trợ dự báo, hiệu chỉnh, phát hiện bất thường, benchmark và giải thích yếu tố ảnh hưởng.</div>", unsafe_allow_html=True)
    st.markdown("<div class='soft-warning'><b>Lưu ý học thuật:</b> Nếu metric đạt rất cao hoặc gần 1.0, không nên hiểu là mô hình hoàn hảo. Với bài toán này, nhãn Low/Medium/High được xây dựng từ ngưỡng PCF; cần kiểm thử bổ sung theo product/country/time hold-out để đánh giá tổng quát hóa thực tế.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Metric phân loại",
        "Metric hồi quy",
        "SHAP / XAI",
        "Sensitivity",
        "Tuning & Tests",
        "Kiểm soát chất lượng ML",
    ])
    with tab1:
        st.dataframe(pd.DataFrame(meta.get("classification_metrics", [])), use_container_width=True)
        for img in ["outputs/figures/model_classification_f1_comparison.png", "outputs/figures/model_confusion_matrix.png", "outputs/figures/model_roc_curve.png"]:
            if Path(img).exists():
                st.image(img, use_container_width=True)
        st.caption("Metric cao là tín hiệu tốt trong phạm vi test set, nhưng vẫn cần disclaimer vì dữ liệu PCF có thể có feature liên quan gần với target/emission factor.")
    with tab2:
        st.dataframe(pd.DataFrame(meta.get("regression_metrics", [])), use_container_width=True)
        for img in ["outputs/figures/model_regression_median_ape_comparison.png", "outputs/figures/model_regression_actual_vs_predicted.png", "outputs/figures/model_regression_residuals.png", "outputs/figures/model_regression_residual_distribution.png"]:
            if Path(img).exists():
                st.image(img, use_container_width=True)
    with tab3:
        shap_info = meta.get("shap_explanation", {})
        st.markdown("""
        **SHAP** được bổ sung để tăng khả năng giải thích mô hình. Permutation Importance trả lời câu hỏi *feature nào quan trọng toàn cục*, còn SHAP giúp nhìn rõ *feature đẩy dự báo theo hướng nào*.
        """)
        if shap_info:
            st.json(shap_info)
        shap_images = [
            ("SHAP Feature Ranking", "outputs/figures/model_shap_summary_bar.png"),
            ("SHAP Beeswarm", "outputs/figures/model_shap_beeswarm.png"),
            ("SHAP Waterfall - một mẫu dự báo", "outputs/figures/model_shap_waterfall_first_sample.png"),
            ("SHAP Dependence - feature quan trọng nhất", "outputs/figures/model_shap_dependence_top_feature.png"),
            ("Permutation Importance", "outputs/figures/model_permutation_importance.png"),
        ]
        found = False
        for title, img in shap_images:
            if Path(img).exists():
                found = True
                st.markdown(f"### {title}")
                st.image(img, use_container_width=True)
        if Path("outputs/tables/shap_feature_importance.csv").exists():
            st.markdown("### Bảng SHAP feature importance")
            st.dataframe(pd.read_csv("outputs/tables/shap_feature_importance.csv").head(30), use_container_width=True)
        if not found:
            st.info("Chưa có SHAP plots. Hãy chạy lại: python train_advanced_models.py sau khi cài shap trong requirements.txt.")
    with tab4:
        st.markdown("""
        **Sensitivity Analysis** giúp trả lời: nếu emission factor hoặc driver vòng đời thay đổi ±20%, PCF thay đổi bao nhiêu.  
        Đây là phần quan trọng để tăng độ sâu môi trường/LCA và minh bạch hóa uncertainty của kết quả.
        """)
        st.code("python sensitivity_analysis.py", language="powershell")
        sensitivity_images = [
            ("Tornado chart - yếu tố nhạy nhất", "outputs/figures/sensitivity_tornado.png"),
            ("2-way heatmap - vật liệu và năng lượng", "outputs/figures/sensitivity_heatmap_2way.png"),
            ("Scenario sensitivity", "outputs/figures/sensitivity_scenarios.png"),
        ]
        found_sens = False
        for title, img in sensitivity_images:
            if Path(img).exists():
                found_sens = True
                st.markdown(f"### {title}")
                st.image(img, use_container_width=True)
        for table in [
            "outputs/tables/sensitivity_tornado.csv",
            "outputs/tables/sensitivity_heatmap_2way.csv",
            "outputs/tables/sensitivity_scenarios.csv",
        ]:
            if Path(table).exists():
                st.markdown(f"### {Path(table).name}")
                st.dataframe(pd.read_csv(table).head(50), use_container_width=True)
        if not found_sens:
            st.info("Chưa có sensitivity plots. Chạy: python sensitivity_analysis.py hoặc python train_advanced_models.py để sinh ảnh/bảng.")

    with tab5:
        st.markdown("""
        **Hyperparameter tuning** dùng `GridSearchCV` để so sánh Random Forest mặc định và Random Forest đã tinh chỉnh.  
        Bản mới dùng `f1_macro` làm scoring chính và có thể kết hợp SMOTE trong pipeline để xử lý mất cân bằng lớp.
        """)
        st.code("python hyperparameter_tuning.py", language="powershell")
        if Path("outputs/tables/hyperparameter_tuning_results.csv").exists():
            st.dataframe(pd.read_csv("outputs/tables/hyperparameter_tuning_results.csv"), use_container_width=True)
        else:
            st.info("Chưa có bảng tuning. Chạy python hyperparameter_tuning.py để tạo.")
        if Path("outputs/figures/hyperparameter_tuning_comparison.png").exists():
            st.image("outputs/figures/hyperparameter_tuning_comparison.png", use_container_width=True)

        st.markdown("### Unit tests")
        st.code("python -m pytest tests -v --cov=carbon_utils --cov=scenario_projection --cov-report=term-missing", language="powershell")
        st.caption("Bộ test kiểm tra xử lý dữ liệu, LCA bottom-up, scenario projection, nhãn Low/Medium/High, outlier detection và các hàm tiện ích.")
    with tab6:
        st.markdown("""
        - Split theo thời gian/source-aware, không dùng random 80/20 đơn giản.
        - Nhãn Low/Medium/High được tạo từ ngưỡng Q25/Q75 trên train set để giảm leakage.
        - Có hồi quy PCF và phân loại mức phát thải.
        - Có uncertainty interval dựa trên residual và ensemble regressors.
        - Có OOD check nhưng hiển thị mềm dưới dạng mức tin cậy kịch bản.
        - Có permutation importance, SHAP và giải thích cục bộ top 6 yếu tố ảnh hưởng.
        - Có class imbalance diagnostics, SMOTE/class_weight và per-class recall để tránh bỏ sót lớp phát thải cao.
        - Có sensitivity analysis để đánh giá độ nhạy PCF với vật liệu, năng lượng, vận chuyển và kịch bản.
        - Có GridSearchCV script để so sánh default vs tuned model.
        - Có unit tests để tăng độ tin cậy khi chỉnh sửa code.
        """)
        st.markdown("<div class='info-box'><b>Phạm vi đúng:</b> hệ thống là prototype nghiên cứu PCF/LCA-oriented, dùng để screening, benchmark và hỗ trợ quyết định; chưa phải công cụ chứng nhận ISO/EPD hoặc khai báo ESG chính thức.</div>", unsafe_allow_html=True)


def guide_page() -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>Quy trình hệ thống EcoPredict Carbon</div>", unsafe_allow_html=True)
    st.markdown("""
    **1. Dữ liệu:** Carbon Catalogue là dữ liệu lịch sử; OpenPCF bổ sung hệ số PCF activity-based theo sản phẩm/quốc gia; Open CEDA bổ sung hệ số phát thải theo quốc gia/ngành.

    **2. LCA bottom-up:** người dùng có thể nhập vật liệu, năng lượng, vận chuyển, bao bì và cuối vòng đời. Hệ thống tính PCF sơ bộ bằng `activity data × emission factor`.

    **3. Machine Learning:** mô hình phân loại Low/Medium/High và mô hình hồi quy PCF học từ dữ liệu hợp nhất. ML hỗ trợ dự báo, benchmark, giải thích yếu tố và phát hiện input ngoài miền dữ liệu.

    **4. Kịch bản tương lai:** người dùng nhập năm đến 2050, chọn các đòn bẩy như tăng điện tái tạo, giảm vật liệu, cải thiện vận chuyển và tối ưu supplier để mô phỏng xu hướng PCF.

    **5. Giao diện doanh nghiệp:** kết quả hiển thị bằng KPI, benchmark ngành, breakdown inventory, đường kịch bản và bảng giải thích yếu tố; không bắt buộc xuất PDF/Excel.
    """)
    st.code("python carbon_eda.py\npython train_advanced_models.py\npython -m streamlit run app.py", language="powershell")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    sources, package = bootstrap()
    with st.sidebar:
        page = st.radio("", ["Dự báo", "Dữ liệu", "LCA/ISO nâng cao", "ML & đánh giá", "Quy trình"], index=0)
    if page == "Dự báo":
        if package is None:
            render_hero()
            train_now_panel()
        else:
            prediction_page(sources, package)
    elif page == "Dữ liệu":
        data_page(sources)
    elif page == "LCA/ISO nâng cao":
        lca_iso_page(sources)
    elif page == "ML & đánh giá":
        ml_page(package)
    else:
        guide_page()


if __name__ == "__main__":
    main()
