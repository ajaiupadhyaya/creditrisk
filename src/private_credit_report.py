"""
Standalone HTML report for the synthetic private-credit fragility track.
Reads CSVs under private_credit/data/ (produced by private_credit/python scripts).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot

warnings.filterwarnings("ignore")

THEME = {
    "paper": "#f7f5f0",
    "ink": "#1c1917",
    "muted": "#57534e",
    "rule": "#d6d3cd",
    "accent": "#b45309",
    "accent2": "#1e3a5f",
    "font": "Iowan Old Style, Georgia, Palatino Linotype, Book Antiqua, serif",
    "font_sans": "system-ui, -apple-system, Segoe UI, Helvetica Neue, Arial, sans-serif",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return _repo_root() / "private_credit" / "data"


def build_fragility_report(output_path: Path | None = None) -> Path | None:
    """Build outputs/private_credit/fragility_summary.html if inputs exist."""
    data = _data_dir()
    fs = data / "fragility_scores.csv"
    mp = data / "ml_predictions.csv"
    if not fs.exists() or not mp.exists():
        return None

    frag = pd.read_csv(fs)
    preds = pd.read_csv(mp)

    if output_path is None:
        out_dir = _repo_root() / "outputs" / "private_credit"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "fragility_summary.html"

    # Fig 1: fragility score distribution (weighted sense in subtitle)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Histogram(
            x=frag["fragility_score"],
            nbinsx=30,
            marker_color=THEME["accent2"],
            opacity=0.85,
        )
    )
    fig1.update_layout(
        title="Composite fragility score (0–100)",
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(title="Fragility score", gridcolor=THEME["rule"]),
        yaxis=dict(title="Loan count", gridcolor=THEME["rule"]),
    )

    # Fig 2: decile lift — distress rate & lift by model decile
    base_rate = preds["distress_flag_actual"].mean()
    dec = (
        preds.groupby("distress_decile", as_index=False)
        .agg(
            n=("loan_id", "count"),
            distress=("distress_flag_actual", "sum"),
            notional=("principal_mm", "sum"),
        )
    )
    dec["distress_rate"] = dec["distress"] / dec["n"].replace(0, np.nan)
    dec["lift"] = dec["distress_rate"] / base_rate if base_rate > 0 else np.nan

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Bar(
            x=dec["distress_decile"],
            y=dec["distress_rate"] * 100,
            name="Distress rate %",
            marker_color=THEME["accent"],
        ),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(
            x=dec["distress_decile"],
            y=dec["lift"],
            name="Lift vs base",
            mode="lines+markers",
            line=dict(color=THEME["accent2"], width=2),
        ),
        secondary_y=True,
    )
    fig2.update_layout(
        title="ML distress model — decile lift (in-sample)",
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Decile (1=lowest proba)", gridcolor=THEME["rule"]),
    )
    fig2.update_yaxes(title_text="Distress rate %", secondary_y=False, gridcolor=THEME["rule"])
    fig2.update_yaxes(title_text="Lift", secondary_y=True, gridcolor=THEME["rule"])

    # Fig 3: risk tier notionals
    tier = frag.groupby("risk_tier", as_index=False)["principal_mm"].sum()
    order = ["Low", "Medium", "High", "Critical"]
    tier["risk_tier"] = pd.Categorical(tier["risk_tier"], categories=order, ordered=True)
    tier = tier.sort_values("risk_tier")

    fig3 = go.Figure(
        go.Bar(
            x=tier["risk_tier"].astype(str),
            y=tier["principal_mm"],
            marker_color=THEME["accent2"],
        )
    )
    fig3.update_layout(
        title="Notional ($mm) by fragility tier",
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(title="Tier", gridcolor=THEME["rule"]),
        yaxis=dict(title="$mm", gridcolor=THEME["rule"]),
    )

    div1 = plotly_plot(fig1, include_plotlyjs=True, output_type="div", config={"displaylogo": False})
    div2 = plotly_plot(fig2, include_plotlyjs=False, output_type="div", config={"displaylogo": False})
    div3 = plotly_plot(fig3, include_plotlyjs=False, output_type="div", config={"displaylogo": False})

    wa = float(
        (frag["fragility_score"] * frag["principal_mm"]).sum() / max(frag["principal_mm"].sum(), 1e-9)
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Private credit fragility — synthetic book</title>
  <style>
    body {{ font-family: {THEME["font_sans"]}; background: {THEME["paper"]}; color: {THEME["ink"]}; margin: 0; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 24px 48px; }}
    h1 {{ font-family: {THEME["font"]}; font-weight: 400; font-size: clamp(24px, 3vw, 32px); margin: 0 0 8px; }}
    .dek {{ color: {THEME["muted"]}; max-width: 60ch; margin: 0 0 20px; font-size: 15px; }}
    .kpi {{ font-size: 14px; margin-bottom: 24px; color: {THEME["muted"]}; }}
    .panel {{ background: #fff; border: 1px solid {THEME["rule"]}; margin-bottom: 20px; padding: 8px; }}
    .back {{ font-size: 13px; margin-bottom: 16px; }}
    .back a {{ color: {THEME["accent2"]}; }}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="back"><a href="../dashboard.html">← Main credit dashboard</a></p>
    <h1>Synthetic private-credit fragility track</h1>
    <p class="dek">600-loan book, Monte Carlo stress, ICR/PIK analytics, and GBM distress scores. Separate from the Kaggle-style loan tape.</p>
    <p class="kpi">Weighted avg fragility score: <strong>{wa:.1f}</strong> / 100 · Loans: {len(frag):,}</p>
    <div class="panel">{div1}</div>
    <div class="panel">{div2}</div>
    <div class="panel">{div3}</div>
    <p style="font-size:12px;color:{THEME["muted"]};">Generated {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
  </div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Fragility report written: {output_path}")
    return output_path


if __name__ == "__main__":
    build_fragility_report()
