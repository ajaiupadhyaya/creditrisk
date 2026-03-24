"""
Integrated dashboard assembly for credit risk analytics.
Builds a single self-contained HTML report with embedded interactive charts.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")


class DashboardAssembler:
    """Assemble a complete single-file dashboard with embedded visuals."""

    def __init__(self, loan_df: pd.DataFrame, output_path: Path | None = None):
        self.loan_df = loan_df.copy()

        if output_path is not None:
            self.output_path = Path(output_path)
        else:
            cwd = Path.cwd()
            self.output_path = cwd.parent / "outputs" if cwd.name == "notebooks" else cwd / "outputs"

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path / "dashboard.html"

        self.credit_ratings_df = self._safe_read_csv("credit_ratings.csv")
        self.vintage_df = self._safe_read_csv("vintage_analysis.csv")
        self.macro_df = self._safe_read_csv("macro_stress_scenarios.csv")
        self.portfolio_metrics_df = self._safe_read_csv("portfolio_metrics.csv")

        self.models_dir = self.output_path / "models"
        self.stress_dir = self.output_path / "stress"
        self.eda_dir = self.output_path / "eda"

    def _safe_read_csv(self, name: str) -> pd.DataFrame:
        for p in [Path(name), Path.cwd() / name, Path.cwd().parent / name]:
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame()

    def get_portfolio_kpis(self) -> Dict[str, float]:
        ead = self.loan_df["ead"]
        wa_coupon = (self.loan_df["coupon_rate"] * ead).sum() / max(ead.sum(), 1)

        if "debt_to_equity" in self.loan_df.columns:
            wa_ltv = (self.loan_df["debt_to_equity"] * ead).sum() / max(ead.sum(), 1)
        else:
            wa_ltv = np.nan

        return {
            "Total Exposure ($B)": ead.sum() / 1e9,
            "Weighted Avg Interest Rate (%)": wa_coupon,
            "Weighted Avg LTV / D/E": wa_ltv,
            "Default Rate (%)": self.loan_df["defaulted"].mean() * 100,
            "Average PD (%)": self.loan_df["pd_annual"].mean() * 100,
            "Loan Count": float(len(self.loan_df)),
            "Sector Count": float(self.loan_df["sector"].nunique()),
        }

    def _apply_dark_theme(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#151926",
            font=dict(family="Avenir Next, Segoe UI, Helvetica, Arial", size=12, color="#e8edf2"),
            margin=dict(l=50, r=30, t=60, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        return fig

    def _fig_sector_sunburst(self) -> go.Figure:
        grp = self.loan_df.groupby(["sector", "initial_rating"], as_index=False).agg(
            loan_count=("loan_id", "count"),
            exposure=("ead", "sum")
        )
        fig = go.Figure(go.Sunburst(
            labels=np.concatenate([
                grp["sector"].astype(str).to_numpy(),
                grp["initial_rating"].astype(str).to_numpy()
            ]),
            parents=np.concatenate([
                np.repeat("Portfolio", len(grp)).astype(str),
                grp["sector"].astype(str).to_numpy()
            ]),
            values=np.concatenate([
                grp["exposure"].to_numpy(),
                grp["exposure"].to_numpy()
            ]),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Exposure: %{value:$,.0f}<extra></extra>",
            marker=dict(colorscale="Tealgrn")
        ))
        return self._apply_dark_theme(fig, "Portfolio Exposure by Sector and Rating")

    def _fig_default_rate_sector(self) -> go.Figure:
        g = self.loan_df.groupby("sector")["defaulted"].agg(["mean", "count"]).reset_index()
        g["se"] = np.sqrt((g["mean"] * (1 - g["mean"])) / g["count"])
        g["ci95"] = 1.96 * g["se"]
        g = g.sort_values("mean", ascending=False)

        fig = go.Figure(go.Bar(
            x=g["sector"],
            y=g["mean"] * 100,
            error_y=dict(type="data", array=g["ci95"] * 100, visible=True),
            marker=dict(color="#5cc8ff"),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y:.2f}%<extra></extra>"
        ))
        fig.update_yaxes(title="Default Rate (%)")
        return self._apply_dark_theme(fig, "Default Rate by Sector with 95% CI")

    def _fig_rating_sankey(self) -> go.Figure:
        if self.credit_ratings_df.empty:
            return self._apply_dark_theme(go.Figure(), "Rating Transition / Outcome")

        tmp = self.credit_ratings_df.copy()
        tmp["outcome"] = np.where(tmp["defaulted"] == 1, "Default", "Non-Default")
        trans = tmp.groupby(["from_rating", "outcome"], as_index=False).size()

        src_labels = trans["from_rating"].astype(str).unique().tolist()
        tgt_labels = ["Non-Default", "Default"]
        labels = src_labels + tgt_labels
        src_map = {v: i for i, v in enumerate(src_labels)}
        tgt_map = {v: i + len(src_labels) for i, v in enumerate(tgt_labels)}

        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=15, thickness=20),
            link=dict(
                source=[src_map[v] for v in trans["from_rating"].astype(str)],
                target=[tgt_map[v] for v in trans["outcome"]],
                value=trans["size"]
            )
        ))
        return self._apply_dark_theme(fig, "Rating Bucket to Outcome Flow")

    def _load_oof(self) -> pd.DataFrame:
        p = self.models_dir / "oof_predictions.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _fig_roc_pr_calibration(self) -> tuple[go.Figure, go.Figure, go.Figure]:
        oof = self._load_oof()
        roc_fig = go.Figure()
        pr_fig = go.Figure()
        cal_fig = go.Figure()

        if oof.empty:
            return (
                self._apply_dark_theme(roc_fig, "ROC Curves"),
                self._apply_dark_theme(pr_fig, "Precision-Recall Curves"),
                self._apply_dark_theme(cal_fig, "Calibration Curves"),
            )

        colors = {"logistic": "#7bdff2", "xgboost": "#f6bd60", "lightgbm": "#84a59d"}

        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#aaaaaa")))
        cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="#aaaaaa")))

        for model in sorted(oof["model"].unique()):
            d = oof[oof["model"] == model]
            y_true = d["y_true"].values
            y_proba = d["y_pred_proba"].values

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            p, r, _ = precision_recall_curve(y_true, y_proba)
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model.upper()} AUC={roc_auc:.3f}", line=dict(width=3, color=colors.get(model))))
            pr_fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=model.upper(), line=dict(width=3, color=colors.get(model))))
            cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=model.upper(), line=dict(width=3, color=colors.get(model))))

        roc_fig.update_xaxes(title="False Positive Rate")
        roc_fig.update_yaxes(title="True Positive Rate")
        pr_fig.update_xaxes(title="Recall")
        pr_fig.update_yaxes(title="Precision")
        cal_fig.update_xaxes(title="Predicted PD")
        cal_fig.update_yaxes(title="Observed Default Rate")

        return (
            self._apply_dark_theme(roc_fig, "ROC Curves"),
            self._apply_dark_theme(pr_fig, "Precision-Recall Curves"),
            self._apply_dark_theme(cal_fig, "Calibration Curves"),
        )

    def _fig_feature_importance(self) -> go.Figure:
        p = self.models_dir / "feature_importance_top25.csv"
        if not p.exists():
            return self._apply_dark_theme(go.Figure(), "Feature Importance (Top 25)")
        fi = pd.read_csv(p).head(25)
        x = fi["importance_pct"] if "importance_pct" in fi.columns else fi["importance"]
        fig = go.Figure(go.Bar(
            x=x,
            y=fi["feature"],
            orientation="h",
            marker=dict(color="#89b0ae"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>"
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Importance")
        return self._apply_dark_theme(fig, "XGBoost Feature Importance")

    def _fig_risk_return_scatter(self) -> go.Figure:
        g = self.loan_df.groupby("sector", as_index=False).agg(
            default_rate=("defaulted", "mean"),
            avg_coupon=("coupon_rate", "mean"),
            exposure=("ead", "sum")
        )
        fig = go.Figure(go.Scatter(
            x=g["default_rate"] * 100,
            y=g["avg_coupon"],
            mode="markers+text",
            text=g["sector"],
            textposition="top center",
            marker=dict(size=(g["exposure"] / g["exposure"].max()) * 60, color=g["exposure"], colorscale="Viridis", showscale=True),
            hovertemplate="<b>%{text}</b><br>Default Rate: %{x:.2f}%<br>Avg Coupon: %{y:.2f}%<extra></extra>"
        ))
        fig.update_xaxes(title="Sector Default Rate (%)")
        fig.update_yaxes(title="Average Coupon Rate (%)")
        return self._apply_dark_theme(fig, "Risk-Return Sector Map")

    def _load_sector_analysis(self) -> pd.DataFrame:
        p = self.stress_dir / "sector_analysis.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _fig_stress_sector_bar(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty:
            return self._apply_dark_theme(go.Figure(), "Stress EL by Sector")

        fig = go.Figure()
        for scenario, df_s in sa.groupby("scenario"):
            fig.add_trace(go.Bar(
                x=df_s["sector"],
                y=df_s["total_el"] / 1e6,
                name=str(scenario),
                hovertemplate="<b>%{x}</b><br>EL: %{y:.2f}M<extra></extra>"
            ))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Expected Loss ($M)")
        return self._apply_dark_theme(fig, "Expected Loss by Sector across Scenarios")

    def _fig_stress_heatmap(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty:
            return self._apply_dark_theme(go.Figure(), "Stress EL Matrix")

        pivot = sa.pivot(index="sector", columns="scenario", values="el_pct_ead")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Reds",
            text=np.round(pivot.values, 2),
            texttemplate="%{text:.2f}",
            colorbar=dict(title="EL % EAD")
        ))
        return self._apply_dark_theme(fig, "Sector x Scenario EL Matrix")

    def _fig_vintage_overlay(self) -> go.Figure:
        if self.vintage_df.empty:
            return self._apply_dark_theme(go.Figure(), "Vintage Base vs Stress")

        fig = go.Figure()
        base_mult = 1.0
        stress_mult = 1.5
        if not self.macro_df.empty and {"scenario", "pd_multiplier"}.issubset(self.macro_df.columns):
            m = self.macro_df.copy()
            m["scenario_lower"] = m["scenario"].astype(str).str.lower()
            base = m[m["scenario_lower"].isin(["base", "baseline"])]["pd_multiplier"].mean()
            adverse = m[m["scenario_lower"].isin(["adverse", "severe", "severely_adverse"])]["pd_multiplier"].mean()
            if not np.isnan(base) and base != 0 and not np.isnan(adverse):
                base_mult = base
                stress_mult = adverse
        ratio = stress_mult / base_mult if base_mult else 1.5

        for vintage, g in self.vintage_df.groupby("vintage"):
            g = g.sort_values("months_on_books")
            base_curve = g["cumulative_default_rate"] * 100
            stressed_curve = np.clip(base_curve * ratio, 0, 100)
            fig.add_trace(go.Scatter(x=g["months_on_books"], y=base_curve, mode="lines", name=f"{vintage} Base", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=g["months_on_books"], y=stressed_curve, mode="lines", name=f"{vintage} Stress", line=dict(width=2, dash="dash")))

        fig.update_xaxes(title="Months on Books")
        fig.update_yaxes(title="Cumulative Default Rate (%)")
        return self._apply_dark_theme(fig, "Vintage Curves: Base vs Adverse Overlay")

    def _fig_macro_trends(self) -> go.Figure:
        if self.portfolio_metrics_df.empty or "date" not in self.portfolio_metrics_df.columns:
            return self._apply_dark_theme(go.Figure(), "Portfolio Macro and Risk Trends")

        df = self.portfolio_metrics_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Avg PD", "EL Rate", "Unemployment", "Credit Spread"))

        mapping = [
            ("avg_pd", 1, 1),
            ("el_rate", 1, 2),
            ("unemployment", 2, 1),
            ("credit_spread_bps", 2, 2),
        ]
        for col, r, c in mapping:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col), row=r, col=c)

        fig.update_layout(height=700, showlegend=False)
        return self._apply_dark_theme(fig, "Portfolio Macro and Risk Trends")

    def _fig_to_div(self, fig: go.Figure, include_js: bool) -> str:
        return plotly_plot(
            fig,
            include_plotlyjs=True if include_js else False,
            output_type="div",
            config={"displaylogo": False, "responsive": True}
        )

    def _kpi_cards_html(self) -> str:
        k = self.get_portfolio_kpis()
        order = [
            "Total Exposure ($B)",
            "Weighted Avg Interest Rate (%)",
            "Weighted Avg LTV / D/E",
            "Default Rate (%)",
            "Average PD (%)",
            "Loan Count",
            "Sector Count",
        ]
        cards = []
        for key in order:
            value = k.get(key, np.nan)
            if "($B)" in key:
                txt = f"${value:,.2f}B"
            elif "%" in key:
                txt = f"{value:,.2f}%"
            else:
                txt = f"{int(value):,}" if not np.isnan(value) else "N/A"
            cards.append(f'<div class="kpi"><div class="kpi-label">{key}</div><div class="kpi-value">{txt}</div></div>')
        return "".join(cards)

    def assemble_dashboard(self, eda_path: Path | None = None, models_path: Path | None = None, stress_path: Path | None = None) -> Path:
        """Build one integrated single-file dashboard."""
        if models_path is not None:
            self.models_dir = Path(models_path)
        if stress_path is not None:
            self.stress_dir = Path(stress_path)
        if eda_path is not None:
            self.eda_dir = Path(eda_path)

        figures: List[tuple[str, go.Figure]] = []
        figures.append(("Portfolio Exposure", self._fig_sector_sunburst()))
        figures.append(("Default Rate by Sector", self._fig_default_rate_sector()))
        figures.append(("Rating Flow", self._fig_rating_sankey()))

        roc, pr, cal = self._fig_roc_pr_calibration()
        figures.append(("ROC", roc))
        figures.append(("Precision-Recall", pr))
        figures.append(("Calibration", cal))

        figures.append(("Feature Importance", self._fig_feature_importance()))
        figures.append(("Risk-Return", self._fig_risk_return_scatter()))
        figures.append(("Stress EL by Sector", self._fig_stress_sector_bar()))
        figures.append(("Stress EL Matrix", self._fig_stress_heatmap()))
        figures.append(("Vintage Stress Overlay", self._fig_vintage_overlay()))
        figures.append(("Macro Risk Trends", self._fig_macro_trends()))

        fig_divs = []
        include_js = True
        for title, fig in figures:
            fig_divs.append(f'<section class="panel"><h3>{title}</h3>{self._fig_to_div(fig, include_js)}</section>')
            include_js = False

        extra_images = []
        for p in [
            self.models_dir / "confusion_matrix_logistic.png",
            self.models_dir / "confusion_matrix_xgboost.png",
            self.models_dir / "confusion_matrix_lightgbm.png",
            self.models_dir / "score_distribution_logistic.png",
            self.models_dir / "score_distribution_xgboost.png",
            self.models_dir / "score_distribution_lightgbm.png",
            self.models_dir / "shap_summary_xgboost.png",
            self.eda_dir / "correlation_heatmap.png",
            self.eda_dir / "vintage_heatmap.png",
            self.stress_dir / "el_heatmap_sector_scenario.png",
        ]:
            if p.exists():
                rel = p.relative_to(self.output_path)
                extra_images.append(f'<div class="img-card"><h4>{p.name}</h4><img src="{rel.as_posix()}" alt="{p.name}" /></div>')

        html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Credit Risk Intelligence Dashboard</title>
  <style>
    :root {{
      --bg:#0b1020;
      --bg2:#111a33;
      --panel:#121a2b;
      --accent:#4cc9f0;
      --accent2:#f4a261;
      --text:#e9eef7;
      --muted:#9aa7bd;
      --border:#24314f;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:'Avenir Next','Segoe UI',Helvetica,Arial,sans-serif; color:var(--text); background: radial-gradient(1200px 700px at 10% 0%, #1b2a52 0%, var(--bg) 55%), linear-gradient(120deg, var(--bg), var(--bg2)); }}
    .wrap {{ max-width:1600px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 8px; font-size:34px; letter-spacing:0.5px; }}
    .sub {{ color:var(--muted); margin-bottom:20px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:14px; margin-bottom:22px; }}
    .kpi {{ background:linear-gradient(180deg,#18233d,#121a2b); border:1px solid var(--border); border-radius:14px; padding:14px; }}
    .kpi-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.7px; }}
    .kpi-value {{ margin-top:8px; font-size:27px; font-weight:700; color:#dff6ff; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .panel {{ background:rgba(18,26,43,0.92); border:1px solid var(--border); border-radius:14px; padding:12px; overflow:hidden; }}
    .panel h3 {{ margin:6px 8px 10px; font-size:14px; color:var(--accent); text-transform:uppercase; letter-spacing:0.7px; }}
    .images {{ margin-top:20px; display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
    .img-card {{ background:rgba(18,26,43,0.92); border:1px solid var(--border); border-radius:14px; padding:10px; }}
    .img-card h4 {{ margin:4px 4px 8px; color:var(--accent2); font-size:12px; }}
    .img-card img {{ width:100%; border-radius:10px; border:1px solid #2d3f61; }}
    .footer {{ margin-top:18px; color:var(--muted); font-size:12px; text-align:center; }}
    @media (max-width:1200px) {{
      .kpis {{ grid-template-columns:repeat(2,minmax(180px,1fr)); }}
      .grid {{ grid-template-columns:1fr; }}
      .images {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Credit Risk Intelligence Dashboard</h1>
    <div class="sub">Integrated EDA, PD Modeling, Stress Testing, and Vintage Analytics • Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <section class="kpis">{self._kpi_cards_html()}</section>
    <section class="grid">{''.join(fig_divs)}</section>
    <section class="images">{''.join(extra_images)}</section>
    <div class="footer">Self-contained dashboard • Professional dark terminal theme • All modules consolidated in one view</div>
  </div>
</body>
</html>
"""
        self.output_file.write_text(html, encoding="utf-8")
        print(f"✅ Dashboard written: {self.output_file}")
        return self.output_file


def assemble_dashboard(loan_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    assembler = DashboardAssembler(loan_df, output_path)
    return assembler.assemble_dashboard()
