"""
Integrated dashboard assembly for private-credit risk analytics.
Single self-contained HTML report with embedded interactive charts — editorial layout,
thesis-driven metrics (liquidity, macro, stress, vintage fragility).
"""

from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.offline import plot as plotly_plot
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)

# Editorial palette: NYT × MoMA — warm paper, ink, single accent
THEME = {
    "paper": "#f7f5f0",
    "plot": "#f0ebe3",
    "ink": "#1c1917",
    "muted": "#57534e",
    "rule": "#d6d3cd",
    "accent": "#b45309",
    "accent2": "#1e3a5f",
    "positive": "#0f766e",
    "negative": "#9f1239",
    "font": "Iowan Old Style, Georgia, Palatino Linotype, Book Antiqua, serif",
    "font_sans": "system-ui, -apple-system, Segoe UI, Helvetica Neue, Arial, sans-serif",
}


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
        wa_coupon = float((self.loan_df["coupon_rate"] * ead).sum() / max(ead.sum(), 1))

        if "debt_to_equity" in self.loan_df.columns:
            wa_ltv = float((self.loan_df["debt_to_equity"] * ead).sum() / max(ead.sum(), 1))
        else:
            wa_ltv = np.nan

        wam = float((self.loan_df["maturity_months"].astype(float) * ead).sum() / max(ead.sum(), 1))
        unsecured_share = np.nan
        if "collateral" in self.loan_df.columns:
            u = self.loan_df["collateral"].astype(str).str.lower().eq("unsecured")
            unsecured_share = float(ead[u].sum() / max(ead.sum(), 1) * 100)

        long_dated = self.loan_df["maturity_months"].astype(float) >= 60
        illiquid_long_share = float((ead * long_dated).sum() / max(ead.sum(), 1) * 100)

        stress_el_ratio = np.nan
        sa = self._load_sector_analysis()
        if not sa.empty and {"scenario", "total_el"}.issubset(sa.columns):
            by_s = sa.groupby("scenario", as_index=False)["total_el"].sum()
            base = by_s[by_s["scenario"].astype(str).str.lower().isin(["baseline", "base"])]
            adv = by_s[by_s["scenario"].astype(str).str.lower() == "adverse"]
            if not base.empty and not adv.empty and base["total_el"].iloc[0] > 0:
                stress_el_ratio = float(adv["total_el"].iloc[0] / base["total_el"].iloc[0])

        return {
            "Total Exposure ($B)": ead.sum() / 1e9,
            "Weighted Avg Coupon (%)": wa_coupon,
            "Weighted Avg LTV / D/E": wa_ltv,
            "Default Rate (%)": self.loan_df["defaulted"].mean() * 100,
            "Average PD (%)": self.loan_df["pd_annual"].mean() * 100,
            "WAM (months)": wam,
            "Unsecured Exposure (%)": unsecured_share,
            "EAD in ≥60mo Tenors (%)": illiquid_long_share,
            "Adverse / Baseline EL": stress_el_ratio,
            "Loan Count": float(len(self.loan_df)),
            "Sector Count": float(self.loan_df["sector"].nunique()),
        }

    def _apply_theme(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color=THEME["ink"], family=THEME["font_sans"])),
            template="plotly_white",
            paper_bgcolor=THEME["paper"],
            plot_bgcolor=THEME["plot"],
            font=dict(family=THEME["font_sans"], size=11, color=THEME["ink"]),
            margin=dict(l=56, r=28, t=56, b=48),
            legend=dict(
                bgcolor="rgba(247,245,240,0.85)",
                bordercolor=THEME["rule"],
                borderwidth=1,
                font=dict(size=10),
            ),
            xaxis=dict(gridcolor=THEME["rule"], linecolor=THEME["rule"], zerolinecolor=THEME["rule"]),
            yaxis=dict(gridcolor=THEME["rule"], linecolor=THEME["rule"], zerolinecolor=THEME["rule"]),
        )
        return fig

    def _fig_sector_sunburst(self) -> go.Figure:
        df = self.loan_df
        sec_exp = df.groupby("sector", as_index=False)["ead"].sum()
        sr = df.groupby(["sector", "initial_rating"], as_index=False)["ead"].sum()

        labels = ["Portfolio"] + sec_exp["sector"].astype(str).tolist()
        parents = [""] + ["Portfolio"] * len(sec_exp)
        values = [float(sec_exp["ead"].sum())] + sec_exp["ead"].astype(float).tolist()

        leaf_labels = [f"{row.sector} — {row.initial_rating}" for _, row in sr.iterrows()]
        labels.extend(leaf_labels)
        parents.extend(sr["sector"].astype(str).tolist())
        values.extend(sr["ead"].astype(float).tolist())

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>Exposure: %{value:$,.0f}<extra></extra>",
                marker=dict(line=dict(color=THEME["paper"], width=1.5)),
                maxdepth=3,
            )
        )
        return self._apply_theme(fig, "Exposure hierarchy: sector → rating")

    def _fig_default_rate_sector(self) -> go.Figure:
        g = self.loan_df.groupby("sector")["defaulted"].agg(["mean", "count"]).reset_index()
        g["se"] = np.sqrt((g["mean"] * (1 - g["mean"])) / g["count"])
        g["ci95"] = 1.96 * g["se"]
        g = g.sort_values("mean", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=g["mean"] * 100,
                y=g["sector"],
                orientation="h",
                error_x=dict(type="data", array=g["ci95"] * 100, visible=True, color=THEME["muted"]),
                marker=dict(color=THEME["accent2"]),
                hovertemplate="<b>%{y}</b><br>DR: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_xaxes(title="Default rate (%)")
        fig.update_yaxes(title="")
        return self._apply_theme(fig, "Default rate by sector (95% CI)")

    def _fig_rating_sankey(self) -> go.Figure:
        if self.credit_ratings_df.empty:
            return self._apply_theme(go.Figure(), "Rating → outcome")

        tmp = self.credit_ratings_df.copy()
        tmp["outcome"] = np.where(tmp["defaulted"] == 1, "Default", "Performing")
        trans = tmp.groupby(["from_rating", "outcome"], as_index=False).size()

        src_labels = trans["from_rating"].astype(str).unique().tolist()
        tgt_labels = ["Performing", "Default"]
        labels = src_labels + tgt_labels
        src_map = {v: i for i, v in enumerate(src_labels)}
        tgt_map = {v: i + len(src_labels) for i, v in enumerate(tgt_labels)}

        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=18,
                    thickness=14,
                    color=[THEME["plot"]] * len(labels),
                    line=dict(color=THEME["rule"], width=0.5),
                ),
                link=dict(
                    source=[src_map[v] for v in trans["from_rating"].astype(str)],
                    target=[tgt_map[v] for v in trans["outcome"]],
                    value=trans["size"],
                    color="rgba(30,58,95,0.35)",
                ),
            )
        )
        return self._apply_theme(fig, "Rating bucket to realized outcome")

    def _load_oof(self) -> pd.DataFrame:
        p = self.models_dir / "oof_predictions.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _fig_roc_pr_calibration(self) -> Tuple[go.Figure, go.Figure, go.Figure]:
        oof = self._load_oof()
        roc_fig = go.Figure()
        pr_fig = go.Figure()
        cal_fig = go.Figure()

        if oof.empty:
            return (
                self._apply_theme(roc_fig, "ROC"),
                self._apply_theme(pr_fig, "Precision–recall"),
                self._apply_theme(cal_fig, "Calibration"),
            )

        colors = {"logistic": THEME["accent2"], "xgboost": THEME["accent"], "lightgbm": THEME["positive"]}

        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color=THEME["muted"]),
            )
        )
        cal_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect",
                line=dict(dash="dash", color=THEME["muted"]),
            )
        )

        for model in sorted(oof["model"].unique()):
            d = oof[oof["model"] == model]
            y_true = d["y_true"].values
            y_proba = d["y_pred_proba"].values

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            p, r, _ = precision_recall_curve(y_true, y_proba)
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

            c = colors.get(model, THEME["accent2"])
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model.upper()} · AUC {roc_auc:.3f}",
                    line=dict(width=2.2, color=c),
                )
            )
            pr_fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=model.upper(), line=dict(width=2.2, color=c)))
            cal_fig.add_trace(
                go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    mode="lines+markers",
                    name=model.upper(),
                    line=dict(width=2, color=c),
                    marker=dict(size=7, color=c),
                )
            )

        roc_fig.update_xaxes(title="False positive rate")
        roc_fig.update_yaxes(title="True positive rate")
        pr_fig.update_xaxes(title="Recall")
        pr_fig.update_yaxes(title="Precision")
        cal_fig.update_xaxes(title="Mean predicted PD")
        cal_fig.update_yaxes(title="Observed default rate")

        return (
            self._apply_theme(roc_fig, "ROC — out-of-fold"),
            self._apply_theme(pr_fig, "Precision–recall"),
            self._apply_theme(cal_fig, "Calibration (reliability)"),
        )

    def _fig_feature_importance(self) -> go.Figure:
        p = self.models_dir / "feature_importance_top25.csv"
        if not p.exists():
            return self._apply_theme(go.Figure(), "Feature importance")
        fi = pd.read_csv(p).head(25)
        x = fi["importance_pct"] if "importance_pct" in fi.columns else fi["importance"]
        fig = go.Figure(
            go.Bar(
                x=x,
                y=fi["feature"],
                orientation="h",
                marker=dict(color=THEME["accent2"]),
                hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Importance")
        return self._apply_theme(fig, "XGBoost — top drivers")

    def _fig_risk_return_scatter(self) -> go.Figure:
        g = self.loan_df.groupby("sector", as_index=False).agg(
            default_rate=("defaulted", "mean"),
            avg_coupon=("coupon_rate", "mean"),
            exposure=("ead", "sum"),
        )
        fig = go.Figure(
            go.Scatter(
                x=g["default_rate"] * 100,
                y=g["avg_coupon"],
                mode="markers+text",
                text=g["sector"],
                textposition="top center",
                textfont=dict(size=9, color=THEME["muted"]),
                marker=dict(
                    size=np.clip((g["exposure"] / g["exposure"].max()) * 56, 10, 56),
                    color=g["exposure"],
                    colorscale=[[0, "#e7e5e0"], [0.5, "#9a3412"], [1, "#1e3a5f"]],
                    showscale=True,
                    colorbar=dict(title="EAD", tickformat=",.0s"),
                    line=dict(width=0.5, color=THEME["ink"]),
                ),
                hovertemplate="<b>%{text}</b><br>DR: %{x:.2f}%<br>Coupon: %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_xaxes(title="Sector default rate (%)")
        fig.update_yaxes(title="Average coupon (%)")
        return self._apply_theme(fig, "Risk–return by sector (size ∝ exposure)")

    def _load_sector_analysis(self) -> pd.DataFrame:
        p = self.stress_dir / "sector_analysis.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _fig_stress_totals_by_scenario(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty:
            return self._apply_theme(go.Figure(), "Stress EL by scenario")

        tot = sa.groupby("scenario", as_index=False)["total_el"].sum()
        order = ["baseline", "mild", "adverse", "severe", "gfc_like", "covid_like"]
        tot["_o"] = tot["scenario"].astype(str).str.lower().map({v: i for i, v in enumerate(order)})
        tot = tot.sort_values("_o", na_position="last")

        fig = go.Figure(
            go.Bar(
                x=tot["scenario"].astype(str),
                y=tot["total_el"] / 1e9,
                marker=dict(
                    color=[THEME["accent2"] if str(s).lower() == "baseline" else THEME["accent"] for s in tot["scenario"]]
                ),
                hovertemplate="<b>%{x}</b><br>EL: $%{y:.2f}B<extra></extra>",
            )
        )
        fig.update_yaxes(title="Portfolio expected loss ($B)")
        fig.update_xaxes(title="")
        return self._apply_theme(fig, "Portfolio EL under macro scenarios")

    def _fig_stress_sector_bar(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty:
            return self._apply_theme(go.Figure(), "Stress EL by sector")

        main = sa[sa["scenario"].astype(str).str.lower().isin(["baseline", "adverse", "severe"])].copy()
        if main.empty:
            main = sa

        fig = go.Figure()
        palette = {"baseline": THEME["accent2"], "adverse": THEME["accent"], "severe": THEME["negative"]}
        for scenario in main["scenario"].unique():
            df_s = main[main["scenario"] == scenario]
            fig.add_trace(
                go.Bar(
                    x=df_s["sector"],
                    y=df_s["total_el"] / 1e6,
                    name=str(scenario),
                    marker=dict(color=palette.get(str(scenario).lower(), THEME["muted"])),
                    hovertemplate="<b>%{x}</b><br>EL: %{y:.1f}M<extra></extra>",
                )
            )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Expected loss ($M)")
        return self._apply_theme(fig, "EL by sector: baseline vs stress")

    def _fig_stress_heatmap(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty:
            return self._apply_theme(go.Figure(), "EL matrix")

        pivot = sa.pivot(index="sector", columns="scenario", values="el_pct_ead")
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=pivot.index,
                colorscale=[[0, "#f7f5f0"], [0.35, "#d97706"], [1, "#7f1d1d"]],
                text=np.round(pivot.values, 2),
                texttemplate="%{text:.2f}",
                textfont=dict(size=9, color=THEME["ink"]),
                colorbar=dict(title="EL % EAD"),
            )
        )
        return self._apply_theme(fig, "EL as % of EAD — sector × scenario")

    def _fig_liquidity_structure(self) -> go.Figure:
        df = self.loan_df
        if "maturity_months" not in df.columns or "loan_type" not in df.columns:
            return self._apply_theme(go.Figure(), "Liquidity & tenor")

        rows = []
        for lt, g in df.groupby("loan_type"):
            w = g["ead"].astype(float)
            m = g["maturity_months"].astype(float)
            wam = float(np.average(m, weights=w)) if w.sum() > 0 else float(m.mean())
            rows.append({"loan_type": lt, "wam": wam, "ead": float(g["ead"].sum())})
        g = pd.DataFrame(rows).sort_values("ead", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=g["wam"],
                y=g["loan_type"],
                orientation="h",
                marker=dict(color=THEME["accent"]),
                text=[f"{v:.0f} mo" for v in g["wam"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>WAM: %{x:.1f} mo<br>Structural liquidity risk in long private credit.<extra></extra>",
            )
        )
        fig.update_xaxes(title="Weighted avg maturity (months)")
        fig.update_yaxes(title="")
        return self._apply_theme(fig, "Structural liquidity: WAM by instrument type")

    def _fig_maturity_distribution(self) -> go.Figure:
        df = self.loan_df
        if "maturity_months" not in df.columns:
            return self._apply_theme(go.Figure(), "Tenor mix")

        fig = go.Figure()
        for coll in sorted(df["collateral"].astype(str).unique())[:8]:
            sub = df[df["collateral"].astype(str) == coll]
            fig.add_trace(
                go.Box(
                    y=sub["maturity_months"],
                    name=coll[:18],
                    boxmean=True,
                    marker=dict(color=THEME["accent2"]),
                    line=dict(color=THEME["ink"]),
                )
            )
        fig.update_yaxes(title="Maturity (months)")
        fig.update_xaxes(title="")
        return self._apply_theme(fig, "Tenor dispersion by collateral — liquidity mismatch lens")

    def _fig_portfolio_tape(self) -> go.Figure:
        if self.portfolio_metrics_df.empty or "date" not in self.portfolio_metrics_df.columns:
            return self._apply_theme(go.Figure(), "Macro & risk tape")

        pm = self.portfolio_metrics_df.copy()
        pm["date"] = pd.to_datetime(pm["date"])

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Average PD",
                "EL rate",
                "Unemployment",
                "Credit spread (bps)",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )

        if "avg_pd" in pm.columns:
            fig.add_trace(
                go.Scatter(x=pm["date"], y=pm["avg_pd"], mode="lines", line=dict(color=THEME["accent2"], width=2), showlegend=False),
                row=1,
                col=1,
            )
        if "el_rate" in pm.columns:
            fig.add_trace(
                go.Scatter(x=pm["date"], y=pm["el_rate"], mode="lines", line=dict(color=THEME["negative"], width=2), showlegend=False),
                row=1,
                col=2,
            )
        if "unemployment" in pm.columns:
            fig.add_trace(
                go.Scatter(x=pm["date"], y=pm["unemployment"], mode="lines", line=dict(color=THEME["ink"], width=1.8), showlegend=False),
                row=2,
                col=1,
            )
        if "credit_spread_bps" in pm.columns:
            fig.add_trace(
                go.Scatter(x=pm["date"], y=pm["credit_spread_bps"], mode="lines", line=dict(color=THEME["accent"], width=1.6), showlegend=False),
                row=2,
                col=2,
            )

        fig.update_layout(height=720, showlegend=False)
        fig = self._apply_theme(fig, "Macro & embedded risk — portfolio tape (synthetic)")
        return fig

    def _fig_new_defaults_flow(self) -> go.Figure:
        if self.portfolio_metrics_df.empty or "date" not in self.portfolio_metrics_df.columns:
            return self._apply_theme(go.Figure(), "Default flow")

        pm = self.portfolio_metrics_df.copy()
        pm["date"] = pd.to_datetime(pm["date"])
        if "new_defaults" not in pm.columns:
            return self._apply_theme(go.Figure(), "Default flow")

        fig = go.Figure(
            go.Scatter(
                x=pm["date"],
                y=pm["new_defaults"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(159,18,57,0.15)",
                line=dict(color=THEME["negative"], width=2),
                hovertemplate="%{x|%Y-%m}<br>New defaults: %{y}<extra></extra>",
            )
        )
        fig.update_yaxes(title="New defaults (count)")
        fig.update_xaxes(title="")
        return self._apply_theme(fig, "Incidence of new defaults — rising tail risk")

    def _fig_vintage_overlay(self) -> go.Figure:
        if self.vintage_df.empty:
            return self._apply_theme(go.Figure(), "Vintage stress")

        vdf = self.vintage_df.copy()
        vintages = sorted(vdf["vintage"].unique())[-8:]
        vdf = vdf[vdf["vintage"].isin(vintages)]

        base_mult = 1.0
        stress_mult = 1.5
        if not self.macro_df.empty and {"scenario", "pd_multiplier"}.issubset(self.macro_df.columns):
            m = self.macro_df.copy()
            m["scenario_lower"] = m["scenario"].astype(str).str.lower()
            base = m[m["scenario_lower"].isin(["baseline", "base"])]["pd_multiplier"].mean()
            adverse = m[m["scenario_lower"] == "adverse"]["pd_multiplier"].mean()
            if not np.isnan(base) and base != 0 and not np.isnan(adverse):
                base_mult = float(base)
                stress_mult = float(adverse)
        ratio = stress_mult / base_mult if base_mult else 1.5

        fig = go.Figure()
        colors = [
            "#1e3a5f",
            "#9a3412",
            "#0f766e",
            "#57534e",
            "#7c2d12",
            "#155e75",
            "#92400e",
            "#44403c",
        ]
        for i, vintage in enumerate(vintages):
            g = vdf[vdf["vintage"] == vintage].sort_values("months_on_books")
            base_curve = g["cumulative_default_rate"] * 100
            stressed_curve = np.clip(base_curve * ratio, 0, 100)
            c = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=g["months_on_books"],
                    y=base_curve,
                    mode="lines",
                    name=f"{vintage} · base",
                    line=dict(width=2, color=c),
                    legendgroup=vintage,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=g["months_on_books"],
                    y=stressed_curve,
                    mode="lines",
                    name=f"{vintage} · adverse×",
                    line=dict(width=1.6, dash="dash", color=c),
                    legendgroup=vintage,
                    showlegend=True,
                )
            )

        fig.update_xaxes(title="Months on books")
        fig.update_yaxes(title="Cumulative default rate (%)")
        return self._apply_theme(fig, "Vintage curves — base vs adverse PD overlay (recent cohorts)")

    def _fig_treemap_ead(self) -> go.Figure:
        df = self.loan_df.copy()
        df["initial_rating"] = df["initial_rating"].astype(str)
        df["sector"] = df["sector"].astype(str)
        fig = px.treemap(
            df,
            path=[px.Constant("Portfolio"), "sector", "initial_rating"],
            values="ead",
            color="pd_annual",
            color_continuous_scale=[[0, "#f7f5f0"], [0.5, "#d97706"], [1, "#7f1d1d"]],
            hover_data={"ead": ":,.0f"},
        )
        fig.update_traces(textinfo="label+value+percent parent", marker=dict(line=dict(color=THEME["paper"], width=1)))
        return self._apply_theme(fig, "Treemap — EAD nested by sector & rating (color = PD)")

    def _fig_pd_violin(self) -> go.Figure:
        df = self.loan_df.copy()
        df["defaulted"] = df["defaulted"].map({0: "Performing", 1: "Defaulted"})
        fig = px.violin(
            df,
            x="sector",
            y="pd_annual",
            color="defaulted",
            box=True,
            points=False,
            color_discrete_map={"Performing": THEME["accent2"], "Defaulted": THEME["negative"]},
        )
        fig.update_layout(violinmode="group")
        fig.update_yaxes(title="Annual PD", tickformat=".1%")
        fig.update_xaxes(title="")
        return self._apply_theme(fig, "PD distribution by sector — violin + box (outcome split)")

    def _fig_origination_animation(self) -> go.Figure:
        df = self.loan_df.copy()
        if "origination_date" not in df.columns:
            return self._apply_theme(go.Figure(), "Vintage formation")
        df["year"] = pd.to_datetime(df["origination_date"], errors="coerce").dt.year
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        years = sorted(df["year"].unique())
        if len(years) < 2:
            return self._apply_theme(go.Figure(), "Vintage formation")
        cap = float(df["ead"].quantile(0.995))
        fig = px.histogram(
            df,
            x="ead",
            animation_frame="year",
            nbins=50,
            range_x=[0, cap],
            color_discrete_sequence=[THEME["accent2"]],
            title="",
        )
        fig.update_xaxes(title="EAD ($)", tickformat=",.0s")
        fig.update_yaxes(title="Loan count")
        try:
            if getattr(fig.layout, "updatemenus", None) and len(fig.layout.updatemenus) > 0:
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 650
        except Exception:
            pass
        return self._apply_theme(fig, "Exposure formation — EAD histogram by origination year (play)")

    def _fig_parallel_coords(self) -> go.Figure:
        df = self.loan_df.copy()
        cols = ["ead", "coupon_rate", "pd_annual", "maturity_months", "leverage"]
        if not all(c in df.columns for c in cols):
            return self._apply_theme(go.Figure(), "Risk dimensions")
        df = df[cols].dropna()
        df = df.sample(min(4000, len(df)), random_state=42)
        pd_color = df["pd_annual"].values
        for c in cols:
            lo, hi = df[c].min(), df[c].max()
            df[c] = (df[c] - lo) / max(hi - lo, 1e-9)
        df["pd_annual_color"] = pd_color
        fig = px.parallel_coordinates(
            df,
            dimensions=cols,
            color="pd_annual_color",
            color_continuous_scale=[[0, THEME["positive"]], [0.5, THEME["accent"]], [1, THEME["negative"]]],
        )
        return self._apply_theme(fig, "Parallel coordinates — normalized risk geometry (color = PD)")

    def _fig_density_pd_coupon(self) -> go.Figure:
        df = self.loan_df.copy()
        try:
            fig = px.density_heatmap(
                df,
                x="coupon_rate",
                y="pd_annual",
                color_continuous_scale=[[0, "#fff"], [0.35, THEME["accent2"]], [1, THEME["negative"]]],
                nbinsx=32,
                nbinsy=32,
                marginal_x="histogram",
                marginal_y="histogram",
            )
        except Exception:
            fig = px.density_contour(df, x="coupon_rate", y="pd_annual", color_continuous_scale=[[0, THEME["accent2"]], [1, THEME["negative"]]])
        fig.update_xaxes(title="Coupon (%)")
        fig.update_yaxes(title="Annual PD", tickformat=".1%")
        return self._apply_theme(fig, "Joint density — coupon × PD (marginal histograms)")

    def _fig_stress_funnel(self) -> go.Figure:
        sa = self._load_sector_analysis()
        if sa.empty or "scenario" not in sa.columns:
            return self._apply_theme(go.Figure(), "Stress funnel")
        tot = sa.groupby("scenario", as_index=False)["total_el"].sum()
        order = ["baseline", "mild", "adverse", "severe", "gfc_like", "covid_like"]
        tot["_o"] = tot["scenario"].astype(str).str.lower().map({v: i for i, v in enumerate(order)})
        tot = tot.sort_values("_o", na_position="last")
        tot = tot.sort_values("total_el", ascending=False)
        colors = [THEME["accent2"] if "base" in str(s).lower() else THEME["accent"] for s in tot["scenario"]]
        fig = go.Figure(
            go.Funnel(
                y=tot["scenario"].astype(str),
                x=tot["total_el"] / 1e9,
                textinfo="value+percent initial",
                marker=dict(color=colors),
                hovertemplate="<b>%{y}</b><br>EL: $%{x:.2f}B<extra></extra>",
            )
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        return self._apply_theme(fig, "Stress funnel — EL ($B) by scenario (ordered ladder)")

    def _fig_scatter_3d_risk(self) -> go.Figure:
        df = self.loan_df.copy()
        if not {"pd_annual", "coupon_rate", "ead"}.issubset(df.columns):
            return self._apply_theme(go.Figure(), "3D risk cloud")
        sub = df.sample(min(6000, len(df)), random_state=42).copy()
        sub["log_ead"] = np.log10(sub["ead"].clip(lower=1))
        fig = px.scatter_3d(
            sub,
            x="pd_annual",
            y="coupon_rate",
            z="log_ead",
            color="sector",
            size="log_ead",
            size_max=12,
            opacity=0.32,
            hover_data=["ead", "initial_rating"],
        )
        fig.update_layout(scene=dict(zaxis=dict(title="log10 EAD")))
        fig.update_layout(scene=dict(xaxis=dict(tickformat=".0%"), yaxis=dict(title="Coupon %")))
        return self._apply_theme(fig, "3D risk cloud — PD × coupon × EAD (color = sector)")

    def _build_seaborn_embeds(self) -> str:
        df = self.loan_df.copy()
        out: List[str] = []
        try:
            fig1, ax1 = plt.subplots(figsize=(7.2, 4.2))
            df_k = df.copy()
            df_k["defaulted"] = df_k["defaulted"].map({0: "Performing", 1: "Defaulted"})
            sns.kdeplot(data=df_k, x="pd_annual", hue="defaulted", fill=True, common_norm=False, alpha=0.45, ax=ax1, palette=[THEME["accent2"], THEME["negative"]])
            ax1.set_xlabel("Annual PD")
            ax1.set_title("PD density — defaulted vs performing (Seaborn)")
            buf = io.BytesIO()
            fig1.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=THEME["paper"])
            plt.close(fig1)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append(f'<div class="img-card"><h4>Seaborn · PD KDE</h4><img src="data:image/png;base64,{b64}" alt="PD KDE" /></div>')

            num_cols = [c for c in ["ead", "coupon_rate", "pd_annual", "maturity_months", "leverage", "interest_coverage"] if c in df.columns]
            if len(num_cols) >= 4:
                cm = df[num_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(5.8, 5))
                sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax2, vmin=-1, vmax=1)
                ax2.set_title("Risk feature correlation (Seaborn)")
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", dpi=130, bbox_inches="tight", facecolor=THEME["paper"])
                plt.close(fig2)
                b64 = base64.b64encode(buf2.getvalue()).decode("ascii")
                out.append(f'<div class="img-card"><h4>Seaborn · Correlation</h4><img src="data:image/png;base64,{b64}" alt="Correlation" /></div>')

            if "credit_score" in df.columns:
                fig3, ax3 = plt.subplots(figsize=(7, 3.8))
                sns.scatterplot(
                    data=df.sample(min(8000, len(df)), random_state=42),
                    x="credit_score",
                    y="pd_annual",
                    hue="defaulted",
                    alpha=0.35,
                    palette=[THEME["accent2"], THEME["negative"]],
                    ax=ax3,
                )
                ax3.set_title("Credit score vs PD (Seaborn)")
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format="png", dpi=130, bbox_inches="tight", facecolor=THEME["paper"])
                plt.close(fig3)
                b64 = base64.b64encode(buf3.getvalue()).decode("ascii")
                out.append(f'<div class="img-card"><h4>Seaborn · Score vs PD</h4><img src="data:image/png;base64,{b64}" alt="Score scatter" /></div>')
        except Exception:
            pass
        return "".join(out)

    def _build_d3_sector_html(self) -> str:
        df = self.loan_df.groupby("sector", as_index=False)["ead"].sum()
        df = df.sort_values("ead", ascending=False).head(14)
        total = float(df["ead"].sum())
        data = [{"sector": str(r["sector"])[:28], "ead": float(r["ead"]), "pct": float(r["ead"] / max(total, 1))} for _, r in df.iterrows()]
        payload = json.dumps(data)
        return f"""
    <section class="section" id="d3-atlas">
      <h2 class="section-title">IX — Concentration (D3.js)</h2>
      <p class="section-dek">Animated horizontal bars — share of portfolio EAD by sector. Reinforces surface diversification vs. notional concentration.</p>
      <div id="d3-sector-bars" class="d3-wrap"></div>
      <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
      <script>
      (function() {{
        const data = {payload};
        const margin = {{top: 16, right: 48, bottom: 16, left: 140}};
        const rowH = 26;
        const W = 900;
        const H = margin.top + margin.bottom + data.length * rowH;
        const maxE = d3.max(data, d => d.ead);
        const x = d3.scaleLinear().domain([0, maxE]).range([0, W - margin.left - margin.right]);
        const y = d3.scaleBand().domain(data.map(d => d.sector)).range([0, data.length * rowH]).padding(0.15);
        const svg = d3.select("#d3-sector-bars").append("svg").attr("viewBox", `0 0 ${{W}} ${{H}}`).attr("class", "d3-svg");
        const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
        const fmt = d3.format(",.0f");
        const bars = g.selectAll("rect").data(data).enter().append("rect")
          .attr("y", d => y(d.sector))
          .attr("height", y.bandwidth())
          .attr("x", 0)
          .attr("width", 0)
          .attr("fill", "#1e3a5f")
          .attr("rx", 2);
        bars.transition().duration(900).delay((d,i) => i * 45).ease(d3.easeCubicOut).attr("width", d => x(d.ead));
        g.selectAll("text.lab").data(data).enter().append("text")
          .attr("class", "lab")
          .attr("x", -8)
          .attr("y", d => y(d.sector) + y.bandwidth() / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "11px")
          .attr("fill", "#1c1917")
          .text(d => d.sector);
        g.selectAll("text.val").data(data).enter().append("text")
          .attr("class", "val")
          .attr("x", d => x(d.ead) + 8)
          .attr("y", d => y(d.sector) + y.bandwidth() / 2)
          .attr("dominant-baseline", "middle")
          .attr("font-size", "11px")
          .attr("fill", "#57534e")
          .text(d => fmt(d.ead) + " (" + d3.format(".1%")(d.pct) + ")");
      }})();
      </script>
    </section>"""

    def _fig_to_div(self, fig: go.Figure, include_js: bool) -> str:
        return plotly_plot(
            fig,
            include_plotlyjs=True if include_js else False,
            output_type="div",
            config={"displaylogo": False, "responsive": True},
        )

    def _kpi_cards_html(self) -> str:
        k = self.get_portfolio_kpis()
        order = [
            "Total Exposure ($B)",
            "Weighted Avg Coupon (%)",
            "Weighted Avg LTV / D/E",
            "Default Rate (%)",
            "Average PD (%)",
            "WAM (months)",
            "Unsecured Exposure (%)",
            "EAD in ≥60mo Tenors (%)",
            "Adverse / Baseline EL",
            "Loan Count",
            "Sector Count",
        ]
        cards = []
        for key in order:
            value = k.get(key, np.nan)
            if "($B)" in key:
                txt = f"${value:,.2f}B"
            elif "%" in key or "LTV" in key:
                txt = f"{value:,.2f}%" if not np.isnan(value) else "N/A"
            elif "WAM" in key:
                txt = f"{value:,.1f}" if not np.isnan(value) else "N/A"
            elif key == "Adverse / Baseline EL":
                txt = f"{value:,.2f}×" if not np.isnan(value) else "N/A"
            elif "Count" in key:
                txt = f"{int(value):,}" if not np.isnan(value) else "N/A"
            else:
                txt = f"{value:,.2f}%" if "%" in key else (f"{int(value):,}" if not np.isnan(value) else "N/A")
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

        section_dek = {
            "I — Surface": "Where capital sits: concentration by sector and rating, and realized default rates with confidence bands.",
            "II — Liquidity": "Private credit is often long-dated and semi-liquid — WAM by instrument and tenor dispersion proxy mismatch risk.",
            "III — Macro tape": "Co-movement of embedded risk (PD, EL) with unemployment, spreads, and the flow of new defaults.",
            "IV — Hidden tail": "Rating migration to outcomes and the sector risk–return map — carry versus credit cost.",
            "V — Model": "Out-of-fold discrimination, calibration, and drivers from the gradient-boosting view.",
            "VI — Stress": "Scenario EL in dollars and as a share of EAD — fragility under macro shocks.",
            "VII — Cohorts": "Vintage curves with an adverse PD overlay — early performance vs stress amplification.",
            "VIII — Atlas": "Depth, motion, and geometry — treemap, violins, animated vintage formation, joint densities, parallel coordinates, 3D cloud, stress funnel.",
        }

        figures: List[Tuple[str, str, go.Figure]] = []

        figures.append(("I — Surface", section_dek["I — Surface"], self._fig_sector_sunburst()))
        figures.append(("I — Surface", section_dek["I — Surface"], self._fig_default_rate_sector()))
        figures.append(("II — Liquidity", section_dek["II — Liquidity"], self._fig_liquidity_structure()))
        figures.append(("II — Liquidity", section_dek["II — Liquidity"], self._fig_maturity_distribution()))
        figures.append(("III — Macro tape", section_dek["III — Macro tape"], self._fig_portfolio_tape()))
        figures.append(("III — Macro tape", section_dek["III — Macro tape"], self._fig_new_defaults_flow()))
        figures.append(("IV — Hidden tail", section_dek["IV — Hidden tail"], self._fig_rating_sankey()))
        figures.append(("IV — Hidden tail", section_dek["IV — Hidden tail"], self._fig_risk_return_scatter()))
        roc_fig, pr_fig, cal_fig = self._fig_roc_pr_calibration()
        figures.append(("V — Model", section_dek["V — Model"], roc_fig))
        figures.append(("V — Model", section_dek["V — Model"], pr_fig))
        figures.append(("V — Model", section_dek["V — Model"], cal_fig))
        figures.append(("V — Model", section_dek["V — Model"], self._fig_feature_importance()))
        figures.append(("VI — Stress", section_dek["VI — Stress"], self._fig_stress_totals_by_scenario()))
        figures.append(("VI — Stress", section_dek["VI — Stress"], self._fig_stress_sector_bar()))
        figures.append(("VI — Stress", section_dek["VI — Stress"], self._fig_stress_heatmap()))
        figures.append(("VII — Cohorts", section_dek["VII — Cohorts"], self._fig_vintage_overlay()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_treemap_ead()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_pd_violin()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_origination_animation()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_parallel_coords()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_density_pd_coupon()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_scatter_3d_risk()))
        figures.append(("VIII — Atlas", section_dek["VIII — Atlas"], self._fig_stress_funnel()))

        fig_divs = []
        include_js = True
        current_section = None
        for section, subtitle, fig in figures:
            if section != current_section:
                if current_section is not None:
                    fig_divs.append("</div></section>")
                fig_divs.append(
                    f'<section class="section"><h2 class="section-title">{section}</h2>'
                    f'<p class="section-dek">{subtitle}</p><div class="grid">'
                )
                current_section = section
            fig_divs.append(f'<div class="panel"><h3 class="panel-title"></h3>{self._fig_to_div(fig, include_js)}</div>')
            include_js = False
        if current_section is not None:
            fig_divs.append("</div></section>")

        seaborn_wrap = ""
        se = self._build_seaborn_embeds()
        if se:
            seaborn_wrap = (
                '<section class="section">'
                '<h2 class="section-title">Seaborn · Static atlas</h2>'
                '<p class="section-dek">Publication-quality density, correlation, and bivariate structure — complements interactive Plotly above.</p>'
                f'<div class="images atlas-grid">{se}</div></section>'
            )

        d3_block = self._build_d3_sector_html()

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

        thesis_block = """
    <aside class="thesis">
      <p class="thesis-label">Thesis</p>
      <p class="thesis-lead">Private credit appears stable on headline yields and diversification — yet <em>liquidity mismatch</em> (long, locked-up structures),
      <em>macro drift</em> in spreads and unemployment, and <em>rising default incidence</em> in the tape imply fragility beneath the surface.</p>
      <p class="thesis-note">Synthetic Kaggle-style loan tape · For practice / education only · Not investment advice.</p>
    </aside>"""

        fragility_link = ""
        frag_path = self.output_path / "private_credit" / "fragility_summary.html"
        if frag_path.exists():
            fragility_link = (
                '<p class="dek" style="margin-top:12px;font-size:15px;max-width:60ch;">'
                '<a href="private_credit/fragility_summary.html" style="color:var(--accent2);">'
                "Synthetic private-credit fragility track</a> — separate 600-loan book: Monte Carlo EL, ICR stress, ML deciles.</p>"
            )

        html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Private Credit — Surface Stability & Latent Fragility</title>
  <style>
    :root {{
      --paper: #f7f5f0;
      --ink: #1c1917;
      --muted: #57534e;
      --rule: #d6d3cd;
      --accent: #b45309;
      --accent2: #1e3a5f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.45;
    }}
    .masthead {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 48px 28px 24px;
      border-bottom: 1px solid var(--rule);
    }}
    .kicker {{
      font-size: 11px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    h1 {{
      font-family: Georgia, "Iowan Old Style", Palatino, serif;
      font-weight: 400;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1.15;
      margin: 0 0 12px;
      letter-spacing: -0.02em;
    }}
    .dek {{
      font-size: 17px;
      color: var(--muted);
      max-width: 52ch;
      margin: 0;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 28px 56px; }}
    .thesis {{
      margin: 28px 0 32px;
      padding: 20px 24px;
      border-left: 4px solid var(--accent2);
      background: #fff;
      box-shadow: 0 1px 0 var(--rule);
    }}
    .thesis-label {{ font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }}
    .thesis-lead {{ font-family: Georgia, serif; font-size: 18px; margin: 0; color: var(--ink); }}
    .thesis-lead em {{ font-style: italic; color: var(--accent2); }}
    .thesis-note {{ font-size: 12px; color: var(--muted); margin: 12px 0 0; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0 32px;
    }}
    .kpi {{
      background: #fff;
      border: 1px solid var(--rule);
      padding: 14px 16px;
      min-height: 88px;
    }}
    .kpi-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .kpi-value {{ margin-top: 8px; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ink); }}
    .section {{ margin-top: 40px; }}
    .section-title {{
      font-family: Georgia, serif;
      font-size: 22px;
      font-weight: 400;
      margin: 0 0 6px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--rule);
    }}
    .section-dek {{ margin: 0 0 18px; font-size: 14px; color: var(--muted); max-width: 60ch; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    .panel {{
      background: #fff;
      border: 1px solid var(--rule);
      padding: 8px 8px 4px;
      overflow: hidden;
    }}
    .panel-title {{ display: none; }}
    .images {{ margin-top: 32px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
    .img-card {{ background: #fff; border: 1px solid var(--rule); padding: 12px; }}
    .img-card h4 {{ margin: 0 0 8px; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .img-card img {{ width: 100%; border-radius: 2px; border: 1px solid var(--rule); }}
    .atlas-grid {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
    .d3-wrap {{ background: #fff; border: 1px solid var(--rule); padding: 12px; min-height: 320px; overflow-x: auto; }}
    .d3-svg {{ display: block; max-width: 100%; height: auto; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--rule); font-size: 12px; color: var(--muted); text-align: center; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .images {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="masthead">
    <p class="kicker">Quantitative credit · March 2026</p>
    <h1>Private credit: calm water, shifting currents</h1>
    <p class="dek">A single-page view of concentration, liquidity structure, macro co-movement, model evidence, and stress — testing whether headline stability masks tail risk.</p>
    {fragility_link}
  </header>
  <div class="wrap">
    {thesis_block}
    <section class="kpis-wrap">
      <h2 class="section-title" style="border:none;padding:0;margin-bottom:12px;">Headline metrics</h2>
      <section class="kpis">{self._kpi_cards_html()}</section>
    </section>
    {''.join(fig_divs)}
    {seaborn_wrap}
    {d3_block}
    <section class="section">
      <h2 class="section-title">Supplementary diagnostics</h2>
      <p class="section-dek">Static exports from the modeling pipeline (confusion matrices, score separation, SHAP, heatmaps).</p>
      <div class="images">{''.join(extra_images)}</div>
    </section>
    <div class="footer">Self-contained HTML · Plotly (incl. animations) · Seaborn · D3.js · Motion via sliders &amp; transitions (no heavy GIFs) · Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
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
