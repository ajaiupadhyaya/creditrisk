"""
Rich standalone HTML report for the synthetic private-credit fragility track.
Plotly (interactive), Seaborn/Matplotlib (static), D3.js (animated fund NAV paths).
Reads CSVs under private_credit/data/.
"""
from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path
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

warnings.filterwarnings("ignore")

THEME = {
    "paper": "#f4f1eb",
    "card": "#ffffff",
    "ink": "#1c1917",
    "muted": "#57534e",
    "rule": "#d6d3cd",
    "accent": "#b45309",
    "accent2": "#1e3a5f",
    "positive": "#0f766e",
    "negative": "#9f1239",
    "font": "Iowan Old Style, Georgia, Palatino Linotype, Book Antiqua, serif",
    "font_sans": "system-ui, -apple-system, Segoe UI, Helvetica Neue, Arial, sans-serif",
    "palette": ["#1e3a5f", "#b45309", "#0f766e", "#7c3aed", "#be123c"],
}

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return _repo_root() / "private_credit" / "data"


def _safe_read_csv(name: str) -> pd.DataFrame | None:
    p = _data_dir() / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def _fig_to_div(fig: go.Figure, include_js: bool) -> str:
    return plotly_plot(
        fig,
        include_plotlyjs=include_js,
        output_type="div",
        config={"displaylogo": False, "responsive": True},
    )


def _sns_to_img_tag(fig: plt.Figure, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=THEME["paper"])
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<figure class="sns-fig"><img src="data:image/png;base64,{b64}" alt="{alt}" loading="lazy" /></figure>'


def _build_kpi_strip(
    loans: pd.DataFrame,
    frag: pd.DataFrame,
    preds: pd.DataFrame,
    mc_sum: pd.DataFrame | None,
    icr_sum: pd.DataFrame | None,
) -> str:
    tot_mm = loans["principal_mm"].sum()
    wa_frag = float((frag["fragility_score"] * frag["principal_mm"]).sum() / max(tot_mm, 1e-9))
    distress = loans["status"].isin(
        ["pik_toggle", "amended", "extended", "lme", "default"]
    ).mean()
    pik_share = (loans["coupon_type"] == "pik").mean()
    base_el = mc_sum[mc_sum["scenario"] == "base"]["el_mean"].iloc[0] * 100 if mc_sum is not None and not mc_sum.empty else float("nan")
    sev_el = mc_sum[mc_sum["scenario"] == "severe"]["el_mean"].iloc[0] * 100 if mc_sum is not None and not mc_sum.empty else float("nan")
    icr_row = icr_sum[icr_sum["scenario"].str.contains("Stress", na=False)] if icr_sum is not None else None
    pct_15 = icr_row["pct_below_1_5x"].iloc[0] * 100 if icr_row is not None and not icr_row.empty else float("nan")
    auc_hint = ""
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(preds["distress_flag_actual"], preds["distress_proba"])
        auc_hint = f'<div class="kpi"><span class="kpi-l">Distress model AUC</span><span class="kpi-v">{auc:.3f}</span></div>'
    except Exception:
        pass

    def fmt_pct(x: float) -> str:
        return f"{x:.1f}%" if np.isfinite(x) else "—"

    return f"""
    <div class="kpi-strip">
      <div class="kpi"><span class="kpi-l">Total notional</span><span class="kpi-v">${tot_mm:,.0f}mm</span></div>
      <div class="kpi"><span class="kpi-l">Loans</span><span class="kpi-v">{len(loans):,}</span></div>
      <div class="kpi"><span class="kpi-l">Shadow distress</span><span class="kpi-v">{fmt_pct(distress * 100)}</span></div>
      <div class="kpi"><span class="kpi-l">PIK / toggle share</span><span class="kpi-v">{fmt_pct(pik_share * 100)}</span></div>
      <div class="kpi"><span class="kpi-l">Wtd. fragility</span><span class="kpi-v">{wa_frag:.1f}</span></div>
      <div class="kpi"><span class="kpi-l">MC EL base → severe</span><span class="kpi-v">{fmt_pct(base_el)} → {fmt_pct(sev_el)}</span></div>
      <div class="kpi"><span class="kpi-l">ICR &lt;1.5× @ +100bp</span><span class="kpi-v">{fmt_pct(pct_15)}</span></div>
      {auc_hint}
    </div>"""


def _d3_fund_nav_block(nav_paths: pd.DataFrame | None) -> str:
    if nav_paths is None or nav_paths.empty:
        return ""
    # One fund for clarity — Blue Owl if present, else first name
    names = nav_paths["fund_name"].unique()
    pick = "Blue Owl Corporate Lending" if "Blue Owl Corporate Lending" in names else names[0]
    sub = nav_paths[nav_paths["fund_name"] == pick].copy()
    series = []
    for scen in sub["scenario"].unique():
        s = sub[sub["scenario"] == scen].sort_values("quarter")
        series.append(
            {
                "name": str(scen),
                "points": [{"q": int(r["quarter"]), "nav": float(r["nav_mm"])} for _, r in s.iterrows()],
            }
        )
    payload = json.dumps({"fund": pick, "series": series})
    return f"""
    <section class="section" id="d3-liquidity">
      <h2>VIII — Liquidity paths (D3.js)</h2>
      <p class="section-dek">Simulated NAV ($mm) over eight quarters under redemption scenarios — <strong>{pick}</strong>. Lines animate on load.</p>
      <div id="d3-chart" class="d3-wrap"></div>
      <p class="fine">Powered by D3 v7 · Paths from <code>fund_nav_paths.csv</code></p>
      <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
      <script>
      (function() {{
        const payload = {payload};
        const margin = {{top: 28, right: 28, bottom: 44, left: 56}};
        const W = Math.min(920, document.getElementById("d3-chart").clientWidth || 920);
        const H = 380;
        const iw = W - margin.left - margin.right;
        const ih = H - margin.top - margin.bottom;
        const svg = d3.select("#d3-chart").append("svg").attr("viewBox", `0 0 ${{W}} ${{H}}`).attr("class", "d3-svg");
        const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
        const allNav = payload.series.flatMap(s => s.points.map(p => p.nav));
        const x = d3.scaleLinear().domain([1, 8]).range([0, iw]);
        const y = d3.scaleLinear().domain([d3.min(allNav) * 0.98, d3.max(allNav) * 1.02]).range([ih, 0]);
        const color = d3.scaleOrdinal().domain(payload.series.map(s => s.name)).range({json.dumps(THEME["palette"])});
        const line = d3.line().x(d => x(d.q)).y(d => y(d.nav)).curve(d3.curveMonotoneX);
        g.append("g").attr("class", "axis").attr("transform", `translate(0,${{ih}})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
        g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6).tickFormat(d => d3.format(",.0f")(d)));
        g.append("text").attr("x", iw/2).attr("y", ih + 36).attr("text-anchor", "middle").attr("fill", "#57534e").attr("font-size", "12px").text("Quarter");
        g.append("text").attr("transform", "rotate(-90)").attr("x", -ih/2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#57534e").attr("font-size", "12px").text("NAV ($mm)");
        payload.series.forEach((s, i) => {{
          const path = g.append("path").datum(s.points).attr("fill", "none").attr("stroke", color(s.name)).attr("stroke-width", 2.2).attr("d", line);
          const len = path.node().getTotalLength();
          path.attr("stroke-dasharray", len + " " + len).attr("stroke-dashoffset", len).transition().duration(1400).delay(i * 180).ease(d3.easeCubicOut).attr("stroke-dashoffset", 0);
        }});
        const leg = svg.append("g").attr("transform", `translate(${{margin.left}}, 8)`);
        payload.series.forEach((s, i) => {{
          leg.append("rect").attr("x", i * 200).attr("width", 12).attr("height", 12).attr("fill", color(s.name));
          leg.append("text").attr("x", i * 200 + 18).attr("y", 11).attr("font-size", "11px").attr("fill", "#1c1917").text(s.name);
        }});
      }})();
      </script>
    </section>"""


def build_fragility_report(output_path: Path | None = None) -> Path | None:
    data = _data_dir()
    fs = data / "fragility_scores.csv"
    mp = data / "ml_predictions.csv"
    loans_path = data / "loans.csv"
    if not fs.exists() or not mp.exists() or not loans_path.exists():
        return None

    frag = pd.read_csv(fs)
    preds = pd.read_csv(mp)
    loans = pd.read_csv(loans_path)
    mc_sum = _safe_read_csv("mc_summary.csv")
    mc_loss = _safe_read_csv("mc_loss_samples.csv")
    icr_sum = _safe_read_csv("icr_scenario_summary.csv")
    sector_icr = _safe_read_csv("sector_icr.csv")
    mat_wall = _safe_read_csv("maturity_wall.csv")
    pik_traj = _safe_read_csv("pik_trajectory_example.csv")
    fund_liq = _safe_read_csv("fund_liquidity.csv")
    nav_paths = _safe_read_csv("fund_nav_paths.csv")
    rates = _safe_read_csv("rate_scenarios.csv")

    if output_path is None:
        out_dir = _repo_root() / "outputs" / "private_credit"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "fragility_summary.html"

    plot_divs: list[str] = []
    first = True

    def add_plot(fig: go.Figure) -> None:
        nonlocal first
        plot_divs.append(_fig_to_div(fig, first))
        first = False

    # --- I. Monte Carlo tail ---
    if mc_loss is not None and not mc_loss.empty:
        fig_mc = px.histogram(
            mc_loss,
            x="loss_pct",
            color="scenario",
            nbins=60,
            opacity=0.72,
            color_discrete_sequence=THEME["palette"],
            title="Portfolio loss rate — Monte Carlo (50k paths × scenario, subsampled)",
        )
        fig_mc.update_layout(
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(title="Loss / notional", tickformat=".1%"),
            yaxis=dict(title="Count"),
        )
        add_plot(fig_mc)

    if mc_sum is not None and not mc_sum.empty:
        m = mc_sum.melt(
            id_vars=["scenario"],
            value_vars=["el_mean", "var_95", "cvar_99"],
            var_name="metric",
            value_name="value",
        )
        fig_bar = px.bar(
            m,
            x="scenario",
            y="value",
            color="metric",
            barmode="group",
            title="Tail metrics — mean EL, VaR 95%, CVaR 99%",
            color_discrete_map={
                "el_mean": THEME["accent2"],
                "var_95": THEME["accent"],
                "cvar_99": THEME["negative"],
            },
        )
        fig_bar.update_layout(
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
            yaxis=dict(tickformat=".1%"),
            legend=dict(title=""),
        )
        add_plot(fig_bar)

    # --- II. ICR stress ladder ---
    if icr_sum is not None and not icr_sum.empty:
        icr_f = icr_sum.copy()
        icr_f["label"] = icr_f["scenario"].str.replace("$", "")
        fig_icr = go.Figure()
        fig_icr.add_trace(
            go.Bar(
                name="ICR &lt; 1.0×",
                x=icr_f["label"],
                y=icr_f["pct_below_1x"] * 100,
                marker_color=THEME["negative"],
            )
        )
        fig_icr.add_trace(
            go.Bar(
                name="ICR &lt; 1.5×",
                x=icr_f["label"],
                y=icr_f["pct_below_1_5x"] * 100,
                marker_color=THEME["accent"],
            )
        )
        fig_icr.add_trace(
            go.Bar(
                name="ICR &lt; 2.0×",
                x=icr_f["label"],
                y=icr_f["pct_below_2x"] * 100,
                marker_color=THEME["accent2"],
            )
        )
        fig_icr.update_layout(
            barmode="group",
            title="Interest coverage stress ladder — share of book below ICR thresholds",
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
            yaxis=dict(title="% of loans", range=[0, max(100, icr_f["pct_below_2x"].max() * 100 * 1.15)]),
            legend=dict(orientation="h", y=1.1),
        )
        add_plot(fig_icr)

        fig_icr_line = go.Figure()
        fig_icr_line.add_trace(
            go.Scatter(
                x=icr_f["label"],
                y=icr_f["avg_icr"],
                mode="lines+markers",
                line=dict(color=THEME["accent2"], width=3),
                marker=dict(size=10),
                name="Avg ICR (×)",
            )
        )
        fig_icr_line.update_layout(
            title="Average ICR by SOFR path",
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            yaxis=dict(title="ICR (×)"),
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
        )
        add_plot(fig_icr_line)

    # --- III. Sector heat ---
    if sector_icr is not None and not sector_icr.empty:
        s = sector_icr.copy()
        if s.columns[0] != "sector" and "sector" not in s.columns:
            s = s.rename(columns={s.columns[0]: "sector"})
        z = s[["pct_below_1_5x", "distress_rate", "pik_rate"]].values.T
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=s["sector"].astype(str).tolist(),
                y=["ICR &lt;1.5×", "Distress rate", "PIK rate"],
                colorscale="Blues",
                hoverongaps=False,
            )
        )
        fig_hm.update_layout(
            title="Sector tape — stress & structure (heatmap)",
            paper_bgcolor=THEME["paper"],
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=11),
            yaxis=dict(tickmode="array"),
            height=420,
        )
        add_plot(fig_hm)

    # --- IV. Maturity wall ---
    if mat_wall is not None and not mat_wall.empty:
        mw = mat_wall.sort_values(mat_wall.columns[0]).head(24)
        xcol = mw.columns[0]
        fig_m = go.Figure(
            go.Bar(
                x=mw[xcol].astype(str),
                y=mw["principal_mm"],
                marker=dict(
                    color=mw["distressed"],
                    colorscale=[[0, THEME["accent2"]], [1, THEME["negative"]]],
                    showscale=True,
                    colorbar=dict(title="Distressed<br>count"),
                ),
            )
        )
        fig_m.update_layout(
            title="Maturity wall — notional maturing by quarter (24m window)",
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            xaxis=dict(title="Quarter", tickangle=-45),
            yaxis=dict(title="$mm"),
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=11),
        )
        add_plot(fig_m)

    # --- V. Macro tape (rates) ---
    if rates is not None and not rates.empty:
        r = rates.copy()
        r["effective_dt"] = pd.to_datetime(r["effective_dt"])
        fig_r = make_subplots(specs=[[{"secondary_y": True}]])
        fig_r.add_trace(
            go.Scatter(x=r["effective_dt"], y=r["sofr"], name="SOFR", line=dict(color=THEME["accent2"], width=2)),
            secondary_y=False,
        )
        if "recession_prob" in r.columns:
            fig_r.add_trace(
                go.Scatter(
                    x=r["effective_dt"],
                    y=r["recession_prob"],
                    name="Recession prob",
                    line=dict(color=THEME["negative"], width=1, dash="dot"),
                ),
                secondary_y=True,
            )
        fig_r.update_layout(
            title="Macro scenario tape — SOFR path (and recession probability if present)",
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
            legend=dict(orientation="h", y=1.1),
        )
        fig_r.update_yaxes(title_text="SOFR", secondary_y=False)
        fig_r.update_yaxes(title_text="Probability", secondary_y=True, rangemode="nonnegative")
        add_plot(fig_r)

    # --- VI. Fragility & ML ---
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(x=frag["fragility_score"], nbinsx=36, marker_color=THEME["accent2"], opacity=0.88, name="Fragility")
    )
    fig_hist.update_layout(
        title="Composite fragility score distribution",
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        xaxis=dict(title="Score"),
        yaxis=dict(title="Loans"),
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
    )
    add_plot(fig_hist)

    tier = frag.groupby("risk_tier", as_index=False)["principal_mm"].sum()
    order = ["Low", "Medium", "High", "Critical"]
    tier["risk_tier"] = pd.Categorical(tier["risk_tier"], categories=order, ordered=True)
    tier = tier.sort_values("risk_tier")
    cmap = {"Low": THEME["positive"], "Medium": THEME["accent2"], "High": THEME["accent"], "Critical": THEME["negative"]}
    tier_colors = [cmap.get(str(x), THEME["accent2"]) for x in tier["risk_tier"]]
    fig_tier = go.Figure(
        go.Bar(
            x=tier["risk_tier"].astype(str),
            y=tier["principal_mm"],
            text=[f"{v:,.0f}" for v in tier["principal_mm"]],
            textposition="outside",
            marker=dict(color=tier_colors),
        )
    )
    fig_tier.update_layout(
        title="Notional by fragility tier",
        paper_bgcolor=THEME["paper"],
        showlegend=False,
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
        yaxis=dict(title="$mm"),
    )
    add_plot(fig_tier)

    base_rate = preds["distress_flag_actual"].mean()
    dec = preds.groupby("distress_decile", as_index=False).agg(
        n=("loan_id", "count"),
        distress=("distress_flag_actual", "sum"),
        notional=("principal_mm", "sum"),
    )
    dec["distress_rate"] = dec["distress"] / dec["n"].replace(0, np.nan)
    dec["lift"] = dec["distress_rate"] / base_rate if base_rate > 0 else np.nan
    fig_lift = make_subplots(specs=[[{"secondary_y": True}]])
    fig_lift.add_trace(
        go.Bar(x=dec["distress_decile"], y=dec["distress_rate"] * 100, name="Distress %", marker_color=THEME["accent"]),
        secondary_y=False,
    )
    fig_lift.add_trace(
        go.Scatter(
            x=dec["distress_decile"],
            y=dec["lift"],
            name="Lift",
            mode="lines+markers",
            line=dict(color=THEME["accent2"], width=2),
        ),
        secondary_y=True,
    )
    fig_lift.update_layout(
        title="ML distress model — decile lift (in-sample)",
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        legend=dict(orientation="h", y=1.08),
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
    )
    fig_lift.update_yaxes(title_text="Distress rate %", secondary_y=False)
    fig_lift.update_yaxes(title_text="Lift vs base", secondary_y=True)
    add_plot(fig_lift)

    merged = preds.merge(frag[["loan_id", "fragility_score", "risk_tier"]], on="loan_id", how="left")
    merged = merged.merge(loans[["loan_id", "spread_bps"]], on="loan_id", how="left")
    fig_sc = px.scatter(
        merged,
        x="distress_proba",
        y="spread_bps",
        color="risk_tier",
        size="principal_mm",
        hover_data=["borrower_name", "sector", "vintage"],
        color_discrete_map={
            "Low": THEME["positive"],
            "Medium": THEME["accent2"],
            "High": THEME["accent"],
            "Critical": THEME["negative"],
        },
        title="Model probability vs. spread — sized by notional",
    )
    fig_sc.update_layout(
        paper_bgcolor=THEME["paper"],
        plot_bgcolor="#fff",
        font=dict(family=THEME["font_sans"], color=THEME["ink"], size=11),
    )
    add_plot(fig_sc)

    # Animated loss histogram (slider) if MC data exists
    if mc_loss is not None and len(mc_loss["scenario"].unique()) > 1:
        fig_anim = px.histogram(
            mc_loss,
            x="loss_pct",
            animation_frame="scenario",
            nbins=45,
            range_x=[0, float(mc_loss["loss_pct"].quantile(0.9995))],
            title="Loss distribution — use play/slider to compare scenarios",
        )
        fig_anim.update_layout(
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=12),
            xaxis=dict(tickformat=".1%"),
        )
        add_plot(fig_anim)

    # --- VII. Fund liquidity (Plotly) ---
    if fund_liq is not None and not fund_liq.empty:
        fig_f = px.bar(
            fund_liq,
            x="fund_name",
            y="nav_erosion_pct",
            color="scenario",
            barmode="group",
            title="Fund liquidity stress — terminal NAV erosion by scenario",
            color_discrete_sequence=THEME["palette"],
        )
        fig_f.update_layout(
            paper_bgcolor=THEME["paper"],
            plot_bgcolor="#fff",
            xaxis=dict(tickangle=-35),
            yaxis=dict(tickformat=".0%", title="NAV erosion"),
            font=dict(family=THEME["font_sans"], color=THEME["ink"], size=11),
            legend=dict(orientation="h", y=1.12),
            height=480,
        )
        add_plot(fig_f)

    # --- Seaborn static panels ---
    sns_html: list[str] = []
    try:
        lf = loans.merge(frag[["loan_id", "fragility_score", "risk_tier"]], on="loan_id")
        fig_kde, ax = plt.subplots(figsize=(7.2, 4.2))
        for tier, sub in lf.groupby("risk_tier"):
            sns.kdeplot(data=sub, x="icr_current", fill=True, alpha=0.35, label=str(tier), ax=ax)
        ax.set_xlabel("ICR (current)")
        ax.set_title("ICR density by fragility tier (Seaborn KDE)")
        ax.legend(title="Tier", fontsize=8)
        sns_html.append(_sns_to_img_tag(fig_kde, "ICR KDE by tier"))

        lf2 = lf.merge(preds[["loan_id", "distress_proba"]], on="loan_id") if "distress_proba" in preds.columns else lf
        fig_joint = plt.figure(figsize=(6.8, 6))
        gs = fig_joint.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax_m = fig_joint.add_subplot(gs[0, 0])
        ax_c = fig_joint.add_subplot(gs[1, 0])
        ax_r = fig_joint.add_subplot(gs[1, 1])
        ax_m.axis("off")
        hue_col = "distress_proba" if "distress_proba" in lf2.columns else "risk_tier"
        sns.scatterplot(
            data=lf2,
            x="leverage_x",
            y="icr_current",
            hue=hue_col,
            palette="rocket" if hue_col == "distress_proba" else None,
            alpha=0.65,
            ax=ax_c,
            legend=(hue_col != "distress_proba"),
        )
        sns.kdeplot(data=lf, x="leverage_x", fill=True, ax=ax_m, color=THEME["accent2"], alpha=0.5)
        sns.kdeplot(data=lf, y="icr_current", fill=True, ax=ax_r, color=THEME["accent2"], alpha=0.5)
        ax_c.set_xlabel("Leverage (×)")
        ax_c.set_ylabel("ICR (×)")
        fig_joint.suptitle("Leverage vs ICR — joint structure (Seaborn)", fontsize=11, y=0.98)
        sns_html.append(_sns_to_img_tag(fig_joint, "Joint leverage ICR"))

        if pik_traj is not None and not pik_traj.empty:
            fig_p, axp = plt.subplots(figsize=(7, 3.8))
            axp.fill_between(pik_traj["year"], 0, pik_traj["balance_mm"], where=pik_traj["is_pik"], alpha=0.3, color=THEME["negative"], label="PIK accrual zone")
            axp.plot(pik_traj["year"], pik_traj["balance_mm"], color=THEME["accent2"], lw=2)
            axp.set_xlabel("Year")
            axp.set_ylabel("Balance ($mm)")
            axp.set_title("Illustrative PIK toggle — balance path (example loan)")
            axp.legend()
            sns_html.append(_sns_to_img_tag(fig_p, "PIK trajectory"))

        corr_cols = [c for c in ["icr_current", "leverage_x", "fragility_score", "spread_bps", "ltv"] if c in lf.columns]
        if len(corr_cols) >= 3:
            cm = lf[corr_cols].corr()
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax_cm, vmin=-1, vmax=1)
            ax_cm.set_title("Feature correlation (Seaborn)")
            sns_html.append(_sns_to_img_tag(fig_cm, "Correlation heatmap"))
    except Exception:
        pass

    kpi_html = _build_kpi_strip(loans, frag, preds, mc_sum, icr_sum)
    d3_block = _d3_fund_nav_block(nav_paths)

    sections = "".join(f'<div class="panel plot-panel">{d}</div>' for d in plot_divs)
    sns_block = ""
    if sns_html:
        sns_block = '<section class="section" id="sns"><h2>IX — Distribution geometry (Seaborn)</h2><p class="section-dek">Static high-DPI panels for print and slow journalism–style reading.</p><div class="sns-grid">' + "".join(sns_html) + "</div></section>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Private credit fragility — analytic atlas</title>
  <style>
    :root {{
      --paper: {THEME["paper"]};
      --card: {THEME["card"]};
      --ink: {THEME["ink"]};
      --muted: {THEME["muted"]};
      --rule: {THEME["rule"]};
      --accent: {THEME["accent"]};
      --accent2: {THEME["accent2"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: {THEME["font_sans"]}; background: var(--paper); color: var(--ink); margin: 0; line-height: 1.5; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }}
    .masthead {{ border-bottom: 1px solid var(--rule); padding-bottom: 24px; margin-bottom: 28px; }}
    .kicker {{ font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }}
    h1 {{ font-family: {THEME["font"]}; font-weight: 400; font-size: clamp(26px, 3.5vw, 38px); margin: 0 0 12px; letter-spacing: -0.02em; }}
    .dek {{ color: var(--muted); max-width: 68ch; font-size: 16px; margin: 0; }}
    .back {{ font-size: 13px; margin-bottom: 20px; }}
    .back a {{ color: var(--accent2); text-decoration: none; border-bottom: 1px solid transparent; }}
    .back a:hover {{ border-bottom-color: var(--accent2); }}
    nav.jump {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 20px 0 8px; font-size: 12px; }}
    nav.jump a {{ color: var(--accent); text-decoration: none; }}
    nav.jump a:hover {{ text-decoration: underline; }}
    .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin: 24px 0 8px; }}
    .kpi {{ background: var(--card); border: 1px solid var(--rule); padding: 12px 14px; min-height: 76px; }}
    .kpi-l {{ display: block; font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .kpi-v {{ display: block; margin-top: 6px; font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .section {{ margin-top: 36px; }}
    .section h2 {{ font-family: Georgia, serif; font-size: 20px; font-weight: 400; margin: 0 0 8px; color: var(--accent2); }}
    .section-dek {{ margin: 0 0 16px; font-size: 14px; color: var(--muted); max-width: 70ch; }}
    .panel {{ background: var(--card); border: 1px solid var(--rule); margin-bottom: 18px; padding: 10px 8px 4px; border-radius: 2px; }}
    .plot-panel {{ overflow: hidden; }}
    .sns-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; align-items: start; }}
    .sns-fig {{ margin: 0; background: var(--card); border: 1px solid var(--rule); padding: 8px; }}
    .sns-fig img {{ width: 100%; height: auto; display: block; }}
    .d3-wrap {{ background: var(--card); border: 1px solid var(--rule); min-height: 400px; }}
    .d3-svg {{ width: 100%; height: auto; display: block; }}
    .axis path, .axis line {{ stroke: var(--rule); }}
    .fine {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--rule); font-size: 12px; color: var(--muted); text-align: center; }}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="back"><a href="../dashboard.html">← Main credit dashboard</a></p>
    <header class="masthead">
      <p class="kicker">Thesis atlas · Synthetic private credit</p>
      <h1>Calm surface, measurable fragility</h1>
      <p class="dek">Monte Carlo tail risk, coverage ladders, sector heat, maturity walls, fund liquidity, and model lift — exploring whether headline stability masks compounding stress.</p>
    </header>
    {kpi_html}
    <nav class="jump" aria-label="Sections">
      <a href="#plots">Plotly</a>
      <a href="#d3-liquidity">D3 liquidity</a>
      <a href="#sns">Seaborn</a>
    </nav>
    <section class="section" id="plots">
      <h2>I–VII — Interactive layers (Plotly)</h2>
      <p class="section-dek">Pan, zoom, and hover. Scenario slider on the animated loss histogram compares macro paths without re-running the engine.</p>
      {sections}
    </section>
    {d3_block}
    {sns_block}
    <div class="footer">Generated {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} · Plotly + Seaborn + D3</div>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Fragility report written: {output_path}")
    return output_path


if __name__ == "__main__":
    build_fragility_report()
