"""
explainer.py
------------
LLM-powered recommendation explainer for the AI Hub Procurement Engine.

For each recommendation in recommendations.csv, calls the Claude API
to generate a plain-English explanation of WHY this supplier was chosen.

This satisfies the Mar 13 MOM requirement:
    "The model must provide a math breakdown or rationale for its
     recommendations to assist procurement managers in decision-making."

We already have the math breakdown. This module wraps it in human language
that a procurement manager can read without knowing Python or data science.

Reads:
    data/processed/recommendations.csv
    data/processed/baseline_comparison.csv   (optional — adds savings context)

Outputs:
    data/processed/recommendations_explained.csv   <- adds 'explanation' column
    data/processed/explanation_report.html         <- presentation-ready report

Usage:
    # Explain all recommendations
    python src/engine/explainer.py

    # Explain top N by savings
    python src/engine/explainer.py --top 10

    # Explain specific station
    python src/engine/explainer.py --station OXG-XAJI0
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed — try manual .env parse
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PROCESSED = Path("data/processed")

PATHS = {
    "recommendations": DATA_PROCESSED / "recommendations.csv",
    "comparison":      DATA_PROCESSED / "baseline_comparison.csv",
    "output_csv":      DATA_PROCESSED / "recommendations_explained.csv",
    "output_html":     DATA_PROCESSED / "explanation_report.html",
}

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-20250514"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM explanation generator
# ---------------------------------------------------------------------------

def build_prompt(row: pd.Series, comp_row: pd.Series = None) -> str:
    """
    Build a structured prompt for Claude from a recommendation row.
    Includes all relevant math so Claude can explain the reasoning.
    """
    forced  = bool(row.get("pemex_forced", False))
    savings = row.get("savings_vs_rank2_mxn", 0)
    comp_savings = float(comp_row["saving_mxn"]) if comp_row is not None else None

    context = f"""
You are an AI procurement assistant explaining a supplier recommendation to a Mexican fuel procurement manager.

RECOMMENDATION DETAILS:
- Station: {row['id_tienda']}
- Product: {row['product_type']}
- Order date: {row['order_date']}
- Required by: {row.get('mb_required_by_date', 'N/A')} ({row.get('mb_deadline_days', 'N/A')} days until tank empty)
- Order quantity: {int(row.get('mb_order_qty_litros', 0)):,} liters

RECOMMENDED SUPPLIER: {row['recommended_supplier']}
- Invoice price: {row.get('mb_base_price_mxn_per_l', 0):.4f} MXN/L
- Distance to station: {row.get('mb_dist_km', 0):.1f} km
- Freight cost: {row.get('mb_freight_mxn_per_l', 0):.4f} MXN/L ({row.get('mb_freight_cost_mxn', 0):.0f} MXN total trip)
- Surcharge: {row.get('mb_surcharge_mxn_per_l', 0):.4f} MXN/L
- TRUE LANDED COST: {row.get('mb_true_landed_cost', 0):.4f} MXN/L
- Total order cost: MXN {row.get('mb_total_order_cost_mxn', 0):,.0f}
- Delivery reliability: {row.get('mb_reliability_score', 0)*100:.1f}% on-time rate
- Avg delivery time: {row.get('mb_avg_delivery_days', 0):.1f} days

PEMEX LEGAL CONSTRAINT:
- PEMEX order forced by law: {'YES' if forced else 'NO'}
- Monthly minimum required: {row.get('mb_pemex_minimum_litros', 0):,.0f} L
- Bought from PEMEX so far this month: {row.get('mb_pemex_bought_this_month', 0):,.0f} L
- Current compliance: {row.get('mb_pemex_compliance_pct', 0):.1f}%

SAVINGS:
- Savings vs next best option: MXN {savings:,.0f}
{f'- Savings vs baseline (historico): MXN {comp_savings:,.0f} ({comp_row["saving_pct"]:.1f}%)' if comp_savings is not None else ''}

Write a 2-3 sentence explanation in plain English for the procurement manager.
- If PEMEX was forced: explain why the law requires it and what the compliance status is.
- If PEMEX was NOT forced: explain why this supplier has the lowest landed cost and what the freight advantage is.
- Always mention the true landed cost and the savings vs baseline if available.
- Be specific with numbers. Be concise. No bullet points. No headers.
- Write as if you are the AI system speaking directly to the manager.
"""
    return context.strip()


def call_claude(prompt: str, api_key: str) -> str:
    """Call Claude API and return the explanation text."""
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      MODEL,
        "max_tokens": 200,
        "messages":   [{"role": "user", "content": prompt}],
    }

    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_html_report(explained: pd.DataFrame) -> str:
    """
    Generate a presentation-ready HTML report showing explained recommendations.
    Dark theme matching the project presentation aesthetic.
    """
    cards = []
    for _, row in explained.iterrows():
        forced_badge = (
            '<span class="badge forced">⚖️ PEMEX Legal Minimum</span>'
            if row.get("pemex_forced") else
            '<span class="badge optimized">✦ Cost Optimized</span>'
        )

        supplier_color = {
            "Pemex":      "#00C896",
            "ExxonMobil": "#4A9EFF",
            "Exxon":      "#4A9EFF",
            "Valero":     "#FF6B35",
            "Marathon":   "#FFD166",
        }.get(str(row.get("recommended_supplier", "")), "#AAAAAA")

        savings_raw  = row.get("saving_mxn", 0)
        savings_text = (
            f'<span class="saving positive">▲ MXN {abs(savings_raw):,.0f} saved vs baseline</span>'
            if savings_raw > 0 else
            f'<span class="saving negative">▼ MXN {abs(savings_raw):,.0f} above baseline</span>'
        )

        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <div class="station-info">
                    <span class="station-id">{row['id_tienda']}</span>
                    <span class="product-tag">{row['product_type']}</span>
                    <span class="date-tag">{row['order_date']}</span>
                </div>
                <div class="supplier-badge" style="color:{supplier_color}">
                    {row.get('recommended_supplier', 'N/A')}
                </div>
            </div>
            <div class="card-body">
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-val">{row.get('mb_true_landed_cost', 0):.4f}</div>
                        <div class="metric-label">Landed Cost MXN/L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-val">{int(row.get('mb_order_qty_litros', 0)):,}</div>
                        <div class="metric-label">Litros</div>
                    </div>
                    <div class="metric">
                        <div class="metric-val">MXN {row.get('mb_total_order_cost_mxn', 0):,.0f}</div>
                        <div class="metric-label">Total Order Cost</div>
                    </div>
                    <div class="metric">
                        <div class="metric-val">{row.get('mb_dist_km', 0):.1f} km</div>
                        <div class="metric-label">Terminal Distance</div>
                    </div>
                </div>
                <div class="math-row">
                    <span class="math-item">Base: <b>{row.get('mb_base_price_mxn_per_l', 0):.4f}</b></span>
                    <span class="math-sep">+</span>
                    <span class="math-item">Freight: <b>{row.get('mb_freight_mxn_per_l', 0):.4f}</b></span>
                    <span class="math-sep">+</span>
                    <span class="math-item">Surcharge: <b>{row.get('mb_surcharge_mxn_per_l', 0):.4f}</b></span>
                    <span class="math-sep">=</span>
                    <span class="math-total">{row.get('mb_true_landed_cost', 0):.4f} MXN/L</span>
                </div>
                <div class="explanation">{row.get('explanation', '')}</div>
                <div class="card-footer">
                    {forced_badge}
                    {savings_text}
                    <span class="deadline">🕐 Required by {row.get('mb_required_by_date', 'N/A')}</span>
                </div>
            </div>
        </div>""")

    cards_html = "\n".join(cards)

    # Summary stats
    total_orders   = len(explained)
    pemex_forced   = int(explained["pemex_forced"].sum()) if "pemex_forced" in explained.columns else 0
    total_savings  = explained["saving_mxn"].sum() if "saving_mxn" in explained.columns else 0
    avg_landed     = explained["mb_true_landed_cost"].mean() if "mb_true_landed_cost" in explained.columns else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Hub Procurement Engine — Recommendation Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #080F1A;
    --surface:  #0D1B2E;
    --card:     #112240;
    --border:   #1E3A5F;
    --accent:   #00C896;
    --accent2:  #4A9EFF;
    --text:     #E8F0F8;
    --muted:    #7A9BB5;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 0;
  }}

  .header {{
    background: linear-gradient(135deg, #0A1628 0%, #0D2040 50%, #0A1628 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,200,150,0.08) 0%, transparent 70%);
    pointer-events: none;
  }}
  .header-eyebrow {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .header-title {{
    font-size: 32px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
    margin-bottom: 8px;
  }}
  .header-sub {{
    font-size: 14px;
    color: var(--muted);
    font-weight: 300;
  }}
  .header-stats {{
    display: flex;
    gap: 40px;
    margin-top: 28px;
  }}
  .hstat {{
    display: flex;
    flex-direction: column;
  }}
  .hstat-val {{
    font-family: var(--mono);
    font-size: 24px;
    font-weight: 600;
    color: var(--accent);
  }}
  .hstat-label {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
  }}

  .body {{ padding: 32px 48px; max-width: 1200px; margin: 0 auto; }}

  .section-title {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  .cards {{ display: flex; flex-direction: column; gap: 20px; }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: rgba(0,200,150,0.3); }}

  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid var(--border);
  }}
  .station-info {{ display: flex; align-items: center; gap: 10px; }}
  .station-id {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }}
  .product-tag, .date-tag {{
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: var(--mono);
  }}
  .product-tag {{
    background: rgba(74,158,255,0.15);
    color: var(--accent2);
    border: 1px solid rgba(74,158,255,0.2);
  }}
  .date-tag {{
    background: rgba(255,255,255,0.05);
    color: var(--muted);
    border: 1px solid var(--border);
  }}
  .supplier-badge {{
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}

  .card-body {{ padding: 20px; }}

  .metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }}
  .metric {{
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }}
  .metric-val {{
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 4px;
  }}
  .metric-label {{
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .math-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(0,200,150,0.04);
    border: 1px solid rgba(0,200,150,0.15);
    border-radius: 6px;
    margin-bottom: 16px;
    font-family: var(--mono);
    font-size: 12px;
    flex-wrap: wrap;
  }}
  .math-item {{ color: var(--muted); }}
  .math-item b {{ color: var(--text); }}
  .math-sep {{ color: var(--border); font-weight: 600; }}
  .math-total {{ color: var(--accent); font-weight: 600; font-size: 13px; margin-left: auto; }}

  .explanation {{
    font-size: 14px;
    line-height: 1.7;
    color: #B8CCE0;
    padding: 14px 16px;
    background: rgba(255,255,255,0.02);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    margin-bottom: 14px;
    font-style: italic;
  }}

  .card-footer {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .badge {{
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: var(--mono);
    font-weight: 600;
  }}
  .badge.forced {{
    background: rgba(255,165,0,0.12);
    color: #FFB347;
    border: 1px solid rgba(255,165,0,0.2);
  }}
  .badge.optimized {{
    background: rgba(0,200,150,0.12);
    color: var(--accent);
    border: 1px solid rgba(0,200,150,0.2);
  }}
  .saving {{
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 4px;
  }}
  .saving.positive {{
    background: rgba(0,200,150,0.08);
    color: var(--accent);
    border: 1px solid rgba(0,200,150,0.15);
  }}
  .saving.negative {{
    background: rgba(255,80,80,0.08);
    color: #FF6B6B;
    border: 1px solid rgba(255,80,80,0.15);
  }}
  .deadline {{
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    margin-left: auto;
  }}

  .footer {{
    text-align: center;
    padding: 32px;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--mono);
    border-top: 1px solid var(--border);
    margin-top: 48px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-eyebrow">AI Hub Procurement Engine · Team GRAV · Globant</div>
  <div class="header-title">LLM-Powered Recommendation Report</div>
  <div class="header-sub">Monterrey Region · Q1 2024 · Natural language explanations generated by Claude AI</div>
  <div class="header-stats">
    <div class="hstat">
      <div class="hstat-val">{total_orders:,}</div>
      <div class="hstat-label">Order Events</div>
    </div>
    <div class="hstat">
      <div class="hstat-val">{pemex_forced:,}</div>
      <div class="hstat-label">PEMEX Forced</div>
    </div>
    <div class="hstat">
      <div class="hstat-val">MXN {avg_landed:.4f}/L</div>
      <div class="hstat-label">Avg Landed Cost</div>
    </div>
    <div class="hstat">
      <div class="hstat-val">MXN {total_savings:,.0f}</div>
      <div class="hstat-label">Total Savings vs Baseline</div>
    </div>
  </div>
</div>

<div class="body">
  <div class="section-title">Procurement Recommendations — AI Explanations</div>
  <div class="cards">
    {cards_html}
  </div>
</div>

<div class="footer">
  AI Hub Procurement Engine · Team GRAV · University at Buffalo · Globant · {pd.Timestamp.now().strftime("%B %Y")}
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_explainer(
    top_n: int = None,
    station: str = None,
    api_key: str = None,
) -> pd.DataFrame:

    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )

    # Load recommendations
    if not PATHS["recommendations"].exists():
        raise FileNotFoundError(
            "recommendations.csv not found. Run engine.py first."
        )

    reco = pd.read_csv(PATHS["recommendations"])
    logger.info("Loaded %d recommendations", len(reco))

    # Load comparison for savings context
    comp = pd.DataFrame()
    if PATHS["comparison"].exists():
        comp = pd.read_csv(PATHS["comparison"])

    # Apply filters
    if station:
        reco = reco[reco["id_tienda"] == station]
        logger.info("Filtered to station %s: %d rows", station, len(reco))

    if top_n:
        # Prioritise rows with a recommendation and sort by savings
        reco_valid = reco[reco["recommended_supplier"].notna()]
        if "saving_mxn" in comp.columns and not comp.empty:
            merge_cols = ["id_tienda", "product_type", "order_date"]
            reco_valid = reco_valid.merge(
                comp[merge_cols + ["saving_mxn", "saving_pct",
                                   "baseline_supplier", "baseline_price_per_l",
                                   "baseline_total_mxn"]],
                on=merge_cols, how="left"
            )
            reco_valid = reco_valid.sort_values("saving_mxn", ascending=False)
        reco = reco_valid.head(top_n)
        logger.info("Top %d recommendations selected", len(reco))

    # Merge savings data
    if not comp.empty and "saving_mxn" not in reco.columns:
        merge_cols = ["id_tienda", "product_type", "order_date"]
        reco = reco.merge(
            comp[merge_cols + ["saving_mxn", "saving_pct",
                               "baseline_supplier", "baseline_price_per_l",
                               "baseline_total_mxn"]],
            on=merge_cols, how="left"
        )

    # Generate explanations
    explanations = []
    comp_dict = {}
    if not comp.empty:
        for _, row in comp.iterrows():
            key = (row["id_tienda"], row["product_type"], row["order_date"])
            comp_dict[key] = row

    logger.info("Generating LLM explanations for %d recommendations...", len(reco))

    for i, (_, row) in enumerate(reco.iterrows()):
        if pd.isna(row.get("recommended_supplier")):
            explanations.append("No supplier could be recommended — all candidates excluded by hard constraints.")
            continue

        key = (row["id_tienda"], row["product_type"], row["order_date"])
        comp_row = comp_dict.get(key)

        try:
            prompt = build_prompt(row, comp_row)
            explanation = call_claude(prompt, api_key)
            explanations.append(explanation)
            logger.info("[%d/%d] %s %s %s → explained",
                        i+1, len(reco), row['id_tienda'],
                        row['product_type'], row['order_date'])
        except Exception as e:
            logger.warning("API error for row %d: %s", i, e)
            explanations.append(f"Explanation unavailable: {e}")

        # Rate limit — avoid hitting API limits
        if i < len(reco) - 1:
            time.sleep(0.3)

    reco = reco.copy()
    reco["explanation"] = explanations

    # Save CSV
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    reco.to_csv(PATHS["output_csv"], index=False)
    logger.info("Saved explained recommendations → %s", PATHS["output_csv"])

    # Generate HTML report
    html = generate_html_report(reco)
    with open(PATHS["output_html"], "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Saved HTML report → %s", PATHS["output_html"])

    return reco


def main():
    parser = argparse.ArgumentParser(
        description="AI Hub — LLM Recommendation Explainer"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Explain top N recommendations by savings (default: all)"
    )
    parser.add_argument(
        "--station", default=None,
        help="Explain recommendations for a specific station"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env variable)"
    )
    args = parser.parse_args()

    run_explainer(
        top_n   = args.top,
        station = args.station,
        api_key = args.api_key,
    )


if __name__ == "__main__":
    main()