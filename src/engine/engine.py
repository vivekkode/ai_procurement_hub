"""
engine.py
---------
Rules Engine for the AI Hub Procurement Engine.

Reads:
    data/processed/normalized_suppliers.csv      <- from normalization layer
    data/capitalgas/outputs/historico_pedidos.csv <- baseline + PEMEX tracking
    data/capitalgas/outputs/minimos_pemex_mensual.csv <- PEMEX monthly minimums
    data/capitalgas/outputs/ventas_diarias_capitalgas.csv <- demand / tank depletion
    data/capitalgas/outputs/presupuesto_compra.csv <- budget ceiling
    data/capitalgas/outputs/inventario_inicial.csv <- starting tank inventory

Outputs:
    data/processed/recommendations.csv   <- ranked supplier per order event
    data/processed/baseline_comparison.csv <- savings vs historico_pedidos

What this engine does:
    1. Simulates order events by detecting when each station hits its
       reorder point using daily sales data + current inventory
    2. For each order event, enforces hard constraints:
          - PEMEX 50% monthly minimum (legal requirement confirmed Mar 13)
          - MOQ and max order volume
          - Budget ceiling
          - Supplier availability
          - Minimum frequency between orders
    3. Ranks all suppliers that pass constraints by:
          landed_cost (60%) + reliability_score (25%) + price_volatility (15%)
    4. Generates a recommendation with full math breakdown (explainability
       requirement from Mar 13 MOM)
    5. Compares every recommendation against the baseline historico_pedidos
       to compute savings

Architecture decision (from our discussion):
    - Hard constraints → binary pass/fail before ranking
    - Soft ranking → weighted score of cost + reliability + volatility
    - Reliability score derived from historico_pedidos delivery history
    - PEMEX constraint checked against real monthly order history
    - Penalty NOT included — no rate in dataset, will add when Roberto provides

Usage:
    # Monterrey MVP — run order simulation for all Monterrey stations
    python src/rules_engine/engine.py --region monterrey

    # Single station
    python src/rules_engine/engine.py --station OXG-XAJI0

    # Specific date range
    python src/rules_engine/engine.py --region monterrey --start 2024-01-01 --end 2024-03-31

    # Show savings vs baseline
    python src/rules_engine/engine.py --region monterrey --compare-baseline
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PROCESSED  = Path("data/processed")
DATA_CAPITALGAS = Path("data/capitalgas/outputs")

PATHS = {
    "normalized":   DATA_PROCESSED  / "normalized_suppliers.csv",
    "historico":    DATA_CAPITALGAS  / "historico_pedidos.csv",
    "minimos":      DATA_CAPITALGAS  / "minimos_pemex_mensual.csv",
    "ventas":       DATA_CAPITALGAS  / "ventas_diarias_capitalgas.csv",
    "presupuesto":  DATA_CAPITALGAS  / "presupuesto_compra.csv",
    "inventario":   DATA_CAPITALGAS  / "inventario_inicial.csv",
    "output_reco":  DATA_PROCESSED   / "recommendations.csv",
    "output_comp":  DATA_PROCESSED   / "baseline_comparison.csv",
}

# Ranking weights — from our architectural decision
WEIGHT_COST        = 0.60
WEIGHT_RELIABILITY = 0.25
WEIGHT_VOLATILITY  = 0.15

# CANACAR freight rate — must match normalization layer
CANACAR_RATE_MXN_PER_KM = 28.40

# Supplier name normalisation — historico uses different casing
SUPPLIER_NORM = {
    "pemex":      "Pemex",
    "PEMEX":      "Pemex",
    "valero":     "Valero",
    "Valero":     "Valero",
    "exxonmobil": "ExxonMobil",
    "ExxonMobil": "ExxonMobil",
    "Exxon":      "ExxonMobil",
    "marathon":   "Marathon",
    "Marathon":   "Marathon",
}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_normalized(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"normalized_suppliers.csv not found: {path}\n"
            "Run src/normalization/standardize.py --region monterrey first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded normalized_suppliers.csv: %d rows", len(df))
    return df


def load_historico(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fecha_pedido"]  = pd.to_datetime(df["fecha_pedido"])
    df["fecha_entrega"] = pd.to_datetime(df["fecha_entrega"])
    df["lead_time_actual"] = (df["fecha_entrega"] - df["fecha_pedido"]).dt.days
    # Normalize supplier names
    df["proveedor"] = df["proveedor"].map(
        lambda x: SUPPLIER_NORM.get(x, x)
    )
    logger.info("Loaded historico_pedidos.csv: %d orders", len(df))
    return df


def load_ventas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    logger.info("Loaded ventas_diarias: %d rows", len(df))
    return df


def load_minimos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded minimos_pemex: %d rows", len(df))
    return df


def load_presupuesto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded presupuesto: %d rows", len(df))
    return df


def load_inventario(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded inventario_inicial: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Reliability score
# ---------------------------------------------------------------------------

def compute_reliability_scores(historico: pd.DataFrame) -> pd.DataFrame:
    """
    Derive supplier reliability score from actual late delivery rate.

    Score = 1 - (late_delivery_rate)
    where late_delivery_rate = orders delivered AFTER promised lead time / total orders

    Promised lead time per supplier (from parametros_pedido industry knowledge):
        PEMEX:      2 days
        Marathon:   3 days
        Valero:     3 days
        ExxonMobil: 3 days

    This is more meaningful than normalising avg lead time 0-1 because:
    - Avg lead 3.0 days vs 2.2 days sounds bad but is actually fine
    - A score of 0.000 implied "never delivers" which is wrong
    - Late delivery rate directly measures the risk of a delayed order

    Results from historico_pedidos data:
        PEMEX:      avg 2.2 days — most reliable
        Marathon:   avg 2.4 days
        Valero:     avg 2.4 days
        ExxonMobil: avg 3.0 days — slowest but not unreliable
    """
    # Promised lead times per supplier (days)
    promised_lead = {
        "Pemex":      2,
        "Marathon":   3,
        "Valero":     3,
        "ExxonMobil": 3,
    }

    results = []
    for supplier, grp in historico.groupby("proveedor"):
        promised = promised_lead.get(supplier, 3)
        total    = len(grp)
        late     = (grp["lead_time_actual"] > promised).sum()
        late_pct = late / total if total > 0 else 0
        score    = round(1 - late_pct, 4)

        results.append({
            "supplier":          supplier,
            "avg_lead_days":     round(grp["lead_time_actual"].mean(), 2),
            "promised_lead_days":promised,
            "late_deliveries":   int(late),
            "total_orders":      total,
            "late_rate_pct":     round(late_pct * 100, 1),
            "reliability_score": score,
        })

    agg = pd.DataFrame(results)

    logger.info("Reliability scores (late delivery rate method):")
    for _, row in agg.iterrows():
        logger.info(
            "  %-12s avg=%.1fd  promised=%.0fd  late=%d/%d (%.1f%%)  reliability=%.3f",
            row["supplier"], row["avg_lead_days"], row["promised_lead_days"],
            row["late_deliveries"], row["total_orders"],
            row["late_rate_pct"], row["reliability_score"]
        )

    return agg


# ---------------------------------------------------------------------------
# PEMEX constraint checker
# ---------------------------------------------------------------------------

def check_pemex_constraint(
    station_id: str,
    product: str,
    order_date: pd.Timestamp,
    order_litros: float,
    historico: pd.DataFrame,
    minimos: pd.DataFrame,
) -> dict:
    """
    Check PEMEX 50% legal constraint for a station/product/month.

    Returns dict with:
        pemex_required: bool — must this order go to PEMEX?
        pemex_bought_so_far: float — litros bought from PEMEX this month
        pemex_minimum: float — required monthly minimum
        pemex_remaining: float — litros still needed from PEMEX
        pemex_compliance_pct: float — current compliance percentage
    """
    month_str = order_date.strftime("%Y-%m")
    product_lower = product.lower()

    # Get monthly minimum for this station + month
    min_col = f"min_pemex_{product_lower}"
    station_min = minimos[
        (minimos["id_tienda"] == station_id) &
        (minimos["anio_mes"] == month_str)
    ]

    if station_min.empty or min_col not in station_min.columns:
        return {
            "pemex_required": False,
            "pemex_bought_so_far": 0,
            "pemex_minimum": 0,
            "pemex_remaining": 0,
            "pemex_compliance_pct": 100.0,
        }

    pemex_minimum = float(station_min[min_col].iloc[0])

    # How much PEMEX has already been ordered this month
    month_start = order_date.replace(day=1)
    month_orders = historico[
        (historico["id_tienda"] == station_id) &
        (historico["proveedor"] == "Pemex") &
        (historico["producto"] == product_lower) &
        (historico["fecha_pedido"] >= month_start) &
        (historico["fecha_pedido"] < order_date)
    ]

    pemex_bought = float(month_orders["litros_pedidos"].sum())
    pemex_remaining = max(0, pemex_minimum - pemex_bought)
    compliance_pct = (pemex_bought / pemex_minimum * 100) if pemex_minimum > 0 else 100.0

    return {
        "pemex_required": pemex_remaining > 0,
        "pemex_bought_so_far": pemex_bought,
        "pemex_minimum": pemex_minimum,
        "pemex_remaining": pemex_remaining,
        "pemex_compliance_pct": round(compliance_pct, 1),
    }


# ---------------------------------------------------------------------------
# Budget checker
# ---------------------------------------------------------------------------

def check_budget(
    station_id: str,
    order_date: pd.Timestamp,
    order_cost_mxn: float,
    historico: pd.DataFrame,
    presupuesto: pd.DataFrame,
) -> dict:
    """
    Check if this order fits within the monthly budget ceiling.
    """
    month_str = order_date.strftime("%Y-%m")

    budget_row = presupuesto[
        (presupuesto["id_tienda"] == station_id) &
        (presupuesto["anio_mes"] == month_str)
    ]

    if budget_row.empty:
        return {"within_budget": True, "budget_remaining": float("inf"), "budget_total": 0}

    budget_total    = float(budget_row["presupuesto_total"].iloc[0])
    reserva         = float(budget_row["reserva_total"].iloc[0])
    usable_budget   = budget_total - reserva

    # What has already been spent this month
    month_start = order_date.replace(day=1)
    spent_so_far = float(
        historico[
            (historico["id_tienda"] == station_id) &
            (historico["fecha_pedido"] >= month_start) &
            (historico["fecha_pedido"] < order_date)
        ]["importe_total"].sum()
    )

    budget_remaining = usable_budget - spent_so_far
    within_budget    = order_cost_mxn <= budget_remaining

    return {
        "within_budget":    within_budget,
        "budget_remaining": round(budget_remaining, 2),
        "budget_total":     round(usable_budget, 2),
        "budget_spent":     round(spent_so_far, 2),
    }


# ---------------------------------------------------------------------------
# Reorder trigger detector
# ---------------------------------------------------------------------------

def detect_reorder_events(
    stations: list,
    ventas: pd.DataFrame,
    inventario: pd.DataFrame,
    normalized: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Simulate daily tank depletion and detect when each station hits
    its reorder point (rop_pct_tanque).

    For each station × product:
        Starting inventory → subtract daily sales → when pct < rop_pct → order event

    Returns DataFrame of order events with:
        id_tienda, product_type, order_date, current_inv_litros,
        cap_tanque_litros, rop_pct_tanque, order_quantity_litros
    """
    events = []

    # Get reorder parameters from normalized (take first row per station)
    # Handle both old normalized files (without available_capacity_litros)
    # and new ones (with it)
    base_cols = ["id_tienda", "product_type", "rop_pct_tanque",
                 "cap_tanque_litros", "moq_litros", "frecuencia_min"]
    optional_cols = ["available_capacity_litros", "max_order_litros",
                     "vol_max_litros"]

    select_cols = base_cols + [c for c in optional_cols if c in normalized.columns]

    params = normalized.drop_duplicates(
        subset=["id_tienda", "product_type"]
    )[select_cols].copy()

    # Compute available_capacity if not present
    if "available_capacity_litros" not in params.columns:
        params["available_capacity_litros"] = params["cap_tanque_litros"]
    if "max_order_litros" not in params.columns:
        params["max_order_litros"] = params.get(
            "vol_max_litros", params["cap_tanque_litros"]
        )

    inv_df = inventario.set_index("id_tienda")

    product_to_ventas_col = {
        "Regular": "litros_regular",
        "Premium": "litros_premium",
        "Diesel":  "litros_diesel",
    }
    product_to_inv_col = {
        "Regular": "inv_regular_litros",
        "Premium": "inv_premium_litros",
        "Diesel":  "inv_diesel_litros",
    }

    for station_id in stations:
        if station_id not in inv_df.index:
            continue

        station_ventas = ventas[ventas["id_tienda"] == station_id].set_index("fecha")
        station_params = params[params["id_tienda"] == station_id]

        for _, param_row in station_params.iterrows():
            product    = param_row["product_type"]
            rop_pct    = param_row["rop_pct_tanque"]
            cap        = param_row["cap_tanque_litros"]
            moq        = param_row["moq_litros"] or 10000
            max_order  = param_row["max_order_litros"] or 30000
            freq_min   = param_row["frecuencia_min"] or 3

            if cap <= 0 or pd.isna(cap):
                continue

            inv_col    = product_to_inv_col.get(product)
            ventas_col = product_to_ventas_col.get(product)
            if not inv_col or not ventas_col:
                continue

            # Starting inventory
            try:
                current_inv = float(inv_df.loc[station_id, inv_col])
            except (KeyError, TypeError):
                continue

            last_order_date = start_date - timedelta(days=freq_min + 1)

            # Simulate day by day
            date_range = pd.date_range(start_date, end_date, freq="D")
            for day in date_range:
                # Subtract daily sales
                if day in station_ventas.index:
                    daily_sale = float(station_ventas.loc[day, ventas_col])
                    current_inv = max(0.0, current_inv - daily_sale)

                # Check reorder point
                pct_full = current_inv / cap
                days_since_last = (day - last_order_date).days

                if pct_full <= rop_pct and days_since_last >= freq_min:
                    # Order enough to fill tank to ~90% (standard practice)
                    order_qty = min(
                        max(moq, cap * 0.9 - current_inv),
                        max_order
                    )

                    # Calculate how many days until tank runs empty
                    # at current daily sales rate — this is the delivery deadline
                    avg_daily = float(
                        station_ventas[ventas_col].mean()
                    ) if ventas_col in station_ventas.columns else 0

                    if avg_daily > 0:
                        days_until_empty = int(current_inv / avg_daily)
                    else:
                        days_until_empty = 999

                    # The deadline must be at least 3 days — the max supplier
                    # lead time in our dataset. If the tank runs out in fewer
                    # days than any supplier can deliver, we still need to order
                    # from the fastest available supplier rather than skip
                    days_until_empty = max(3, days_until_empty)

                    events.append({
                        "id_tienda":             station_id,
                        "product_type":          product,
                        "order_date":            day,
                        "current_inv_litros":    round(current_inv, 2),
                        "cap_tanque_litros":     cap,
                        "pct_full_at_trigger":   round(pct_full * 100, 1),
                        "rop_pct":               round(rop_pct * 100, 1),
                        "order_quantity_litros":  round(order_qty, 0),
                        "days_until_empty":       days_until_empty,
                        "required_by_date":       (day + timedelta(days=days_until_empty)).strftime("%Y-%m-%d"),
                    })

                    # Refill
                    current_inv += order_qty
                    last_order_date = day

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df["order_date"] = pd.to_datetime(events_df["order_date"])
    logger.info("Detected %d order events across %d stations", len(events_df), len(stations))
    return events_df


# ---------------------------------------------------------------------------
# Core ranking engine
# ---------------------------------------------------------------------------

def rank_suppliers(
    station_id: str,
    product: str,
    order_date: pd.Timestamp,
    order_qty: float,
    normalized: pd.DataFrame,
    reliability: pd.DataFrame,
    historico: pd.DataFrame,
    minimos: pd.DataFrame,
    presupuesto: pd.DataFrame,
    normalized_idx: dict = None,
    historico_idx: dict = None,
    minimos_idx: dict = None,
    presupuesto_idx: dict = None,
    deadline_days: int = 999,
) -> dict:
    """
    For one order event, rank all available suppliers that can deliver
    within the deadline and return the top recommendation with full
    math breakdown.

    deadline_days: how many days before the tank runs empty.
                   Suppliers whose lead_time_dias > deadline_days are
                   excluded — they cannot deliver in time.
                   This satisfies the brief requirement: "rank vendors
                   within specific deadlines."
    """
    # Use pre-built index for O(1) lookup — falls back to full scan if not provided
    if normalized_idx is not None:
        base_candidates = normalized_idx.get((station_id, product), pd.DataFrame())
    else:
        base_candidates = normalized[
            (normalized["id_tienda"]    == station_id) &
            (normalized["product_type"] == product) &
            (normalized["supplier_available"] == True)
        ].copy()

    if base_candidates.empty:
        return None

    candidates = base_candidates[
        base_candidates["date"] <= order_date
    ].copy()

    # Fallback: use earliest available if no prices before order_date
    if candidates.empty:
        candidates = base_candidates.copy()

    # Filter out stale prices if fresh ones exist
    if "price_stale" in candidates.columns:
        fresh = candidates[candidates["price_stale"] == False]
        if not fresh.empty:
            candidates = fresh

    # Take most recent price per supplier
    candidates = candidates.sort_values("date", ascending=False)
    candidates = candidates.drop_duplicates(subset=["supplier"])

    # Ensure all required columns exist with safe defaults
    # This handles normalized files generated by older versions of standardize.py
    if "freight_cost_mxn" not in candidates.columns:
        # Fallback: compute from freight_mxn_per_l at 30000L default
        candidates["freight_cost_mxn"] = candidates.get(
            "freight_mxn_per_l", pd.Series(0.0, index=candidates.index)
        ) * 30000
    if "surcharge_mxn_per_l" not in candidates.columns:
        candidates["surcharge_mxn_per_l"] = 0.0
    if "price_volatility_30d" not in candidates.columns:
        candidates["price_volatility_30d"] = 0.0

    # Recalculate freight with actual order quantity
    candidates["freight_mxn_per_l_actual"] = (
        candidates["freight_cost_mxn"] / order_qty
    ).round(6)

    # True landed cost at actual order quantity
    candidates["true_landed_cost"] = (
        candidates["price_mxn_per_l"] +
        candidates["freight_mxn_per_l_actual"] +
        candidates["surcharge_mxn_per_l"]
    ).round(6)

    # Total order cost
    candidates["total_order_cost"] = (
        candidates["true_landed_cost"] * order_qty
    ).round(2)

    # Add reliability score
    candidates = candidates.merge(
        reliability[["supplier", "reliability_score", "avg_lead_days"]],
        on="supplier", how="left"
    )
    candidates["reliability_score"] = candidates["reliability_score"].fillna(0.5)

    # --- PEMEX constraint ---
    hist_station = historico_idx.get(station_id, historico) if historico_idx else historico
    min_station  = minimos_idx.get(station_id, minimos) if minimos_idx else minimos
    pemex_status = check_pemex_constraint(
        station_id, product, order_date, order_qty,
        hist_station, min_station
    )

    # --- Hard constraint filtering ---
    passed    = []
    excluded  = []

    for _, row in candidates.iterrows():
        reasons = []

        # Deadline constraint — can supplier deliver before tank runs empty?
        # lead_time_dias from normalized, compared against days_until_empty
        supplier_lead = row.get("lead_time_dias") or row.get("mb_avg_delivery_days") or 3
        if pd.notna(supplier_lead) and float(supplier_lead) > deadline_days:
            reasons.append(
                f"cannot meet deadline — lead time {supplier_lead:.0f}d > {deadline_days}d until empty"
            )

        # MOQ check
        if order_qty < (row.get("moq_litros") or 0):
            reasons.append(f"below MOQ ({row.get('moq_litros',0):,.0f}L)")

        # Budget check
        pres_station = presupuesto_idx.get(station_id, presupuesto) if presupuesto_idx else presupuesto
        budget_check = check_budget(
            station_id, order_date,
            row["total_order_cost"],
            hist_station, pres_station
        )
        if not budget_check["within_budget"]:
            reasons.append(
                f"exceeds budget (remaining: {budget_check['budget_remaining']:,.0f} MXN)"
            )

        # Price guardrail
        if row["price_mxn_per_l"] < 15 or row["price_mxn_per_l"] > 35:
            reasons.append("price outside guardrail (15-35 MXN/L)")

        if reasons:
            excluded.append({
                "supplier": row["supplier"],
                "reason":   " | ".join(reasons),
            })
        else:
            passed.append(row)

    if not passed:
        return {
            "id_tienda":             station_id,
            "product_type":          product,
            "order_date":            order_date,
            "order_qty_litros":      order_qty,
            "recommended_supplier":  None,
            "reason":                "All suppliers excluded by hard constraints",
            "pemex_status":          pemex_status,
            "excluded_suppliers":    excluded,
            "all_ranked":            [],
            "math_breakdown":        {},
        }

    passed_df = pd.DataFrame(passed)

    # --- PEMEX forced order ---
    pemex_forced = False
    if pemex_status["pemex_required"]:
        pemex_available = "Pemex" in passed_df["supplier"].values
        if pemex_available:
            pemex_forced = True
            # Force PEMEX to top — move to front
            pemex_row = passed_df[passed_df["supplier"] == "Pemex"]
            others    = passed_df[passed_df["supplier"] != "Pemex"]
            passed_df = pd.concat([pemex_row, others], ignore_index=True)
        else:
            logger.warning(
                "PEMEX required for %s %s on %s but PEMEX not available",
                station_id, product, order_date.date()
            )

    # --- Soft ranking (only when PEMEX not forced) ---
    if not pemex_forced:
        # Normalize cost score (lower cost = higher score)
        cost_min = passed_df["true_landed_cost"].min()
        cost_max = passed_df["true_landed_cost"].max()
        cost_range = cost_max - cost_min if cost_max > cost_min else 1

        passed_df["cost_score"] = (
            1 - (passed_df["true_landed_cost"] - cost_min) / cost_range
        )

        # Normalize volatility score (lower volatility = higher score)
        vol_min = passed_df["price_volatility_30d"].min()
        vol_max = passed_df["price_volatility_30d"].max()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1

        passed_df["volatility_score"] = (
            1 - (passed_df["price_volatility_30d"] - vol_min) / (vol_range + 1e-9)
        )

        # Weighted final score
        passed_df["final_score"] = (
            WEIGHT_COST        * passed_df["cost_score"] +
            WEIGHT_RELIABILITY * passed_df["reliability_score"] +
            WEIGHT_VOLATILITY  * passed_df["volatility_score"]
        ).round(4)

        passed_df = passed_df.sort_values("final_score", ascending=False)
    else:
        passed_df["cost_score"]       = None
        passed_df["volatility_score"] = None
        passed_df["final_score"]      = None

    # --- Build recommendation ---
    top = passed_df.iloc[0]
    rank_2 = passed_df.iloc[1] if len(passed_df) > 1 else None

    math_breakdown = {
        "base_price_mxn_per_l":    round(float(top["price_mxn_per_l"]), 4),
        "dist_km":                 round(float(top["dist_km"]), 1),
        "freight_cost_mxn":        round(float(top["freight_cost_mxn"]), 2),
        "freight_mxn_per_l":       round(float(top["freight_mxn_per_l_actual"]), 6),
        "surcharge_mxn_per_l":     round(float(top["surcharge_mxn_per_l"]), 4),
        "true_landed_cost":        round(float(top["true_landed_cost"]), 4),
        "order_qty_litros":        int(order_qty),
        "total_order_cost_mxn":    round(float(top["total_order_cost"]), 2),
        "reliability_score":       round(float(top["reliability_score"]), 3),
        "avg_delivery_days":       round(float(top.get("avg_lead_days", 0)), 1),
        "price_volatility_30d":    round(float(top["price_volatility_30d"]), 4),
        "cost_score":              round(float(top["cost_score"]), 3) if top["cost_score"] is not None else None,
        "final_score":             round(float(top["final_score"]), 3) if top["final_score"] is not None else None,
        "pemex_forced":            pemex_forced,
        "pemex_compliance_pct":    pemex_status["pemex_compliance_pct"],
        "pemex_minimum_litros":    pemex_status["pemex_minimum"],
        "pemex_bought_this_month": pemex_status["pemex_bought_so_far"],
        "deadline_days":           deadline_days,
        "required_by_date":        (order_date + timedelta(days=deadline_days)).strftime("%Y-%m-%d"),
    }

    all_ranked = []
    for rank_i, (_, r) in enumerate(passed_df.iterrows(), 1):
        all_ranked.append({
            "rank":            rank_i,
            "supplier":        r["supplier"],
            "invoice_mxn_per_l": round(float(r["price_mxn_per_l"]), 4),
            "freight_mxn_per_l": round(float(r["freight_mxn_per_l_actual"]), 4),
            "surcharge_mxn_per_l": round(float(r["surcharge_mxn_per_l"]), 4),
            "landed_cost":     round(float(r["true_landed_cost"]), 4),
            "total_cost_mxn":  round(float(r["total_order_cost"]), 2),
            "reliability":     round(float(r["reliability_score"]), 3),
            "final_score":     round(float(r["final_score"]), 3) if r["final_score"] is not None else None,
            "dist_km":         round(float(r["dist_km"]), 1),
        })

    savings_vs_rank2 = (
        (rank_2["total_order_cost"] - top["total_order_cost"])
        if rank_2 is not None else 0
    )

    return {
        "id_tienda":            station_id,
        "product_type":         product,
        "order_date":           order_date,
        "order_qty_litros":     int(order_qty),
        "recommended_supplier": top["supplier"],
        "true_landed_cost":     round(float(top["true_landed_cost"]), 4),
        "total_order_cost_mxn": round(float(top["total_order_cost"]), 2),
        "pemex_forced":         pemex_forced,
        "pemex_compliance_pct": pemex_status["pemex_compliance_pct"],
        "savings_vs_rank2_mxn": round(float(savings_vs_rank2), 2),
        "math_breakdown":       math_breakdown,
        "all_ranked":           all_ranked,
        "excluded_suppliers":   excluded,
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    recommendations: pd.DataFrame,
    historico: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare our recommendations against what historico_pedidos actually ordered.

    For each order event we detect, find the matching baseline order
    (same station, product, same approximate date) and compute the
    savings: baseline_cost - our_cost.
    """
    hist = historico.copy()
    hist["anio_mes"] = hist["fecha_pedido"].dt.to_period("M").astype(str)
    hist["producto_norm"] = hist["producto"].str.title()

    comparisons = []

    for _, rec in recommendations.iterrows():
        # Find baseline orders for same station + product + month
        month_str = pd.to_datetime(rec["order_date"]).strftime("%Y-%m")
        baseline_match = hist[
            (hist["id_tienda"]    == rec["id_tienda"]) &
            (hist["producto_norm"] == rec["product_type"]) &
            (hist["anio_mes"]     == month_str)
        ]

        if baseline_match.empty:
            continue

        baseline_cost_per_l = float(baseline_match["precio_unitario"].mean())
        baseline_qty        = float(baseline_match["litros_pedidos"].mean())
        baseline_total      = baseline_cost_per_l * rec["order_qty_litros"]

        our_total = rec["total_order_cost_mxn"]
        saving    = baseline_total - our_total

        comparisons.append({
            "id_tienda":              rec["id_tienda"],
            "product_type":           rec["product_type"],
            "order_date":             rec["order_date"],
            "our_supplier":           rec["recommended_supplier"],
            "our_landed_cost":        rec["true_landed_cost"],
            "our_total_mxn":          round(our_total, 2),
            "baseline_supplier":      baseline_match["proveedor"].mode()[0],
            "baseline_price_per_l":   round(baseline_cost_per_l, 4),
            "baseline_total_mxn":     round(baseline_total, 2),
            "saving_mxn":             round(saving, 2),
            "saving_pct":             round(saving / baseline_total * 100, 2) if baseline_total > 0 else 0,
        })

    return pd.DataFrame(comparisons)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(recommendations: pd.DataFrame, comparison: pd.DataFrame):
    print()
    print("=" * 70)
    print("  RULES ENGINE SUMMARY")
    print("=" * 70)

    if recommendations.empty:
        print("  No recommendations generated.")
        print("=" * 70)
        return

    total_orders = len(recommendations)
    pemex_forced = recommendations["pemex_forced"].sum()
    total_cost   = recommendations["total_order_cost_mxn"].sum()

    print(f"  Total order events    : {total_orders:,}")
    print(f"  PEMEX forced orders   : {pemex_forced:,} ({pemex_forced/total_orders*100:.1f}%)")
    print(f"  Total recommended cost: MXN {total_cost:,.0f}")
    print()

    # Supplier distribution
    print(f"  {'Supplier':<14} {'Orders':>8}  {'Share':>7}  {'Avg Landed':>12}")
    print("-" * 70)
    for sup, grp in recommendations.groupby("recommended_supplier"):
        avg_lc = grp["true_landed_cost"].mean()
        share  = len(grp) / total_orders * 100
        print(f"  {sup:<14} {len(grp):>8,}  {share:>6.1f}%  {avg_lc:>12.4f} MXN/L")

    print("-" * 70)

    if not comparison.empty:
        total_savings = comparison["saving_mxn"].sum()
        avg_saving_pct = comparison["saving_pct"].mean()
        print()
        print(f"  SAVINGS VS BASELINE")
        print(f"  Total savings         : MXN {total_savings:,.0f}")
        print(f"  Avg saving per order  : {avg_saving_pct:.2f}%")
        print(f"  Baseline total (sample): MXN {comparison['baseline_total_mxn'].sum():,.0f}")
        print(f"  Our total (sample)    : MXN {comparison['our_total_mxn'].sum():,.0f}")

    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_engine(
    region: str = None,
    station: str = None,
    start_date: str = None,
    end_date: str = None,
    compare_baseline: bool = False,
) -> pd.DataFrame:
    """Run the full rules engine."""

    logger.info("Starting rules engine")

    # Load all inputs
    normalized   = load_normalized(PATHS["normalized"])
    historico    = load_historico(PATHS["historico"])
    ventas       = load_ventas(PATHS["ventas"])
    minimos      = load_minimos(PATHS["minimos"])
    presupuesto  = load_presupuesto(PATHS["presupuesto"])
    inventario   = load_inventario(PATHS["inventario"])

    # Compute reliability scores from historical delivery data
    reliability = compute_reliability_scores(historico)

    # Determine station list
    all_stations = normalized["id_tienda"].unique().tolist()
    if station:
        all_stations = [station]
        logger.info("Single station mode: %s", station)
    elif region and region.lower() == "monterrey":
        monterrey_stations = normalized[
            normalized["ciudad"] == "Monterrey"
        ]["id_tienda"].unique().tolist()
        all_stations = monterrey_stations
        logger.info("Monterrey region: %d stations", len(all_stations))

    # Date range
    data_start = normalized["date"].min()
    data_end   = normalized["date"].max()
    start = pd.to_datetime(start_date) if start_date else data_start
    end   = pd.to_datetime(end_date)   if end_date   else data_end

    logger.info("Simulation period: %s to %s", start.date(), end.date())

    # Detect reorder events
    events = detect_reorder_events(
        all_stations, ventas, inventario,
        normalized, start, end
    )

    if events.empty:
        logger.warning("No reorder events detected — check station list and date range")
        return pd.DataFrame()

    # PRE-INDEX normalized by (station, product) for O(1) lookups
    logger.info("Pre-indexing data for fast lookups...")
    normalized_index = {}
    for (station_id, product), grp in normalized[
        normalized["supplier_available"] == True
    ].groupby(["id_tienda", "product_type"]):
        normalized_index[(station_id, product)] = grp

    # Pre-index historico by station for fast PEMEX and budget checks
    historico_by_station = {
        sid: grp for sid, grp in historico.groupby("id_tienda")
    }

    # Pre-index minimos by station
    minimos_by_station = {
        sid: grp for sid, grp in minimos.groupby("id_tienda")
    }

    # Pre-index presupuesto by station
    presupuesto_by_station = {
        sid: grp for sid, grp in presupuesto.groupby("id_tienda")
    }

    logger.info(
        "Index built: %d station-product combos, %d stations in historico",
        len(normalized_index), len(historico_by_station)
    )

    # Generate recommendations
    logger.info("Generating recommendations for %d order events...", len(events))
    results = []

    for _, event in events.iterrows():
        rec = rank_suppliers(
            station_id         = event["id_tienda"],
            product            = event["product_type"],
            order_date         = event["order_date"],
            order_qty          = event["order_quantity_litros"],
            deadline_days      = int(event.get("days_until_empty", 999)),
            normalized         = normalized,
            normalized_idx     = normalized_index,
            reliability        = reliability,
            historico          = historico,
            historico_idx      = historico_by_station,
            minimos            = minimos,
            minimos_idx        = minimos_by_station,
            presupuesto        = presupuesto,
            presupuesto_idx    = presupuesto_by_station,
        )
        if rec:
            # Flatten math_breakdown into top-level for CSV
            flat = {k: v for k, v in rec.items()
                    if k not in ("math_breakdown", "all_ranked", "excluded_suppliers")}
            flat.update({
                f"mb_{k}": v
                for k, v in rec["math_breakdown"].items()
            })
            results.append(flat)

    recommendations = pd.DataFrame(results)

    # Save recommendations
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(PATHS["output_reco"], index=False)
    logger.info("Saved recommendations → %s (%d rows)", PATHS["output_reco"], len(recommendations))

    # Generate Draft Purchase Orders — clean human-readable document
    # This is what gets sent to the procurement manager for approval
    po_path = DATA_PROCESSED / "draft_purchase_orders.csv"
    po_rows = []
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        po_rows.append({
            "po_number":            f"PO-{i:06d}",
            "status":               "PENDING APPROVAL",
            "station_id":           rec["id_tienda"],
            "product":              rec["product_type"],
            "order_date":           rec["order_date"],
            "recommended_supplier": rec["recommended_supplier"],
            "order_qty_litros":     int(rec["order_qty_litros"]),
            "invoice_price_mxn_l":  rec.get("mb_base_price_mxn_per_l", ""),
            "freight_mxn_l":        rec.get("mb_freight_mxn_per_l", ""),
            "surcharge_mxn_l":      rec.get("mb_surcharge_mxn_per_l", 0),
            "landed_cost_mxn_l":    rec["true_landed_cost"],
            "total_cost_mxn":       rec["total_order_cost_mxn"],
            "pemex_required":       "YES" if rec["pemex_forced"] else "NO",
            "pemex_compliance_pct": rec["pemex_compliance_pct"],
            "est_delivery_days":    rec.get("mb_avg_delivery_days", ""),
            "supplier_reliability": f"{rec.get('mb_reliability_score', 0)*100:.1f}%",
            "savings_vs_next_mxn":  rec["savings_vs_rank2_mxn"],
            "math_check":           f"{rec.get('mb_base_price_mxn_per_l',0):.4f} + {rec.get('mb_freight_mxn_per_l',0):.4f} + {rec.get('mb_surcharge_mxn_per_l',0):.4f} = {rec['true_landed_cost']:.4f} MXN/L",
        })

    po_df = pd.DataFrame(po_rows)
    po_df.to_csv(po_path, index=False)
    logger.info("Draft POs → %s (%d orders pending approval)", po_path, len(po_df))

    # Baseline comparison
    comparison = pd.DataFrame()
    if compare_baseline and not recommendations.empty:
        logger.info("Computing savings vs baseline...")
        comparison = compare_to_baseline(recommendations, historico)
        comparison.to_csv(PATHS["output_comp"], index=False)
        logger.info("Saved comparison → %s (%d rows)", PATHS["output_comp"], len(comparison))

    print_summary(recommendations, comparison)
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="AI Procurement Hub — Rules Engine"
    )
    parser.add_argument(
        "--region", choices=["monterrey"], default=None,
        help="Filter to a region (monterrey for MVP)"
    )
    parser.add_argument(
        "--station", default=None,
        help="Run for a single station (e.g. OXG-XAJI0)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: earliest in normalized data)"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: latest in normalized data)"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Compare recommendations against historico_pedidos baseline"
    )
    args = parser.parse_args()

    run_engine(
        region           = args.region,
        station          = args.station,
        start_date       = args.start,
        end_date         = args.end,
        compare_baseline = args.compare_baseline,
    )


if __name__ == "__main__":
    main()