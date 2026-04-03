"""
standardize.py
--------------
Normalization layer for the AI Hub Procurement Engine.

Designed to work for ANY buyer — not just CapitalGas. Switch buyers
by passing --buyer flag. New buyer = drop 4 CSV files in data/<buyer>/

Reads:
    data/processed/all_suppliers.csv
    data/processed/surcharges.csv
    data/<buyer>/tiendas.csv                  (tiendas_capitalgas.csv for CapitalGas)
    data/<buyer>/cobertura_logistica.csv
    data/<buyer>/parametros_pedido.csv
    data/<buyer>/restriccion_pemex.csv
    data/<buyer>/presupuesto_compra.csv
    data/<buyer>/inventario_inicial.csv

Outputs:
    data/processed/normalized_suppliers.csv

Output adds these columns on top of all_suppliers.csv:
    id_tienda, station_name, ciudad, estado_station, zona, lat, lon
    dist_km, freight_cost_mxn, supplier_available
    moq_litros, vol_max_litros, lead_time_dias, rop_pct_tanque, frecuencia_min
    pct_minimo_pemex, aplica_pemex
    presupuesto_total, reserva_total            <- budget ceiling
    cap_tanque_litros, inv_inicial_litros        <- starting inventory
    available_capacity_litros                    <- space left in tank for new order
    max_order_litros                             <- min(vol_max, available_capacity)
    price_volatility_30d                         <- 30-day rolling price std
    price_freshness_days                         <- days since this price was published
    price_stale                                  <- True if price older than FRESHNESS_THRESHOLD
    surcharge_mxn_per_l
    landed_cost                                  <- price + freight/vol + surcharge

Usage:
    python src/normalization/standardize.py                          # all stations, CapitalGas
    python src/normalization/standardize.py --region monterrey       # MVP scope
    python src/normalization/standardize.py --city "San Luis Potosi" # specific city
    python src/normalization/standardize.py --buyer otro_comprador   # different buyer
    python src/normalization/standardize.py --supplier Valero        # single supplier
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CANACAR official freight rate — MXN per km per truck trip
# Source: CANACAR (Cámara Nacional del Autotransporte de Carga)
# Used by Vector Bulls and standard in Mexican fuel logistics
# Freight cost per liter = (dist_km × CANACAR_RATE) / order_volume_litros
# This means freight per liter DECREASES as order volume increases — correct economics
CANACAR_RATE_MXN_PER_KM = 28.40

# Default order volume for freight calculation when actual order size is unknown
# Rules engine will override this with the actual order quantity
DEFAULT_ORDER_LITROS = 30000

# Price freshness threshold — flag prices older than this many days as stale
# Pemex publishes weekly so 7 days is the maximum acceptable staleness
FRESHNESS_THRESHOLD_DAYS = 7

DATA_PROCESSED = Path("data/processed")
DEFAULT_BUYER  = "capitalgas"

BUYER_FILES = {
    "tiendas":     "tiendas.csv",
    "cobertura":   "cobertura_logistica.csv",
    "parametros":  "parametros_pedido.csv",
    "restriccion": "restriccion_pemex.csv",
    "presupuesto": "presupuesto_compra.csv",
    "inventario":  "inventario_inicial.csv",
}

PATHS = {
    "all_suppliers": DATA_PROCESSED / "all_suppliers.csv",
    "surcharges":    DATA_PROCESSED / "surcharges.csv",
    "output":        DATA_PROCESSED / "normalized_suppliers.csv",
}

# Maps supplier name from all_suppliers.csv to cobertura column prefix
SUPPLIER_PREFIX = {
    "Valero":      "valero",
    "ExxonMobil":  "exxonmobil",
    "Exxon":       "exxonmobil",
    "Marathon":    "marathon",
    "Pemex":       "pemex",
    "PEMEX":       "pemex",
}

MOQ_COL = {
    "Regular": "vol_min_regular_litros",
    "Premium": "vol_min_premium_litros",
    "Diesel":  "vol_min_diesel_litros",
}

LEAD_TIME_COL = {
    "valero":     "lead_time_valero_dias",
    "exxonmobil": "lead_time_exxon_dias",
    "marathon":   "lead_time_marathon_dias",
    "pemex":      "lead_time_pemex_dias",
}

PEMEX_APLICA_COL = {
    "Regular": "aplica_regular",
    "Premium": "aplica_premium",
    "Diesel":  "aplica_diesel",
}

CAP_TANQUE_COL = {
    "Regular": "cap_tanque_regular",
    "Premium": "cap_tanque_premium",
    "Diesel":  "cap_tanque_diesel",
}

INV_INICIAL_COL = {
    "Regular": "inv_regular_litros",
    "Premium": "inv_premium_litros",
    "Diesel":  "inv_diesel_litros",
}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Buyer path resolver
# ---------------------------------------------------------------------------

def get_buyer_paths(buyer: str) -> dict:
    """
    Resolve file paths for a given buyer.

    CapitalGas uses its legacy output folder and filename conventions.
    All other buyers must have files in data/<buyer>/ with standard names.

    Adding a new buyer:
        mkdir data/<buyer_name>/
        # Add these files with standard column schema:
        #   tiendas.csv, cobertura_logistica.csv,
        #   parametros_pedido.csv, restriccion_pemex.csv,
        #   presupuesto_compra.csv, inventario_inicial.csv
    """
    if buyer.lower() == "capitalgas":
        base = Path("data/capitalgas/outputs")
        return {
            "tiendas":     base / "tiendas_capitalgas.csv",
            "cobertura":   base / "cobertura_logistica.csv",
            "parametros":  base / "parametros_pedido.csv",
            "restriccion": base / "restriccion_pemex.csv",
            "presupuesto": base / "presupuesto_compra.csv",
            "inventario":  base / "inventario_inicial.csv",
        }

    base = Path(f"data/{buyer.lower()}")
    if not base.exists():
        raise FileNotFoundError(
            f"Buyer folder not found: {base}\n"
            f"To onboard '{buyer}', create {base}/ and add:\n"
            + "\n".join(f"  {base}/{f}" for f in BUYER_FILES.values())
        )

    paths = {}
    for key, filename in BUYER_FILES.items():
        p = base / filename
        if not p.exists():
            raise FileNotFoundError(
                f"Required file missing for buyer '{buyer}': {p}"
            )
        paths[key] = p
    return paths


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_suppliers(path: Path) -> pd.DataFrame:
    """Load all_suppliers.csv — preserve all columns unchanged."""
    logger.info("Loading supplier prices: %s", path)
    if not path.exists():
        raise FileNotFoundError(
            f"all_suppliers.csv not found: {path}\n"
            "Run src/ingestion/run_ingestion.py first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    logger.info("Loaded %d supplier price rows", len(df))
    return df


def load_surcharges(path: Path) -> pd.DataFrame:
    """Load surcharges.csv — returns empty if file does not exist."""
    if not path.exists():
        logger.info("No surcharges.csv — no active surcharges will be applied")
        return pd.DataFrame(columns=[
            "supplier", "terminal", "product",
            "surcharge_per_l", "effective_from", "effective_to"
        ])
    df = pd.read_csv(path)
    df["effective_from"] = pd.to_datetime(df["effective_from"], errors="coerce")
    df["effective_to"]   = pd.to_datetime(df["effective_to"],   errors="coerce")
    logger.info("Loaded %d surcharge events", len(df))
    return df


def load_buyer_files(buyer_paths: dict) -> dict:
    """Load all buyer station files."""
    result = {}
    for key, path in buyer_paths.items():
        result[key] = pd.read_csv(path)
        logger.info("Loaded %s: %d rows", path.name, len(result[key]))
    return result


# ---------------------------------------------------------------------------
# Price volatility
# ---------------------------------------------------------------------------

def add_price_volatility(suppliers: pd.DataFrame) -> pd.DataFrame:
    """
    Add 30-day rolling price standard deviation per supplier + terminal + product.

    This measures how volatile a supplier's prices are at a given terminal.
    High volatility = higher risk for procurement planning.
    Used by the intelligence layer's risk scorer.
    """
    suppliers = suppliers.sort_values(["supplier", "terminal_id", "product_type", "date"])

    suppliers["price_volatility_30d"] = (
        suppliers
        .groupby(["supplier", "terminal_id", "product_type"])["price_mxn_per_l"]
        .transform(lambda x: x.rolling(window=30, min_periods=2).std())
        .round(6)
    )

    suppliers["price_volatility_30d"] = suppliers["price_volatility_30d"].fillna(0.0)
    logger.info("Added price_volatility_30d column")
    return suppliers


# ---------------------------------------------------------------------------
# Vectorized surcharge lookup (replaces slow row-by-row apply)
# ---------------------------------------------------------------------------

def apply_surcharges_vectorized(
    df: pd.DataFrame,
    surcharges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply active surcharges efficiently.

    For small surcharge lists (typical case — a few events at a time)
    this is fast. For large DataFrames we avoid a full cross-merge
    which would OOM on 39M rows.

    Strategy: iterate over surcharge events (always a small number —
    typically 0-20 at a time) and apply each one with boolean masking.
    This is O(surcharge_events × df_rows) not O(df_rows²).
    """
    df["surcharge_mxn_per_l"] = 0.0

    if surcharges.empty:
        return df

    for _, event in surcharges.iterrows():
        sup  = str(event["supplier"]).lower()
        term = str(event["terminal"]).lower()
        prod = str(event["product"]).lower()
        amt  = float(event["surcharge_per_l"])
        d_from = pd.to_datetime(event["effective_from"])
        d_to   = pd.to_datetime(event["effective_to"])

        if amt <= 0:
            continue

        # Build boolean mask — all vectorized, no apply()
        mask_sup  = df["supplier"].str.lower() == sup
        mask_date = (df["date"] >= d_from) & (df["date"] <= d_to)
        mask_prod = (
            (prod == "all") |
            (df["product_type"].str.lower() == prod)
        )
        mask_term = (
            df["terminal_name"].str.lower().str.contains(term, regex=False, na=False) |
            df["terminal_id"].str.lower().str.contains(term, regex=False, na=False)
        )

        mask = mask_sup & mask_date & mask_prod & mask_term
        df.loc[mask, "surcharge_mxn_per_l"] += amt

        matched = mask.sum()
        if matched > 0:
            logger.info(
                "Surcharge applied: %s %s %s +%.2f MXN/L → %d rows",
                event["supplier"], event["terminal"],
                event["product"], amt, matched
            )

    return df


# ---------------------------------------------------------------------------
# Core join
# ---------------------------------------------------------------------------

def build_normalized(
    suppliers: pd.DataFrame,
    buyer: dict,
    surcharges: pd.DataFrame,
    region_filter: str = None,
    city_filter: str = None,
    supplier_filter: str = None,
) -> pd.DataFrame:
    """
    Core normalization join.

    For every supplier price row, create one row per station that
    supplier can serve. Add freight, order params, budget, inventory,
    surcharge, and landed cost — all vectorized for performance.
    """
    tiendas     = buyer["tiendas"]
    cobertura   = buyer["cobertura"]
    parametros  = buyer["parametros"]
    restriccion = buyer["restriccion"]
    presupuesto = buyer["presupuesto"]
    inventario  = buyer["inventario"]

    # Station filter
    stations = tiendas.copy()
    if region_filter and region_filter.lower() == "monterrey":
        stations = stations[stations["ciudad"] == "Monterrey"]
        logger.info("MVP filter: Monterrey only — %d stations", len(stations))
    elif city_filter:
        stations = stations[stations["ciudad"] == city_filter]
        logger.info("City filter: %s — %d stations", city_filter, len(stations))

    # Supplier filter
    if supplier_filter:
        suppliers = suppliers[
            suppliers["supplier"].str.lower() == supplier_filter.lower()
        ]

    if stations.empty:
        raise ValueError(f"No stations found for filter: {region_filter or city_filter}")

    # Warn if running full dataset — this produces 39M+ rows
    if not region_filter and not city_filter:
        total_est = len(suppliers) * len(stations)
        logger.warning(
            "Running without region filter — estimated %d rows. "
            "This requires ~4GB RAM. Use --region monterrey for MVP scope.",
            total_est
        )

    unique_suppliers = suppliers["supplier"].unique()
    logger.info("Processing %d suppliers across %d stations",
                len(unique_suppliers), len(stations))

    frames = []

    for supplier_name in unique_suppliers:
        sup_df = suppliers[suppliers["supplier"] == supplier_name].copy()
        prefix = SUPPLIER_PREFIX.get(
            supplier_name,
            supplier_name.lower().replace(" ", "")
        )

        disp_col = f"{prefix}_disponible"
        dist_col = f"{prefix}_dist_km"

        if disp_col not in cobertura.columns:
            logger.warning(
                "\n  ⚠  Supplier '%s' has no entry in cobertura_logistica.csv.\n"
                "     Price data is in all_suppliers.csv but EXCLUDED from\n"
                "     normalized_suppliers.csv — no freight distances available.\n"
                "     Add columns %s_disponible, %s_dist_km, %s_terminal, %s_productos\n"
                "     to cobertura_logistica.csv for each station it can serve.\n",
                supplier_name, prefix, prefix, prefix, prefix,
            )
            continue

        # Stations where supplier is available
        cob_avail = cobertura[cobertura[disp_col] == True][
            ["id_tienda", disp_col, dist_col]
        ].copy()
        cob_avail.columns = ["id_tienda", "supplier_available", "dist_km"]
        cob_avail = cob_avail[cob_avail["id_tienda"].isin(stations["id_tienda"])]

        if cob_avail.empty:
            logger.warning("No stations available for '%s' in this region", supplier_name)
            continue

        # Cross join: price rows × available stations (vectorized)
        sup_df["_key"] = 1
        cob_avail["_key"] = 1
        crossed = sup_df.merge(cob_avail, on="_key").drop(columns=["_key"])

        # Add station master info
        crossed = crossed.merge(
            stations[[
                "id_tienda", "nombre", "ciudad", "estado",
                "zona", "latitud", "longitud"
            ]].rename(columns={
                "nombre":   "station_name",
                "estado":   "estado_station",
                "latitud":  "lat",
                "longitud": "lon",
            }),
            on="id_tienda", how="left"
        )

        # Add order parameters
        lead_col = LEAD_TIME_COL.get(prefix)
        param_merge_cols = [
            "id_tienda", "rop_pct_tanque",
            "vol_max_entrega_litros", "frecuencia_min_entre_pedidos"
        ]
        if lead_col and lead_col in parametros.columns:
            param_merge_cols.append(lead_col)

        crossed = crossed.merge(
            parametros[param_merge_cols].rename(columns={
                "vol_max_entrega_litros":       "vol_max_litros",
                "frecuencia_min_entre_pedidos": "frecuencia_min",
                lead_col: "lead_time_dias" if lead_col else "__drop__",
            }),
            on="id_tienda", how="left"
        )
        if "__drop__" in crossed.columns:
            crossed = crossed.drop(columns=["__drop__"])

        # Add MOQ — fully vectorized using melt + merge
        moq_long = parametros[["id_tienda",
            "vol_min_regular_litros",
            "vol_min_premium_litros",
            "vol_min_diesel_litros",
        ]].melt(
            id_vars="id_tienda",
            value_vars=["vol_min_regular_litros","vol_min_premium_litros","vol_min_diesel_litros"],
            var_name="moq_col", value_name="moq_litros"
        )
        moq_long["product_type"] = moq_long["moq_col"].map({
            "vol_min_regular_litros": "Regular",
            "vol_min_premium_litros": "Premium",
            "vol_min_diesel_litros":  "Diesel",
        })
        moq_long = moq_long.drop(columns=["moq_col"])
        crossed = crossed.merge(moq_long, on=["id_tienda", "product_type"], how="left")

        # Add PEMEX constraint — fully vectorized
        restr_cols = ["id_tienda", "pct_minimo_pemex",
                      "aplica_regular", "aplica_premium", "aplica_diesel"]
        crossed = crossed.merge(
            restriccion[restr_cols], on="id_tienda", how="left"
        )
        # Map aplica_pemex per product using numpy select (no apply)
        crossed["aplica_pemex"] = np.select(
            [
                crossed["product_type"] == "Regular",
                crossed["product_type"] == "Premium",
                crossed["product_type"] == "Diesel",
            ],
            [
                crossed["aplica_regular"],
                crossed["aplica_premium"],
                crossed["aplica_diesel"],
            ],
            default=False
        )
        crossed = crossed.drop(
            columns=["aplica_regular", "aplica_premium", "aplica_diesel"],
            errors="ignore"
        )

        # Add budget ceiling — presupuesto has one row per station per month
        # We join on station only (rules engine applies monthly budget logic)
        # Here we add the annual average as a reference
        budget_summary = presupuesto.groupby("id_tienda").agg(
            presupuesto_total=("presupuesto_total", "mean"),
            reserva_total=("reserva_total", "mean"),
        ).reset_index()
        crossed = crossed.merge(budget_summary, on="id_tienda", how="left")

        # Add starting inventory — fully vectorized using numpy select
        inv_cols = [
            "id_tienda",
            "cap_tanque_regular", "cap_tanque_premium", "cap_tanque_diesel",
            "inv_regular_litros", "inv_premium_litros", "inv_diesel_litros",
        ]
        crossed = crossed.merge(
            inventario[[c for c in inv_cols if c in inventario.columns]],
            on="id_tienda", how="left"
        )

        # Map cap_tanque and inv_inicial per product using numpy select
        prod = crossed["product_type"]
        crossed["cap_tanque_litros"] = np.select(
            [prod == "Regular", prod == "Premium", prod == "Diesel"],
            [crossed.get("cap_tanque_regular", 0),
             crossed.get("cap_tanque_premium", 0),
             crossed.get("cap_tanque_diesel",  0)],
            default=0
        )
        crossed["inv_inicial_litros"] = np.select(
            [prod == "Regular", prod == "Premium", prod == "Diesel"],
            [crossed.get("inv_regular_litros", 0),
             crossed.get("inv_premium_litros", 0),
             crossed.get("inv_diesel_litros",  0)],
            default=0
        )

        # Drop individual product inventory cols
        drop_inv = [
            "cap_tanque_regular", "cap_tanque_premium", "cap_tanque_diesel",
            "inv_regular_litros", "inv_premium_litros", "inv_diesel_litros",
        ]
        crossed = crossed.drop(columns=[c for c in drop_inv if c in crossed.columns])

        # Available tank capacity and max order size
        # available_capacity = how much space is left in the tank right now
        # max_order_litros = min(what fits in tank, max tanker size)
        # Rules engine uses max_order_litros to cap order quantity
        crossed["available_capacity_litros"] = (
            crossed["cap_tanque_litros"] - crossed["inv_inicial_litros"]
        ).clip(lower=0)
        crossed["max_order_litros"] = crossed[
            ["vol_max_litros", "available_capacity_litros"]
        ].min(axis=1)

        # Freight cost — CANACAR rate (28.40 MXN/km per truck trip)
        # freight_cost_mxn is the FIXED cost per delivery trip regardless of volume
        # Rules engine divides by actual order_litros to get per-liter cost:
        #   freight_per_l = freight_cost_mxn / order_litros
        # We also store a reference freight_mxn_per_l at DEFAULT_ORDER_LITROS
        # so the normalization output is immediately usable without the rules engine
        crossed["dist_km"]           = crossed["dist_km"].fillna(0.0)
        crossed["freight_cost_mxn"]  = (
            crossed["dist_km"] * CANACAR_RATE_MXN_PER_KM
        ).round(2)
        crossed["freight_mxn_per_l"] = (
            crossed["freight_cost_mxn"] / DEFAULT_ORDER_LITROS
        ).round(6)

        frames.append(crossed)
        logger.info(
            "%-12s → %d rows (%d stations × %d price rows)",
            supplier_name, len(crossed),
            cob_avail["id_tienda"].nunique(), len(sup_df),
        )

    if not frames:
        raise ValueError("No data produced — check filters and buyer files")

    combined = pd.concat(frames, ignore_index=True)

    # Price freshness flag
    # Compares each price row's date against the most recent price date
    # per supplier+terminal+product combination.
    # Pemex publishes weekly — a 7-day-old price is acceptable.
    # Valero/Exxon/Marathon publish daily — anything older than 7 days is stale.
    # Rules engine should deprioritize stale prices when fresher alternatives exist.
    run_date = combined["date"].max()
    combined["price_freshness_days"] = (
        run_date - combined["date"]
    ).dt.days.fillna(0).astype(int)
    combined["price_stale"] = combined["price_freshness_days"] > FRESHNESS_THRESHOLD_DAYS

    stale_count = combined["price_stale"].sum()
    if stale_count > 0:
        logger.warning(
            "%d rows have prices older than %d days — marked price_stale=True",
            stale_count, FRESHNESS_THRESHOLD_DAYS
        )
    else:
        logger.info("All prices are fresh (within %d days)", FRESHNESS_THRESHOLD_DAYS)

    # Apply surcharges — vectorized
    combined = apply_surcharges_vectorized(combined, surcharges)

    # Landed cost
    # Uses freight_mxn_per_l at DEFAULT_ORDER_LITROS as reference
    # Rules engine recalculates with actual order_litros using freight_cost_mxn
    combined["landed_cost"] = (
        combined["price_mxn_per_l"] +
        combined["freight_mxn_per_l"] +
        combined["surcharge_mxn_per_l"]
    ).round(6)

    combined = combined.sort_values(
        ["date", "supplier", "id_tienda", "product_type"]
    ).reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, output_path: Path):
    print()
    print("=" * 70)
    print("  NORMALIZATION SUMMARY")
    print("=" * 70)
    print(f"  {'Supplier':<14} {'Rows':>10}  {'Stations':>10}  {'Avg Landed':>12}")
    print("-" * 70)

    for sup in sorted(df["supplier"].unique()):
        sub = df[df["supplier"] == sup]
        stations = sub["id_tienda"].nunique() if "id_tienda" in sub else 0
        avg_lc = sub["landed_cost"].mean() if "landed_cost" in sub else 0
        print(f"  {sup:<14} {len(sub):>10,}  {stations:>10,}    {avg_lc:>10.4f} MXN/L")

    print("-" * 70)
    print(f"  {'TOTAL':<14} {len(df):>10,}")
    print("=" * 70)

    if not df.empty:
        print()
        print(f"  Date range      : {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Stations        : {df['id_tienda'].nunique():,}")
        print(f"  Products        : {sorted(df['product_type'].unique())}")
        print(f"  Avg landed cost : {df['landed_cost'].mean():.4f} MXN/L")
        if "price_volatility_30d" in df.columns:
            print(f"  Avg price vol   : {df['price_volatility_30d'].mean():.4f} MXN/L (30d std)")
        if "price_stale" in df.columns:
            stale = df["price_stale"].sum()
            pct = stale / len(df) * 100
            print(f"  Stale prices    : {stale:,} rows ({pct:.1f}%) — older than {FRESHNESS_THRESHOLD_DAYS} days")
        if "available_capacity_litros" in df.columns:
            avg_cap = df["available_capacity_litros"].mean()
            print(f"  Avg avail tank  : {avg_cap:,.0f} L")
        if "freight_cost_mxn" in df.columns:
            # Show freight at default volume for reference
            sample = df[df["supplier"] == "Valero"].head(1) if "Valero" in df["supplier"].values else df.head(1)
            if not sample.empty:
                dist = sample["dist_km"].iloc[0]
                fc   = sample["freight_cost_mxn"].iloc[0]
                fpl  = sample["freight_mxn_per_l"].iloc[0]
                print(f"  Freight example : {dist:.1f}km × 28.40 = {fc:.0f} MXN/trip "
                      f"÷ {DEFAULT_ORDER_LITROS:,}L = {fpl:.4f} MXN/L")
        if "surcharge_mxn_per_l" in df.columns:
            active = (df["surcharge_mxn_per_l"] > 0).sum()
            print(f"  Active surcharges: {active:,} rows")
        print(f"  Output file     : {output_path}")

    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize(
    buyer: str = DEFAULT_BUYER,
    region: str = None,
    city: str = None,
    supplier: str = None,
) -> pd.DataFrame:
    """Run the full normalization pipeline."""
    logger.info("Starting normalization — buyer: %s", buyer)

    buyer_paths = get_buyer_paths(buyer)
    suppliers   = load_suppliers(PATHS["all_suppliers"])
    surcharges  = load_surcharges(PATHS["surcharges"])
    buyer_data  = load_buyer_files(buyer_paths)

    # Add price volatility before cross join
    suppliers = add_price_volatility(suppliers)

    df = build_normalized(
        suppliers, buyer_data, surcharges,
        region_filter=region,
        city_filter=city,
        supplier_filter=supplier,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS["output"], index=False)
    logger.info("Saved → %s  (%d rows)", PATHS["output"], len(df))

    print_summary(df, PATHS["output"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="AI Procurement Hub — Normalization Layer"
    )
    parser.add_argument(
        "--buyer", default=DEFAULT_BUYER,
        help="Buyer name (default: capitalgas). New buyer: create data/<buyer>/ with 6 CSV files."
    )
    parser.add_argument(
        "--region", choices=["monterrey"], default=None,
        help="Filter to a region ('monterrey' for MVP)"
    )
    parser.add_argument(
        "--city", default=None,
        help="Filter to a specific city"
    )
    parser.add_argument(
        "--supplier", default=None,
        help="Filter to a specific supplier"
    )
    args = parser.parse_args()

    normalize(
        buyer=args.buyer,
        region=args.region,
        city=args.city,
        supplier=args.supplier,
    )


if __name__ == "__main__":
    main()