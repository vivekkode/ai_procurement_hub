"""
app.py — AI Hub Procurement Engine
Team GRAV · Globant · University at Buffalo

Run:  streamlit run app.py
Deps: pip install streamlit pandas plotly requests python-dotenv
"""

import os, sys, time, subprocess, shutil
from pathlib import Path
from datetime import date
from collections import Counter

import pandas as pd
import streamlit as st

# ── env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env = Path(".env")
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── paths ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
PROCESSED  = ROOT / "data" / "processed"
CAPITALGAS = ROOT / "data" / "CapitalGas" / "outputs"
RAW        = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
INBOX = ROOT / "data" / "raw" / "inbox"
INBOX.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Hub · Procurement Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1C1C1A; }
    header[data-testid="stHeader"] { background-color: #1C1C1A !important; }

    section[data-testid="stSidebar"] {
        background-color: #141412 !important;
        min-width: 240px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #9CA3AF !important;
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div { color: #D1D5DB; }
    h1, h2, h3, h4, h5 { color: #F5F5F0 !important; }

    .stTextInput input, .stNumberInput input,
    .stDateInput input, .stTextArea textarea {
        background-color: #2A2A26 !important;
        color: #F5F5F0 !important;
        border-color: #3A3A36 !important;
    }
    .stSelectbox > div > div { background-color: #2A2A26 !important; color: #F5F5F0 !important; }
    div[data-baseweb="select"] { background-color: #2A2A26 !important; }
    div[data-baseweb="select"] * { color: #F5F5F0 !important; }

    .stButton > button {
        background-color: #2A2A26 !important;
        color: #F5F5F0 !important;
        border: 1px solid #3A3A36 !important;
    }
    .stButton > button:hover {
        background-color: #3A3A36 !important;
        border-color: #A3E635 !important;
    }
    section[data-testid="stFileUploader"] { background-color: #2A2A26 !important; }
    .stDataFrame { background-color: #2A2A26 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1C1C1A !important; }
    .stTabs [data-baseweb="tab"] { color: #9CA3AF !important; }
    .stTabs [aria-selected="true"] { color: #A3E635 !important; }
    [data-testid="stMetricValue"] { color: #A3E635 !important; }
    [data-testid="stMetricLabel"] { color: #9CA3AF !important; }
    .streamlit-expanderHeader { color: #D1D5DB !important; }
</style>
""", unsafe_allow_html=True)

# ── session state ────────────────────────────────────────────────────────
if "active_stage" not in st.session_state:
    st.session_state.active_stage = 0
if "rec_history" not in st.session_state:
    st.session_state.rec_history = []
if "agent_status" not in st.session_state:
    st.session_state.agent_status = "idle"   # idle | processing | error
if "agent_error" not in st.session_state:
    st.session_state.agent_error = ""
if "agent_alerts" not in st.session_state:
    st.session_state.agent_alerts = []       # list of dicts
if "agent_processed_files" not in st.session_state:
    st.session_state.agent_processed_files = set()
if "agent_enabled" not in st.session_state:
    st.session_state.agent_enabled = False
if "agent_poll_interval" not in st.session_state:
    st.session_state.agent_poll_interval = 60
if "agent_instruction" not in st.session_state:
    st.session_state.agent_instruction = ""

# ── Live Risk API ─────────────────────────────────────────────────────────
class LiveRiskAPI:
    """
    Computes risk multipliers from live external data:
      phi_market   — RBOB futures volatility (yfinance)
      phi_weather  — OpenWeatherMap storm/rain signals
      phi_finance  — USD/MXN FX rate (yfinance)
      phi_logistics— News mentions of terminal disruptions (NewsAPI)
      phi_regulatory— Static ESG / PEMEX penalty
      phi_infra    — State-level pipeline / rail risk
    """
    WEATHER_KEY  = "e078a99412eacc76b8ead4da0fd5b526"
    NEWS_KEY     = "0dd176d7bd224407931f819c2b5a876c"

    def __init__(self):
        self._cache_market  = None
        self._cache_finance = None
        self._cache_weather : dict = {}
        self._cache_logistics: dict = {}

    # ── individual factor fetchers ────────────────────────────────────────
    def _market(self, contract_type: str) -> float:
        if self._cache_market is None:
            try:
                import yfinance as yf
                rbob = yf.Ticker("RB=F").history(period="5d")["Close"]
                vol  = rbob.pct_change().std()
                self._cache_market = 1.10 if vol > 0.03 else 1.05
            except Exception:
                self._cache_market = 1.05
        return self._cache_market + (0.05 if contract_type != "Branded" else 0.0)

    def _weather(self, state: str) -> float:
        if state not in self._cache_weather:
            phi = 1.00
            if state in ["Chihuahua", "JAL", "TMS", "MX", "Veracruz", "Nuevo Leon"]:
                try:
                    url = (f"https://api.openweathermap.org/data/2.5/weather"
                           f"?q={state},MX&appid={self.WEATHER_KEY}")
                    res = __import__("requests").get(url, timeout=5).json()
                    desc = str(res.get("weather", [{}])[0].get("description", "")).lower()
                    if any(x in desc for x in ["storm","hurricane","cyclone","heavy rain"]):
                        phi = 1.25
                    elif "rain" in desc:
                        phi = 1.10
                except Exception:
                    phi = 1.05
            self._cache_weather[state] = phi
        return self._cache_weather[state]

    def _finance(self) -> float:
        if self._cache_finance is None:
            try:
                import yfinance as yf
                fx = yf.Ticker("USDMXN=X").fast_info["last_price"]
                self._cache_finance = 1.12 if fx > 20.50 else 1.00
            except Exception:
                self._cache_finance = 1.02
        return self._cache_finance

    def _logistics(self, terminal: str) -> float:
        if terminal not in self._cache_logistics:
            phi = 1.04
            try:
                q   = f'"{terminal}" AND (demurrage OR "port congestion" OR strike OR fire)'
                url = f"https://newsapi.org/v2/everything?q={q}&apiKey={self.NEWS_KEY}"
                news = __import__("requests").get(url, timeout=5).json()
                if news.get("totalResults", 0) > 0:
                    phi = 1.15
            except Exception:
                pass
            self._cache_logistics[terminal] = phi
        return self._cache_logistics[terminal]

    def get_coefficients(self, supplier: str, state: str = "Nuevo Leon",
                         terminal: str = "Monterrey", contract_type: str = "Branded") -> dict:
        phi_market     = self._market(contract_type)
        phi_weather    = self._weather(state)
        phi_finance    = self._finance()
        phi_logistics  = self._logistics(terminal)
        phi_regulatory = 1.08 if supplier.lower() in ("pemex","p.m.i.") else 1.02
        phi_infra      = 1.10 if state == "Chihuahua" else 1.03
        total          = round(phi_market * phi_weather * phi_finance * phi_logistics
                               * phi_regulatory * phi_infra, 4)
        return {
            "phi_market":     round(phi_market,     3),
            "phi_weather":    round(phi_weather,     3),
            "phi_finance":    round(phi_finance,     3),
            "phi_logistics":  round(phi_logistics,   3),
            "phi_regulatory": round(phi_regulatory,  3),
            "phi_infra":      round(phi_infra,       3),
            "total_risk_mult":total,
        }


# ── Agent helpers ─────────────────────────────────────────────────────────
AGENT_LOG = PROCESSED / "agent_log.csv"

def agent_log_entry(file_processed, action_taken, stations_affected, alerts_generated, status):
    """Append a row to agent_log.csv."""
    import datetime
    row = pd.DataFrame([{
        "timestamp":          datetime.datetime.now().isoformat(),
        "file_processed":     file_processed,
        "action_taken":       action_taken,
        "stations_affected":  stations_affected,
        "alerts_generated":   alerts_generated,
        "status":             status,
    }])
    if AGENT_LOG.exists():
        existing = pd.read_csv(AGENT_LOG)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(AGENT_LOG, index=False)


def detect_file_type(filepath):
    """Detect if a file is a structured price file or an unstructured surcharge notice."""
    ext = filepath.suffix.lower()
    name_lower = filepath.name.lower()

    # Known supplier price file patterns
    price_exts = {".csv", ".xlsx", ".xls", ".html", ".htm", ".txt"}
    surcharge_keywords = ["surcharge", "notice", "alert", "closure", "email"]

    if any(kw in name_lower for kw in surcharge_keywords):
        return "surcharge"
    if ext == ".pdf":
        # PDFs could be either — check filename
        if any(kw in name_lower for kw in surcharge_keywords):
            return "surcharge"
        if "pemex" in name_lower:
            return "price"
        return "surcharge"  # default PDF to surcharge
    if ext in price_exts:
        return "price"
    return "unknown"


KNOWN_SUPPLIERS_LIST = ["valero", "exxon", "marathon", "pemex", "g500"]

def detect_supplier_from_filename(filename):
    fl = filename.lower()
    for s in KNOWN_SUPPLIERS_LIST:
        if s in fl:
            return s
    return "unknown"


def agent_process_file(filepath):
    """Process a single inbox file. Returns (action, alerts_list)."""
    ftype = detect_file_type(filepath)
    alerts = []

    if ftype == "price":
        # Route to ingestion
        sup = detect_supplier_from_filename(filepath.name)
        if sup == "pemex":
            target_dir = RAW / "pemex" / "reportes_pdf"
        else:
            target_dir = RAW / sup
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(filepath), str(target_dir / filepath.name))

        # Run ingestion
        result = subprocess.run(
            [sys.executable, "src/ingestion/run_ingestion.py"],
            capture_output=True, text=True,
            cwd=str(ROOT), env={**os.environ}, timeout=300,
        )
        if result.returncode != 0:
            return f"ingest_failed:{filepath.name}", []

        action = f"ingested:{sup}:{filepath.name}"

    elif ftype == "surcharge":
        # Route to LLM surcharge parser
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "surcharge_no_api_key", []

        try:
            from src.ingestion.parse_llm import UnstructuredParser
            parser = UnstructuredParser()
            events = parser.parse(str(filepath))
            if events:
                parser.save_events(events)
                action = f"surcharge_parsed:{len(events)}_events:{filepath.name}"
            else:
                action = f"surcharge_no_events:{filepath.name}"
        except Exception as e:
            return f"surcharge_parse_error:{e}", []
    else:
        return f"unknown_type:{filepath.name}", []

    # Snapshot old recommendations if they exist
    old_recs = None
    recs_path = PROCESSED / "recommendations.csv"
    if recs_path.exists():
        old_recs = pd.read_csv(recs_path)

    # Re-run normalization
    subprocess.run(
        [sys.executable, "src/normalization/standardize.py",
         "--region", "monterrey", "--buyer", "capitalgas"],
        capture_output=True, text=True,
        cwd=str(ROOT), env={**os.environ}, timeout=300,
    )

    # Re-run engine
    subprocess.run(
        [sys.executable, "src/engine/engine.py",
         "--region", "monterrey", "--start", "2024-01-01", "--end", "2024-03-31",
         "--compare-baseline"],
        capture_output=True, text=True,
        cwd=str(ROOT), env={**os.environ}, timeout=600,
    )

    # Compare recommendations
    if recs_path.exists() and old_recs is not None:
        new_recs = pd.read_csv(recs_path)
        alerts = compare_recommendations(old_recs, new_recs)

    return action, alerts


def compare_recommendations(old_df, new_df):
    """Compare old vs new recommendations. Return list of alert dicts."""
    alerts = []

    # Find the winner per station in old and new
    for df, label in [(old_df, "old"), (new_df, "new")]:
        if "id_tienda" not in df.columns or "supplier" not in df.columns:
            return []

    old_winners = {}
    new_winners = {}

    lc_col = "landed_cost" if "landed_cost" in old_df.columns else "actual_landed"
    if lc_col not in old_df.columns:
        return []

    for station in old_df["id_tienda"].unique():
        old_s = old_df[old_df["id_tienda"] == station].sort_values(lc_col)
        if not old_s.empty:
            old_winners[station] = {
                "supplier": old_s.iloc[0]["supplier"],
                "landed": float(old_s.iloc[0][lc_col]),
            }

    lc_col_new = "landed_cost" if "landed_cost" in new_df.columns else "actual_landed"
    for station in new_df["id_tienda"].unique():
        new_s = new_df[new_df["id_tienda"] == station].sort_values(lc_col_new)
        if not new_s.empty:
            new_winners[station] = {
                "supplier": new_s.iloc[0]["supplier"],
                "landed": float(new_s.iloc[0][lc_col_new]),
            }

    # Find changes
    for station in set(old_winners.keys()) | set(new_winners.keys()):
        old_w = old_winners.get(station)
        new_w = new_winners.get(station)
        if old_w and new_w:
            supplier_changed = old_w["supplier"] != new_w["supplier"]
            delta = abs(new_w["landed"] - old_w["landed"])
            if supplier_changed or delta > 0.5:
                alerts.append({
                    "station": station,
                    "old_supplier": old_w["supplier"],
                    "new_supplier": new_w["supplier"],
                    "old_landed": old_w["landed"],
                    "new_landed": new_w["landed"],
                    "delta": round(new_w["landed"] - old_w["landed"], 4),
                    "changed": supplier_changed,
                })

    return alerts


def agent_filter_alerts(alerts, instruction):
    """Use LLM to interpret a natural language filter instruction and apply it."""
    if not instruction or not alerts:
        return alerts

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return alerts

    import json as _json
    import requests as _req
    prompt = (
        f"You are a filter engine. Given these procurement alerts:\n"
        f"{_json.dumps(alerts, indent=2)}\n\n"
        f"User instruction: \"{instruction}\"\n\n"
        f"Return ONLY a JSON array of the alerts that should be shown based on the instruction. "
        f"If the instruction says to ignore a supplier at a station, remove those alerts. "
        f"If it specifies a delta threshold, only keep alerts exceeding that threshold. "
        f"Return the exact same alert objects, just filtered. Return [] if none match. "
        f"Return ONLY the JSON array, no markdown fences, no explanation."
    )

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 1000,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json()["content"][0]["text"].strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return _json.loads(text)
    except Exception:
        pass
    return alerts


def _scan_inbox():
    """Scan data/raw/inbox/ for new files and process them."""
    if not INBOX.exists():
        return

    new_files = []
    for f in INBOX.iterdir():
        if f.is_file() and f.name not in st.session_state.agent_processed_files:
            new_files.append(f)

    if not new_files:
        return

    st.session_state.agent_status = "processing"
    all_alerts = []

    for fpath in new_files:
        try:
            action, alerts = agent_process_file(fpath)

            # Apply natural language filter if set
            if alerts and st.session_state.agent_instruction:
                alerts = agent_filter_alerts(alerts, st.session_state.agent_instruction)

            stations_affected = len(set(a["station"] for a in alerts)) if alerts else 0
            agent_log_entry(
                file_processed=fpath.name,
                action_taken=action,
                stations_affected=stations_affected,
                alerts_generated=len(alerts),
                status="success",
            )
            all_alerts.extend(alerts)
            st.session_state.agent_processed_files.add(fpath.name)

        except Exception as e:
            st.session_state.agent_error = str(e)[:200]
            st.session_state.agent_status = "error"
            agent_log_entry(
                file_processed=fpath.name,
                action_taken="error",
                stations_affected=0,
                alerts_generated=0,
                status=f"error:{e}",
            )

    if all_alerts:
        st.session_state.agent_alerts = all_alerts

    st.cache_data.clear()
    if st.session_state.agent_status != "error":
        st.session_state.agent_status = "idle"


def check_files():
    return {
        "ingest":    (PROCESSED / "all_suppliers.csv").exists(),
        "normalize": (PROCESSED / "normalized_suppliers.csv").exists(),
        "engine":    (PROCESSED / "recommendations.csv").exists(),
    }

def stage_unlocked(n):
    f = check_files()
    if n == 1: return True
    if n == 5: return True   # Surcharge notice — always accessible
    if n == 6: return True   # Agent monitor — always accessible
    if n == 2: return f["ingest"]
    if n == 3: return f["normalize"]  # Historical analysis — optional
    if n == 4: return f["normalize"]  # Live recommend — needs normalized data, NOT engine
    return False

def reset_pipeline(keep_raw=False):
    """Delete processed outputs. Optionally keep raw uploaded files."""
    for fname in ["all_suppliers.csv","normalized_suppliers.csv",
                  "recommendations.csv","baseline_comparison.csv",
                  "draft_purchase_orders.csv","surcharges.csv"]:
        p = PROCESSED / fname
        if p.exists():
            p.unlink()
    if not keep_raw:
        for sub in RAW.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
            else:
                sub.unlink()
    st.cache_data.clear()

# ── data loaders ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_csv(path: str):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()

def load_suppliers():    return load_csv(str(PROCESSED / "all_suppliers.csv"))
def load_normalized():   return load_csv(str(PROCESSED / "normalized_suppliers.csv"))
def load_recommendations(): return load_csv(str(PROCESSED / "recommendations.csv"))

def load_tiendas():
    for fname in ["tiendas_capitalgas.csv","catalogo_tiendas.csv","stations.csv"]:
        p = CAPITALGAS / fname
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()

def load_hist():
    p = CAPITALGAS / "historico_pedidos.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "fecha_pedido" in df.columns:
            df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"])
        return df
    return pd.DataFrame()


def rank_live(sel_station, product, order_qty, order_dt):
    """
    Reusable ranking function — same logic used by Stage 4 and Agent.
    Returns (result_dict, error_string). One of them will be None.

    result_dict keys:
        ranked_suppliers: list of dicts with full cost breakdown per supplier
        winner: dict with the top supplier details
        pemex_pct, pemex_forced: compliance info
        excluded: list of suppliers removed by hard constraints
    """
    norm = load_normalized()
    hist = load_hist()

    if norm.empty:
        return None, "No normalized data found. Run Ingest + Normalize first."

    # Filter candidates
    mask = pd.Series([True] * len(norm), index=norm.index)
    if "id_tienda"         in norm.columns: mask &= norm["id_tienda"]         == sel_station
    if "product_type"      in norm.columns: mask &= norm["product_type"]      == product
    if "supplier_available" in norm.columns: mask &= norm["supplier_available"] == True
    cands = norm[mask].copy()

    if "date" in cands.columns:
        cands["date"] = pd.to_datetime(cands["date"], errors="coerce")
        cands = cands.sort_values("date", ascending=False).drop_duplicates(subset=["supplier"], keep="first")

    if cands.empty:
        return None, f"No supplier data for station {sel_station} / {product}."

    # Surcharge — apply only if order date falls within effective window
    sur_col = pd.Series(0.0, index=cands.index)
    if "surcharge_mxn_per_l" in cands.columns:
        sur_col = cands["surcharge_mxn_per_l"].fillna(0).copy()

    # Cross-check against surcharges.csv to zero out expired surcharges
    _sur_path = PROCESSED / "surcharges.csv"
    if _sur_path.exists() and sur_col.sum() > 0:
        _sur_df = pd.read_csv(_sur_path)
        _order_dt_ts = pd.Timestamp(order_dt)
        for _ci, _cr in cands.iterrows():
            if sur_col.loc[_ci] > 0:
                # Check if ANY active surcharge covers this supplier+product+date
                _sup = str(_cr.get("supplier", "")).lower()
                _active = False
                for _, _sr in _sur_df.iterrows():
                    if str(_sr.get("supplier", "")).lower() == _sup:
                        _s_from = pd.to_datetime(_sr.get("effective_from"), errors="coerce")
                        _s_to = pd.to_datetime(_sr.get("effective_to"), errors="coerce")
                        _s_prod = str(_sr.get("product", "all")).lower()
                        if (_s_prod == "all" or _s_prod == product.lower()):
                            if pd.notna(_s_from) and pd.notna(_s_to):
                                if _s_from <= _order_dt_ts <= _s_to:
                                    _active = True
                                    break
                if not _active:
                    sur_col.loc[_ci] = 0.0  # Surcharge expired for this order date

    # Freight
    cands["actual_freight_per_l"] = (cands["freight_cost_mxn"] / order_qty).round(6) if "freight_cost_mxn" in cands.columns else pd.Series(0, index=cands.index)

    # Distance penalty
    def _dist_penalty(dist_km):
        d = float(dist_km) if pd.notna(dist_km) else 0
        if d <= 50:   return 0.0
        if d <= 100:  return 0.15
        if d <= 150:  return 0.30
        if d <= 200:  return 0.50
        if d <= 300:  return 0.75
        return 1.00

    dist_col = cands["dist_km"] if "dist_km" in cands.columns else pd.Series(0, index=cands.index)
    cands["dist_penalty"] = dist_col.apply(_dist_penalty)

    # Live Risk
    _risk_api = LiveRiskAPI()
    risk_mults = []
    for _, row in cands.iterrows():
        try:
            coeffs = _risk_api.get_coefficients(
                supplier=str(row.get("supplier", "Unknown")),
                state=str(row.get("state", "Nuevo Leon")),
                terminal=str(row.get("terminal_name", sel_station)),
                contract_type=str(row.get("contract_type", "Branded")),
            )
            risk_mults.append(coeffs["total_risk_mult"])
        except Exception:
            risk_mults.append(1.0)
    cands["risk_mult"] = risk_mults

    # Landed cost
    cands["base_landed"] = (cands["price_mxn_per_l"] + cands["actual_freight_per_l"] + sur_col).round(4)
    cands["actual_landed"] = ((cands["base_landed"] + cands["dist_penalty"]) * cands["risk_mult"]).round(4)
    cands["total_cost"] = (cands["actual_landed"] * order_qty).round(0)

    # Hard constraints
    excluded = []
    passed_idx = []
    for ci, cr in cands.iterrows():
        reasons = []
        if cr["price_mxn_per_l"] < 15 or cr["price_mxn_per_l"] > 35:
            reasons.append("Price outside 15–35 MXN/L guardrail")
        moq = cr.get("moq_litros", 0) or 0
        if order_qty < moq:
            reasons.append(f"Below MOQ ({moq:,.0f}L)")
        if reasons:
            excluded.append({"supplier": cr["supplier"], "reasons": reasons})
        else:
            passed_idx.append(ci)

    if passed_idx:
        cands = cands.loc[passed_idx].copy()

    # Reliability scores
    reliability_scores = {}
    if not hist.empty and "proveedor" in hist.columns and "dias_entrega" in hist.columns:
        hist["_lead_promised"] = hist.get("lead_time_prometido", 3)
        for sup in cands["supplier"].unique():
            sup_hist = hist[hist["proveedor"].str.lower() == sup.lower()]
            if len(sup_hist) > 10:
                late = (sup_hist["dias_entrega"] > sup_hist["_lead_promised"]).sum()
                reliability_scores[sup] = round(1.0 - (late / len(sup_hist)), 3)
            else:
                reliability_scores[sup] = 0.5

    cands["reliability_score"] = cands["supplier"].map(reliability_scores).fillna(0.5)

    # Volatility
    vol_col = cands["price_volatility_30d"] if "price_volatility_30d" in cands.columns else pd.Series(0, index=cands.index)

    # Weighted scoring
    cost_min = cands["actual_landed"].min()
    cost_max = cands["actual_landed"].max()
    cost_range = cost_max - cost_min if cost_max > cost_min else 1
    cands["cost_score"] = (1 - (cands["actual_landed"] - cost_min) / cost_range).round(4)

    vol_min = vol_col.min()
    vol_max = vol_col.max()
    vol_range = vol_max - vol_min if vol_max > vol_min else 1
    cands["volatility_score"] = (1 - (vol_col - vol_min) / (vol_range + 1e-9)).round(4)

    cands["final_score"] = (
        0.60 * cands["cost_score"] +
        0.25 * cands["reliability_score"] +
        0.15 * cands["volatility_score"]
    ).round(4)

    cands = cands.sort_values("final_score", ascending=False)

    # PEMEX compliance
    pemex_pct = 0.0
    pemex_forced = False
    pemex_applies = True

    restriccion_path = CAPITALGAS / "restriccion_pemex.csv"
    if restriccion_path.exists():
        restr = pd.read_csv(restriccion_path)
        st_restr = restr[restr["id_tienda"] == sel_station]
        if not st_restr.empty:
            prod_col = f"aplica_{product.lower()}"
            if prod_col in st_restr.columns:
                pemex_applies = bool(st_restr.iloc[0][prod_col])

    if pemex_applies and not hist.empty and "id_tienda" in hist.columns:
        ots = pd.Timestamp(order_dt)
        month_start = ots.replace(day=1)
        month_end = month_start + pd.offsets.MonthEnd(1)
        mo = hist[
            (hist["id_tienda"] == sel_station) &
            (hist["fecha_pedido"] >= month_start) &
            (hist["fecha_pedido"] <= month_end)
        ]
        if "producto" in mo.columns:
            mo = mo[mo["producto"].str.lower() == product.lower()]
        if len(mo) > 0 and "litros_pedidos" in mo.columns:
            tot_l = mo["litros_pedidos"].sum()
            pmx_l = mo[mo["es_pemex"] == True]["litros_pedidos"].sum() if "es_pemex" in mo.columns else 0
            pemex_pct = (pmx_l / tot_l * 100) if tot_l > 0 else 0.0
    elif not pemex_applies:
        pemex_pct = 100.0

    pemex_forced = pemex_applies and pemex_pct < 50.0
    if pemex_forced and "Pemex" in cands["supplier"].values:
        cands = pd.concat([cands[cands["supplier"] == "Pemex"], cands[cands["supplier"] != "Pemex"]], ignore_index=True)

    # Build result
    best = cands.iloc[0]
    ranked = []
    for _, row in cands.iterrows():
        ranked.append({
            "supplier": row["supplier"],
            "invoice": round(float(row["price_mxn_per_l"]), 4),
            "freight": round(float(row["actual_freight_per_l"]), 4),
            "surcharge": round(float(sur_col.loc[row.name] if row.name in sur_col.index else 0), 4),
            "dist_penalty": round(float(row.get("dist_penalty", 0)), 2),
            "dist_km": round(float(row.get("dist_km", 0)), 1),
            "risk_mult": round(float(row.get("risk_mult", 1.0)), 3),
            "landed": round(float(row["actual_landed"]), 4),
            "total_cost": round(float(row["total_cost"]), 0),
            "reliability": round(float(row.get("reliability_score", 0.5)), 3),
            "final_score": round(float(row.get("final_score", 0)), 4),
        })

    return {
        "ranked": ranked,
        "winner": ranked[0] if ranked else None,
        "pemex_pct": round(pemex_pct, 1),
        "pemex_forced": pemex_forced,
        "pemex_applies": pemex_applies,
        "excluded": excluded,
        "order": {"station": sel_station, "product": product, "qty": order_qty, "date": str(order_dt)},
    }, None

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ─────────────────────────────────────────────────────────
    import base64 as _b64
    _logo_path = ROOT / "assets" / "logo.png"
    if _logo_path.exists():
        _logo_b64 = _b64.b64encode(_logo_path.read_bytes()).decode()
        st.markdown(
            f'<div style="text-align:center;padding:8px 0">' +
            f'<img src="data:image/png;base64,{_logo_b64}" ' +
            f'style="width:180px;height:auto;border-radius:12px;image-rendering:crisp-edges"></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="text-align:center"><span style="font-size:22px;font-weight:800;color:#A3E635">◆ AI Hub</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center"><span style="font-size:15px;color:#6B7280">Procurement Engine · Globant x UB</span></div>', unsafe_allow_html=True)
    st.divider()

    files = check_files()
    done_count = sum(files.values())
    st.progress(done_count / 3, text=f"{done_count} of 3 stages complete")
    st.markdown("")

    surcharge_file = PROCESSED / "surcharges.csv"
    stages = [
        {"n":1,"icon":"📁","title":"Upload & Ingest",    "done":files["ingest"]},
        {"n":5,"icon":"📨","title":"Surcharge Notice",   "done":surcharge_file.exists()},
        {"n":2,"icon":"🔗","title":"Normalize",          "done":files["normalize"]},
        {"n":3,"icon":"📊","title":"Historical Analysis",  "done":files["engine"]},
        {"n":4,"icon":"💡","title":"Live Recommend",       "done":False},
    ]

    for s in stages:
        n = s["n"]
        is_unlocked = stage_unlocked(n)
        is_active   = (st.session_state.active_stage == n)
        is_done     = s["done"]

        if is_done:    label = f"✅  {s['title']}"
        elif is_active: label = f"▶  {s['title']}"
        elif is_unlocked: label = f"{s['icon']}  {s['title']}"
        else:           label = f"🔒  {s['title']}"

        if is_unlocked:
            if st.button(label, key=f"nav_{n}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_stage = n
                st.rerun()
        else:
            st.markdown(f"<small style='color:#4B5563;padding:8px;display:block'>{label}</small>",
                        unsafe_allow_html=True)

    st.divider()

    # ── RESET SECTION ──────────────────────────────────────────────────
    with st.expander("🗑️ Reset Pipeline"):
        st.markdown("<small style='color:#9CA3AF'>Delete processed outputs and start fresh. Raw uploaded files can be kept or also cleared.</small>", unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Clear Processed Data Only", use_container_width=True):
            reset_pipeline(keep_raw=True)
            st.session_state.active_stage = 1
            st.success("Cleared. Raw files kept.")
            st.rerun()
        if st.button("🗑️ Clear Everything (incl. raw files)", use_container_width=True):
            reset_pipeline(keep_raw=False)
            st.session_state.active_stage = 1
            st.success("All pipeline data cleared.")
            st.rerun()

    # ── SURCHARGE MANAGEMENT ───────────────────────────────────────────
    with st.expander("⚡ Surcharges"):
        st.markdown("<small style='color:#9CA3AF'>Add or view active surcharges. Surcharges are added on top of invoice + freight to compute landed cost. Re-run normalization after any change.</small>", unsafe_allow_html=True)

        sur_path = PROCESSED / "surcharges.csv"

        # Show existing surcharges
        if sur_path.exists():
            sur_df = pd.read_csv(sur_path)
            if not sur_df.empty:
                st.markdown("<small style='color:#A3E635'>Active surcharge events:</small>", unsafe_allow_html=True)
                st.dataframe(sur_df, use_container_width=True, hide_index=True, height=120)
            else:
                st.caption("No surcharges on file.")
        else:
            st.caption("No surcharges.csv yet.")

        st.markdown("")
        st.markdown("<small style='color:#9CA3AF'>Add a new surcharge:</small>", unsafe_allow_html=True)
        s_supplier  = st.text_input("Supplier", value="Valero",        key="sur_sup")
        s_terminal  = st.text_input("Terminal (or 'all')", value="all",key="sur_term")
        s_product   = st.selectbox("Product", ["all","Regular","Premium","Diesel"], key="sur_prod")
        s_amount    = st.number_input("Amount (MXN/L)", 0.0001, 5.0, 0.05, 0.01, key="sur_amt", format="%.4f")
        s_from      = st.date_input("Effective from", value=date(2024,1,1),  key="sur_from")
        s_to        = st.date_input("Effective to",   value=date(2024,12,31), key="sur_to")

        if st.button("➕ Add Surcharge", use_container_width=True, key="add_sur"):
            new_row = pd.DataFrame([{
                "supplier":       s_supplier,
                "terminal":       s_terminal,
                "product":        s_product,
                "surcharge_per_l":s_amount,
                "effective_from": str(s_from),
                "effective_to":   str(s_to),
            }])
            if sur_path.exists():
                existing = pd.read_csv(sur_path)
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row
            combined.to_csv(sur_path, index=False)
            st.cache_data.clear()
            st.success(f"Added: {s_supplier} +{s_amount:.4f} MXN/L")
            st.rerun()

        if sur_path.exists() and st.button("🗑️ Clear All Surcharges", use_container_width=True, key="clear_sur"):
            sur_path.unlink()
            st.cache_data.clear()
            st.rerun()

        st.markdown("<small style='color:#F59E0B'>⚠ Re-run Normalization after adding surcharges for them to take effect in recommendations.</small>", unsafe_allow_html=True)

    st.divider()

    # ── AGENTIC MONITOR (sidebar compact view) ────────────────────────
    _agent_status = st.session_state.agent_status
    _n_alerts = len(st.session_state.agent_alerts)

    if _agent_status == "idle" and st.session_state.agent_enabled:
        _dot_html = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22C55E;margin-right:6px;animation:pulse 2s infinite"></span>'
        _label = "Watching"
    elif _agent_status == "processing":
        _dot_html = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#F59E0B;margin-right:6px;animation:pulse 0.5s infinite"></span>'
        _label = "Processing..."
    elif _agent_status == "error":
        _dot_html = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#EF4444;margin-right:6px"></span>'
        _label = "Error"
    else:
        _dot_html = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4B5563;margin-right:6px"></span>'
        _label = "Off"

    # Pulse animation
    st.markdown("""<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>""", unsafe_allow_html=True)

    _alert_badge = ""
    if _n_alerts > 0:
        _alert_badge = (f'<span style="background:#7F1D1D;color:#FCA5A5;font-weight:700;font-size:10px;'
                        f'padding:2px 7px;border-radius:10px;margin-left:8px">🔔 {_n_alerts}</span>')

    st.markdown(
        f'<div style="display:flex;align-items:center;padding:4px 0">'
        f'{_dot_html}'
        f'<span style="font-size:12px;color:#D1D5DB;font-weight:600">Agent: {_label}</span>'
        f'{_alert_badge}'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.session_state.agent_error:
        st.markdown(f"<div style='font-size:10px;color:#EF4444;padding-left:14px'>{st.session_state.agent_error[:80]}</div>", unsafe_allow_html=True)

    # Nav to agent page
    _agent_btn_label = f"🤖  Procurement Agent" if _n_alerts == 0 else f"🤖  Agent ({_n_alerts} alerts)"
    if st.button(_agent_btn_label, key="nav_6", use_container_width=True,
                 type="primary" if st.session_state.active_stage == 6 else "secondary"):
        st.session_state.active_stage = 6
        st.rerun()

    st.divider()
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.active_stage = 0
        st.rerun()

    st.caption("Globant · University at Buffalo")

# ══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════
active = st.session_state.active_stage

# ── LANDING ──────────────────────────────────────────────────────────────
if active == 0:
    st.markdown("")
    st.markdown("")
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("# ◆ AI Hub Procurement Engine")
        st.markdown(
            "Config-driven pipeline that ingests multi-supplier fuel pricing, "
            "normalizes against station logistics, enforces PEMEX compliance, "
            "and recommends the optimal supplier per order."
        )
        st.markdown("")

        f = check_files()
        if any(f.values()):
            st.info(
                f"Previous pipeline data found — "
                f"{'Ingest ✓ ' if f['ingest'] else ''}"
                f"{'Normalize ✓ ' if f['normalize'] else ''}"
                f"{'Engine ✓' if f['engine'] else ''}"
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Start New Pipeline", use_container_width=True, type="primary"):
                st.session_state.active_stage = 1
                st.rerun()
        with c2:
            if f["normalize"]:
                if st.button("💡 Go to Recommend", use_container_width=True):
                    st.session_state.active_stage = 4
                    st.rerun()
            elif f["ingest"]:
                if st.button("↩ Resume Pipeline", use_container_width=True):
                    st.session_state.active_stage = 2
                    st.rerun()

# ── STAGE 1: UPLOAD & INGEST ─────────────────────────────────────────────
elif active == 1:
    st.header("📁 Stage 1 — Upload & Ingest")
    st.markdown(
        "Upload raw supplier pricing files. The pipeline detects formats, "
        "parses prices, and produces `all_suppliers.csv`."
    )

    KNOWN_SUPPLIERS = ["valero", "exxon", "marathon", "pemex", "g500"]

    def detect_supplier(filename):
        fl = filename.lower()
        for s in KNOWN_SUPPLIERS:
            if s in fl: return s
        return "unknown"

    # Supplier folder mapping info
    with st.expander("📂 Expected folder structure"):
        st.markdown("""
| Supplier | Format | Saved to |
|----------|--------|----------|
| Valero | HTML | `data/raw/valero/` |
| Exxon | XLSX | `data/raw/exxon/` |
| Marathon | TXT | `data/raw/marathon/` |
| PEMEX | PDF | `data/raw/pemex/reportes_pdf/` |
| G500 | CSV | `data/raw/g500/` |

Include the supplier name in the filename: `valero_20240101.html`
        """)

    uploaded = st.file_uploader(
        "Drop supplier files here — any format",
        accept_multiple_files=True,
    )

    if uploaded:
        st.markdown("#### Detected Files")
        meta = []
        for uf in uploaded:
            sup  = detect_supplier(uf.name)
            ext  = Path(uf.name).suffix
            size = len(uf.getvalue()) / 1024
            st.markdown(f"- **{uf.name}** → supplier: `{sup}` · format: `{ext}` · {size:.0f} KB")
            meta.append({"name": uf.name, "sup": sup, "obj": uf})

        st.markdown("")

        if st.button("⚡ Run Ingestion Pipeline", type="primary", use_container_width=True):
            # Save to correct supplier subfolders
            for m in meta:
                if m["sup"] == "pemex":
                    target_dir = RAW / "pemex" / "reportes_pdf"
                else:
                    target_dir = RAW / m["sup"]
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / m["name"]).write_bytes(m["obj"].getvalue())

            with st.spinner("Running ingestion pipeline..."):
                result = subprocess.run(
                    [sys.executable, "src/ingestion/run_ingestion.py"],
                    capture_output=True, text=True,
                    cwd=str(ROOT), env={**os.environ}, timeout=300,
                )

            st.cache_data.clear()

            out = PROCESSED / "all_suppliers.csv"
            # Auto-find if written elsewhere
            if not out.exists():
                found = [p for p in ROOT.rglob("all_suppliers.csv")
                         if "venv" not in str(p) and "__pycache__" not in str(p)]
                if found and str(found[0]) != str(out):
                    shutil.copy(str(found[0]), str(out))

            if out.exists():
                df = load_suppliers()
                st.success(f"✅ Ingestion complete — {len(df):,} records from "
                           f"{df['supplier'].nunique() if not df.empty else 0} suppliers")
                if not df.empty:
                    with st.expander("Preview", expanded=True):
                        st.dataframe(df.head(20), use_container_width=True)
                if st.button("→ Continue to Normalize", type="primary", use_container_width=True):
                    st.session_state.active_stage = 2
                    st.rerun()
            else:
                st.error("Ingestion did not produce all_suppliers.csv")
                with st.expander("stdout"): st.code(result.stdout[-3000:] or "(empty)")
                with st.expander("stderr"):  st.code(result.stderr[-3000:] or "(empty)")

    elif check_files()["ingest"]:
        df = load_suppliers()
        st.info(f"Previous ingestion found: {len(df):,} records on disk.")
        with st.expander("Preview existing data"):
            st.dataframe(df.head(20), use_container_width=True)
        if st.button("→ Continue to Normalize", type="primary"):
            st.session_state.active_stage = 2
            st.rerun()

# ── STAGE 5: SURCHARGE NOTICE ────────────────────────────────────────────
elif active == 5:
    st.header("📨 Surcharge Notice Parser")
    st.markdown(
        "Paste or upload an unstructured email or notice from a supplier. "
        "The LLM parser reads the prose, extracts the surcharge amount, affected product, "
        "terminal, and effective dates — then writes to `surcharges.csv`. "
        "The next normalization run will add this surcharge to the landed cost automatically."
    )

    SURCHARGES_CSV = PROCESSED / "surcharges.csv"

    # Show existing surcharges
    if SURCHARGES_CSV.exists():
        existing = pd.read_csv(SURCHARGES_CSV)
        if not existing.empty:
            st.success(f"✅ {len(existing)} active surcharge event(s) on disk — will be included in next normalization run.")
            with st.expander("View active surcharges"):
                st.dataframe(existing[["supplier","terminal","product","surcharge_per_l",
                                       "effective_from","effective_to","reason","confidence"]],
                             use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear all surcharges", type="secondary"):
                SURCHARGES_CSV.unlink()
                st.rerun()

    st.divider()

    # Input method
    input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    notice_text = ""
    supplier_hint = st.selectbox("Supplier (optional hint for LLM accuracy)",
                                  ["Auto-detect", "Valero", "Exxon", "Marathon", "Pemex", "Other"])
    hint = None if supplier_hint == "Auto-detect" else supplier_hint

    if input_method == "Paste text":
        notice_text = st.text_area(
            "Paste surcharge email or notice here",
            height=220,
            placeholder="""Example:
From: pricing@valero.com
Subject: Temporary Surcharge — Monterrey Terminal

Due to highway closure, we are adding a 0.85 MXN/L surcharge on Regular and Premium
at our Monterrey TMS terminal, effective 2024-02-15 through 2024-02-22."""
        )

        # Demo button
        if st.button("📋 Load demo surcharge email", type="secondary"):
            demo_path = ROOT / "demo_surcharge_email.txt"
            if demo_path.exists():
                st.session_state["_demo_notice"] = demo_path.read_text()
                st.rerun()

        # Load demo if set
        if st.session_state.get("_demo_notice"):
            notice_text = st.session_state["_demo_notice"]
            st.info("Demo email loaded. Click Parse to extract the surcharge.")

    else:
        uploaded_notice = st.file_uploader("Upload surcharge notice (TXT, PDF, HTML, XLSX)",
                                           type=["txt","pdf","html","htm","xlsx","xls"])
        if uploaded_notice:
            tmp_path = ROOT / "data" / "raw" / "unknown" / uploaded_notice.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded_notice.getvalue())
            notice_text = str(tmp_path)  # will be parsed as file

    if notice_text and st.button("🤖 Parse with LLM", type="primary", use_container_width=True):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.error("ANTHROPIC_API_KEY not set in .env — required for LLM parsing.")
        else:
            with st.spinner("Sending to LLM for extraction..."):
                try:
                    sys.path.insert(0, str(ROOT))
                    from src.ingestion.parse_llm import UnstructuredParser
                    parser = UnstructuredParser()

                    # Parse as text or file
                    if input_method == "Paste text" or st.session_state.get("_demo_notice"):
                        actual_text = st.session_state.get("_demo_notice") or notice_text
                        events = parser.parse_text(actual_text, hint, "manual_input")
                    else:
                        events = parser.parse(notice_text, hint)

                    if not events:
                        st.warning("No surcharge events found in this document. "
                                   "The LLM found no pricing changes to extract.")
                    else:
                        # Save to surcharges.csv
                        parser.save_events(events)
                        st.session_state["_demo_notice"] = None

                        st.success(f"✅ Extracted and saved {len(events)} surcharge event(s)")
                        st.markdown("#### Extracted Events")

                        for i, ev in enumerate(events, 1):
                            bc = "#A3E635" if ev.confidence == "high" else "#F59E0B" if ev.confidence == "medium" else "#EF4444"
                            st.markdown(f"""
                            <div style="background:#1A2910;border:1px solid #365314;border-radius:10px;
                                        padding:14px 18px;margin-bottom:10px">
                              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                                <span style="font-size:14px;font-weight:700;color:#F5F5F0">
                                  Event {i} — {ev.supplier} · {ev.terminal}
                                </span>
                                <span style="font-size:10px;font-weight:700;color:{bc};
                                             border:1px solid {bc};padding:2px 8px;border-radius:20px">
                                  {ev.confidence.upper()} CONFIDENCE
                                </span>
                              </div>
                              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
                                <div>
                                  <div style="font-size:9px;color:#4B5563;text-transform:uppercase;letter-spacing:0.1em">Surcharge</div>
                                  <div style="font-size:20px;font-weight:700;color:#F59E0B;font-family:monospace">+{ev.surcharge_per_l:.2f} MXN/L</div>
                                </div>
                                <div>
                                  <div style="font-size:9px;color:#4B5563;text-transform:uppercase;letter-spacing:0.1em">Product</div>
                                  <div style="font-size:15px;font-weight:600;color:#D1D5DB">{ev.product}</div>
                                </div>
                                <div>
                                  <div style="font-size:9px;color:#4B5563;text-transform:uppercase;letter-spacing:0.1em">Effective</div>
                                  <div style="font-size:13px;color:#D1D5DB;font-family:monospace">{ev.effective_from} → {ev.effective_to}</div>
                                </div>
                              </div>
                              <div style="margin-top:10px;font-size:12px;color:#6B7280">📝 {ev.reason}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # ── AUTO CLOSE THE LOOP: normalize → engine → recommend ──
                        st.markdown("""
                        <div style="background:#1A2910;border:2px solid #A3E635;border-radius:10px;
                                    padding:14px 18px;margin-top:14px;display:flex;align-items:center;gap:12px">
                          <span style="font-size:22px">⚡</span>
                          <div>
                            <div style="font-size:13px;font-weight:700;color:#A3E635">Auto-applying surcharge to pipeline...</div>
                            <div style="font-size:11px;color:#6B7280;margin-top:2px">
                              Running Normalize → Engine automatically so you can see the impact immediately.
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        _loop_ok = True

                        # Step A: Re-run normalization
                        with st.spinner("🔗 Re-running normalization with new surcharge..."):
                            _norm_result = subprocess.run(
                                [sys.executable, "src/normalization/standardize.py",
                                 "--region", "monterrey", "--buyer", "capitalgas"],
                                capture_output=True, text=True,
                                cwd=str(ROOT), env={**os.environ}, timeout=300,
                            )
                        st.cache_data.clear()
                        if (PROCESSED / "normalized_suppliers.csv").exists():
                            st.success("✅ Normalization updated with surcharge")
                        else:
                            st.error("Normalization failed after surcharge")
                            if _norm_result.stderr:
                                with st.expander("stderr"): st.code(_norm_result.stderr[-2000:])
                            _loop_ok = False

                        # Step B: Re-run engine
                        if _loop_ok:
                            with st.spinner("⚙️ Re-running engine to refresh recommendations..."):
                                _eng_result = subprocess.run(
                                    [sys.executable, "src/engine/engine.py",
                                     "--region", "monterrey",
                                     "--start", "2024-01-01", "--end", "2024-03-31",
                                     "--compare-baseline"],
                                    capture_output=True, text=True,
                                    cwd=str(ROOT), env={**os.environ}, timeout=600,
                                )
                            st.cache_data.clear()
                            if (PROCESSED / "recommendations.csv").exists():
                                st.success("✅ Engine refreshed — recommendations updated")
                            else:
                                st.error("Engine failed after surcharge normalization")
                                if _eng_result.stderr:
                                    with st.expander("stderr"): st.code(_eng_result.stderr[-2000:])
                                _loop_ok = False

                        # Step C: Go to Recommend
                        if _loop_ok:
                            st.markdown("""
                            <div style="background:#1A2910;border:1px solid #365314;border-radius:10px;
                                        padding:14px 18px;margin-top:10px">
                              <div style="font-size:13px;font-weight:700;color:#A3E635">
                                ✅ Full loop complete — surcharge is live in recommendations
                              </div>
                              <div style="font-size:11px;color:#6B7280;margin-top:2px">
                                Go to Recommend to see the impact on supplier rankings.
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("")
                            if st.button("💡 Go to Recommendations", type="primary",
                                         use_container_width=True, key="sur_goto_rec"):
                                st.session_state.active_stage = 4
                                st.rerun()
                        else:
                            st.markdown("")
                            if st.button("🔗 Go to Normalize manually", type="secondary",
                                         use_container_width=True, key="sur_goto_norm_fallback"):
                                st.session_state.active_stage = 2
                                st.rerun()

                except ImportError as e:
                    st.error(f"Could not import parse_llm: {e}. "
                             f"Ensure src/ingestion/parse_llm.py exists in your project.")
                except Exception as e:
                    st.error(f"Parsing failed: {e}")

# ── STAGE 2: NORMALIZE ───────────────────────────────────────────────────
elif active == 2:
    st.header("🔗 Stage 2 — Normalize")
    st.markdown(
        "Join supplier prices with CapitalGas station data to compute landed costs "
        "(invoice + CANACAR freight 28.40 MXN/km + surcharges) per supplier per station."
    )

    if not stage_unlocked(2):
        st.warning("🔒 Complete Stage 1 first.")
    else:
        col1, col2 = st.columns(2)
        with col1: region = st.selectbox("Region", ["monterrey", "all"], index=0)
        with col2: buyer  = st.text_input("Buyer", value="capitalgas")

        _auto_norm = st.session_state.pop("_auto_normalize", False)
        if _auto_norm:
            st.success("⚡ Surcharge detected — auto-running Normalize to apply it now...")

        if st.button("🔗 Run Normalization", type="primary", use_container_width=True) or _auto_norm:
            cmd = [sys.executable, "src/normalization/standardize.py",
                   "--region", region, "--buyer", buyer.lower()]

            with st.spinner("Running normalization..."):
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(ROOT), env={**os.environ}, timeout=300,
                )

            st.cache_data.clear()

            if (PROCESSED / "normalized_suppliers.csv").exists():
                ndf = load_normalized()
                n_stations = ndf["id_tienda"].nunique() if "id_tienda" in ndf.columns else "?"
                st.success(f"✅ Normalization complete — {len(ndf):,} rows, {n_stations} stations")
                with st.expander("Landed cost preview", expanded=True):
                    show_cols = [c for c in ["id_tienda","supplier","product_type",
                                             "price_mxn_per_l","dist_km",
                                             "freight_cost_mxn","surcharge_mxn_per_l",
                                             "landed_cost"] if c in ndf.columns]
                    st.dataframe(ndf[show_cols].head(20), use_container_width=True)
                if st.button("→ Continue to Recommend", type="primary", use_container_width=True):
                    st.session_state.active_stage = 4
                    st.rerun()
            else:
                st.error("Normalization did not produce normalized_suppliers.csv")
                with st.expander("stdout"): st.code(result.stdout[-3000:] or "(empty)")
                with st.expander("stderr"):  st.code(result.stderr[-3000:] or "(empty)")

        if check_files()["normalize"]:
            ndf = load_normalized()
            st.info(f"Previous normalization found: {len(ndf):,} rows on disk.")
            if st.button("→ Continue to Recommend (use existing)", type="secondary"):
                st.session_state.active_stage = 4
                st.rerun()

# ── STAGE 3: ENGINE ──────────────────────────────────────────────────────
elif active == 3:
    st.header("📊 Stage 3 — Historical Analysis (Optional)")
    st.markdown(
        "Replay past orders through the engine to measure savings vs baseline. "
        "This is **not required** for live recommendations — go directly to Stage 4 for that. "
        "This stage requires `historico_pedidos.csv` data for the selected date range."
    )

    if not stage_unlocked(3):
        st.warning("🔒 Complete Stage 2 first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1: eng_region = st.selectbox("Region", ["monterrey","all"], index=0, key="eng_region")
        with col2: eng_start  = st.date_input("Start Date", value=date(2024,1,1), key="eng_start")
        with col3: eng_end    = st.date_input("End Date",   value=date(2024,3,31), key="eng_end")
        compare_baseline = st.checkbox("Compare against naive baseline", value=True)

        if st.button("⚙️ Run Engine", type="primary", use_container_width=True):
            # ── CORRECT PATH: src/engine/engine.py ──────────────
            cmd = [
                sys.executable, "src/engine/engine.py",
                "--region", eng_region,
                "--start",  str(eng_start),
                "--end",    str(eng_end),
            ]
            if compare_baseline:
                cmd.append("--compare-baseline")

            log_area = st.empty()
            log_area.info("⏳ Engine running — this may take 30–90 seconds for Monterrey region...")

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(ROOT), env={**os.environ}, timeout=600,
            )

            st.cache_data.clear()
            log_area.empty()

            reco_p = PROCESSED / "recommendations.csv"
            if reco_p.exists():
                rdf = load_recommendations()
                pf  = int(rdf["pemex_forced"].sum()) if "pemex_forced" in rdf.columns else "?"
                st.success(f"✅ Engine complete — {len(rdf):,} order events, {pf} PEMEX-forced")
                with st.expander("Preview", expanded=True):
                    st.dataframe(rdf.head(20), use_container_width=True)
                if st.button("→ Go to Recommendations", type="primary", use_container_width=True):
                    st.session_state.active_stage = 4
                    st.rerun()
            else:
                st.error("Engine did not produce recommendations.csv")
                if result.stdout:
                    with st.expander("stdout (last 3000 chars)"):
                        st.code(result.stdout[-3000:])
                if result.stderr:
                    with st.expander("stderr (last 3000 chars)"):
                        st.code(result.stderr[-3000:])

        if check_files()["engine"]:
            rdf = load_recommendations()
            st.info(f"Previous engine output found: {len(rdf):,} rows on disk.")
            if st.button("→ Go to Recommendations (use existing)", type="secondary"):
                st.session_state.active_stage = 4
                st.rerun()

# ── STAGE 4: RECOMMEND ───────────────────────────────────────────────────
elif active == 4:
    st.header("💡 Stage 4 — Live Recommendation")
    st.markdown(
        "Get a real-time supplier recommendation for any station, any product, any date. "
        "The engine ranks suppliers by risk-adjusted landed cost using the latest normalized data."
    )

    if not stage_unlocked(4):
        st.warning("🔒 Complete Stage 2 (Normalize) first — the recommendation engine needs normalized supplier data.")
    else:
        norm    = load_normalized()
        tiendas = load_tiendas()
        hist    = load_hist()

        form_col, result_col = st.columns([1, 2], gap="large")

        with form_col:
            st.subheader("Order Parameters")

            # Station selector
            if not tiendas.empty and "ciudad" in tiendas.columns:
                mty = tiendas[tiendas["ciudad"]=="Monterrey"][["id_tienda","nombre"]].copy()
                if not mty.empty:
                    mty["display"] = mty["nombre"] + " (" + mty["id_tienda"] + ")"
                    smap = dict(zip(mty["display"], mty["id_tienda"]))
                    sel_station = smap[st.selectbox("Station", list(smap.keys()))]
                else:
                    sel_station = st.selectbox("Station",
                        norm["id_tienda"].unique().tolist() if "id_tienda" in norm.columns else ["N/A"])
            elif not norm.empty and "id_tienda" in norm.columns:
                sel_station = st.selectbox("Station", norm["id_tienda"].unique().tolist())
            else:
                sel_station = st.text_input("Station ID")

            product   = st.selectbox("Product", ["Regular","Premium","Diesel"])
            order_qty = st.number_input("Order Quantity (litros)", 5000, 45000, 30000, 1000)
            order_dt  = st.date_input("Order Date", value=date(2024,2,19))
            get_rec   = st.button("◆ Get Recommendation", type="primary", use_container_width=True)

        with result_col:
            if get_rec and not norm.empty:
                # ── Compute candidates ────────────────────────────────────
                mask = pd.Series([True] * len(norm), index=norm.index)
                if "id_tienda"         in norm.columns: mask &= norm["id_tienda"]         == sel_station
                if "product_type"      in norm.columns: mask &= norm["product_type"]      == product
                if "supplier_available" in norm.columns: mask &= norm["supplier_available"] == True
                cands = norm[mask].copy()
                if "date" in cands.columns:
                    cands["date"] = pd.to_datetime(cands["date"], errors="coerce")
                    cands = cands.sort_values("date", ascending=False).drop_duplicates(subset=["supplier"], keep="first")

                if cands.empty:
                    st.warning("No supplier data found for this station/product combination.")
                else:
                    # ── Surcharge diagnostic ──────────────────────────────
                    sur_col = cands["surcharge_mxn_per_l"].fillna(0).copy() if "surcharge_mxn_per_l" in cands.columns else pd.Series(0, index=cands.index)

                    # Zero out surcharges that don't apply to the selected order date
                    _sur_file_check = PROCESSED / "surcharges.csv"
                    if _sur_file_check.exists() and sur_col.sum() > 0:
                        _sur_check = pd.read_csv(_sur_file_check)
                        _order_ts = pd.Timestamp(order_dt)
                        for _ci, _cr in cands.iterrows():
                            if sur_col.loc[_ci] > 0:
                                _sup = str(_cr.get("supplier", "")).lower()
                                _active = False
                                for _, _sr in _sur_check.iterrows():
                                    if str(_sr.get("supplier", "")).lower() == _sup:
                                        _s_from = pd.to_datetime(_sr.get("effective_from"), errors="coerce")
                                        _s_to = pd.to_datetime(_sr.get("effective_to"), errors="coerce")
                                        _s_prod = str(_sr.get("product", "all")).lower()
                                        if (_s_prod == "all" or _s_prod == product.lower()):
                                            if pd.notna(_s_from) and pd.notna(_s_to):
                                                if _s_from <= _order_ts <= _s_to:
                                                    _active = True
                                                    break
                                if not _active:
                                    sur_col.loc[_ci] = 0.0

                    # Check surcharges.csv directly too for verification
                    _sur_file = PROCESSED / "surcharges.csv"
                    _sur_on_disk = None
                    if _sur_file.exists():
                        _sur_on_disk = pd.read_csv(_sur_file)

                    cands["actual_freight_per_l"] = (cands["freight_cost_mxn"] / order_qty).round(6) if "freight_cost_mxn" in cands.columns else pd.Series(0, index=cands.index)

                    # ── Distance penalty tiers ────────────────────────────
                    # Penalizes suppliers far from the station to reflect
                    # operational risk: delays, driver fatigue, fuel waste
                    def _dist_penalty(dist_km):
                        d = float(dist_km) if pd.notna(dist_km) else 0
                        if d <= 50:   return 0.0
                        if d <= 100:  return 0.15
                        if d <= 150:  return 0.30
                        if d <= 200:  return 0.50
                        if d <= 300:  return 0.75
                        return 1.00  # 300+ km

                    dist_col = cands["dist_km"] if "dist_km" in cands.columns else pd.Series(0, index=cands.index)
                    cands["dist_penalty"] = dist_col.apply(_dist_penalty)

                    # ── Live Risk API multipliers ─────────────────────────
                    _risk_api = LiveRiskAPI()
                    risk_mults = []
                    risk_details = []
                    for _, row in cands.iterrows():
                        try:
                            coeffs = _risk_api.get_coefficients(
                                supplier=str(row.get("supplier", "Unknown")),
                                state=str(row.get("state", "Nuevo Leon")),
                                terminal=str(row.get("terminal_name", sel_station)),
                                contract_type=str(row.get("contract_type", "Branded")),
                            )
                            risk_mults.append(coeffs["total_risk_mult"])
                            risk_details.append(coeffs)
                        except Exception:
                            risk_mults.append(1.0)
                            risk_details.append({})
                    cands["risk_mult"] = risk_mults

                    # ── Compute risk-adjusted landed cost ─────────────────
                    cands["base_landed"] = (cands["price_mxn_per_l"] + cands["actual_freight_per_l"] + sur_col).round(4)
                    cands["actual_landed"] = ((cands["base_landed"] + cands["dist_penalty"]) * cands["risk_mult"]).round(4)
                    cands["total_cost"]    = (cands["actual_landed"] * order_qty).round(0)

                    # ── Hard constraints (from engine logic) ──────────────
                    _excluded = []
                    _passed_idx = []
                    for _ci, _cr in cands.iterrows():
                        _reasons = []
                        # Price guardrail
                        if _cr["price_mxn_per_l"] < 15 or _cr["price_mxn_per_l"] > 35:
                            _reasons.append("Price outside 15–35 MXN/L guardrail")
                        # MOQ check
                        _moq = _cr.get("moq_litros", 0) or 0
                        if order_qty < _moq:
                            _reasons.append(f"Below MOQ ({_moq:,.0f}L)")
                        # Budget check (if presupuesto data available)
                        _pres = CAPITALGAS / "presupuesto_compra.csv"
                        if _pres.exists() and not hist.empty:
                            try:
                                _pres_df = pd.read_csv(_pres)
                                _month_str = pd.Timestamp(order_dt).strftime("%Y-%m")
                                _brow = _pres_df[(_pres_df["id_tienda"]==sel_station) & (_pres_df["anio_mes"]==_month_str)]
                                if not _brow.empty:
                                    _btotal = float(_brow["presupuesto_total"].iloc[0])
                                    _breserva = float(_brow["reserva_total"].iloc[0])
                                    _busable = _btotal - _breserva
                                    _month_start = pd.Timestamp(order_dt).replace(day=1)
                                    _spent = float(hist[
                                        (hist["id_tienda"]==sel_station) &
                                        (hist["fecha_pedido"]>=_month_start) &
                                        (hist["fecha_pedido"]<pd.Timestamp(order_dt))
                                    ]["importe_total"].sum()) if "importe_total" in hist.columns else 0
                                    _bremaining = _busable - _spent
                                    if _cr["total_cost"] > _bremaining and _bremaining > 0:
                                        _reasons.append(f"Exceeds budget (remaining: {_bremaining:,.0f} MXN)")
                            except Exception:
                                pass

                        if _reasons:
                            _excluded.append({"supplier": _cr["supplier"], "reasons": _reasons})
                        else:
                            _passed_idx.append(_ci)

                    if _excluded:
                        with st.expander(f"⚠ {len(_excluded)} supplier(s) excluded by constraints"):
                            for _ex in _excluded:
                                st.markdown(f"- **{_ex['supplier']}**: {' | '.join(_ex['reasons'])}")

                    if _passed_idx:
                        cands = cands.loc[_passed_idx].copy()

                    # ── Weighted scoring (60% cost + 25% reliability + 15% volatility) ──
                    # Load reliability from historico
                    _reliability_scores = {}
                    if not hist.empty and "proveedor" in hist.columns and "dias_entrega" in hist.columns:
                        hist["_lead_promised"] = hist.get("lead_time_prometido", 3)
                        for _sup in cands["supplier"].unique():
                            _sup_hist = hist[hist["proveedor"].str.lower() == _sup.lower()]
                            if len(_sup_hist) > 10:
                                _late = (_sup_hist["dias_entrega"] > _sup_hist["_lead_promised"]).sum()
                                _reliability_scores[_sup] = round(1.0 - (_late / len(_sup_hist)), 3)
                            else:
                                _reliability_scores[_sup] = 0.5  # default when insufficient data

                    cands["reliability_score"] = cands["supplier"].map(_reliability_scores).fillna(0.5)

                    # Volatility score from normalized data
                    _vol_col = cands["price_volatility_30d"] if "price_volatility_30d" in cands.columns else pd.Series(0, index=cands.index)

                    # Normalize scores
                    _cost_min = cands["actual_landed"].min()
                    _cost_max = cands["actual_landed"].max()
                    _cost_range = _cost_max - _cost_min if _cost_max > _cost_min else 1
                    cands["cost_score"] = (1 - (cands["actual_landed"] - _cost_min) / _cost_range).round(4)

                    _vol_min = _vol_col.min()
                    _vol_max = _vol_col.max()
                    _vol_range = _vol_max - _vol_min if _vol_max > _vol_min else 1
                    cands["volatility_score"] = (1 - (_vol_col - _vol_min) / (_vol_range + 1e-9)).round(4)

                    # Weighted final score
                    cands["final_score"] = (
                        0.60 * cands["cost_score"] +
                        0.25 * cands["reliability_score"] +
                        0.15 * cands["volatility_score"]
                    ).round(4)

                    cands = cands.sort_values("final_score", ascending=False)

                    # ── PEMEX monthly compliance (per station, per product) ──
                    # Rule: 50% of each product's monthly volume at this station
                    # must come from PEMEX. We calculate month-to-date % and decide
                    # whether to force a PEMEX order.
                    pemex_pct = 0.0
                    pemex_forced = False
                    pemex_applies = True  # assume applies unless restriccion says otherwise

                    # Check if PEMEX rule applies to this product at this station
                    restriccion_path = CAPITALGAS / "restriccion_pemex.csv"
                    if restriccion_path.exists():
                        restr = pd.read_csv(restriccion_path)
                        st_restr = restr[restr["id_tienda"] == sel_station]
                        if not st_restr.empty:
                            prod_col = f"aplica_{product.lower()}"
                            if prod_col in st_restr.columns:
                                pemex_applies = bool(st_restr.iloc[0][prod_col])

                    if pemex_applies and not hist.empty and "id_tienda" in hist.columns:
                        ots = pd.Timestamp(order_dt)
                        month_start = ots.replace(day=1)
                        month_end = (month_start + pd.offsets.MonthEnd(1))

                        # Filter: this station, this product, this month
                        mo = hist[
                            (hist["id_tienda"] == sel_station) &
                            (hist["fecha_pedido"] >= month_start) &
                            (hist["fecha_pedido"] <= month_end)
                        ]
                        # Filter by product if column exists
                        if "producto" in mo.columns:
                            mo = mo[mo["producto"].str.lower() == product.lower()]

                        if len(mo) > 0 and "litros_pedidos" in mo.columns:
                            tot_l = mo["litros_pedidos"].sum()
                            pmx_l = mo[mo["es_pemex"] == True]["litros_pedidos"].sum() if "es_pemex" in mo.columns else 0
                            pemex_pct = (pmx_l / tot_l * 100) if tot_l > 0 else 0.0
                        else:
                            # No orders this month yet — PEMEX % is 0, must order from PEMEX
                            pemex_pct = 0.0

                    elif not pemex_applies:
                        # PEMEX rule doesn't apply to this product at this station
                        pemex_pct = 100.0  # show as satisfied

                    # Force PEMEX if below 50% for the month AND PEMEX is available
                    pemex_forced = pemex_applies and pemex_pct < 50.0
                    if pemex_forced and "Pemex" in cands["supplier"].values:
                        cands = pd.concat([cands[cands["supplier"]=="Pemex"], cands[cands["supplier"]!="Pemex"]], ignore_index=True)

                    best     = cands.iloc[0]
                    best_lc  = float(best["actual_landed"])
                    best_tot = float(best["total_cost"])
                    r2_tot   = float(cands.iloc[1]["total_cost"]) if len(cands)>1 else best_tot
                    savings2 = r2_tot - best_tot

                    # Save to history immediately
                    st.session_state.rec_history.append({
                        "station": sel_station, "product": product, "qty": order_qty,
                        "date": str(order_dt), "supplier": best.get("supplier","?"),
                        "landed": float(best_lc), "total": float(best_tot),
                        "pemex_pct": pemex_pct, "pemex_forced": pemex_forced, "savings2": savings2,
                    })

                    # ── Three tabs: Results | Charts | History | Live Risk ──
                    tab_res, tab_charts, tab_hist, tab_risk = st.tabs(["📋 Results", "📊 Charts", "🕘 History", "🌐 Live Risk"])

                    # ── TAB: RESULTS ───────────────────────────────────────
                    with tab_res:
                        # Winner headline
                        winner_sup = best.get("supplier","?")
                        st.markdown(f"""
                        <div style="background:#1A2910;border:1px solid #365314;border-radius:10px;
                                    padding:16px 20px;margin-bottom:16px">
                          <div style="font-size:11px;font-weight:700;color:#A3E635;letter-spacing:0.12em;
                                      text-transform:uppercase;margin-bottom:4px">Recommended Supplier</div>
                          <div style="font-size:26px;font-weight:800;color:#F5F5F0">{winner_sup}</div>
                          <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap">
                            <span style="font-family:monospace;font-size:20px;font-weight:700;color:#A3E635">
                              {best_lc:.4f} <span style="font-size:12px;color:#6B7280;font-weight:400">MXN/L</span>
                            </span>
                            <span style="font-family:monospace;font-size:20px;font-weight:700;color:#F5F5F0">
                              MXN {best_tot:,.0f} <span style="font-size:12px;color:#6B7280;font-weight:400">total</span>
                            </span>
                            <span style="font-family:monospace;font-size:14px;color:#F59E0B;margin-top:4px">
                              saves MXN {savings2:,.0f} vs next best
                            </span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Cost breakdown table — compact, with red badge on non-zero surcharge
                        def _fmt_surcharge(val):
                            try:
                                v = float(val)
                                if v > 0:
                                    return f'<span style="background:#7F1D1D;color:#FCA5A5;font-weight:700;font-family:monospace;padding:1px 7px;border-radius:12px;font-size:12px">+{v:.4f} ⚠</span>'
                            except: pass
                            return f'<span style="color:#4B5563;font-family:monospace">{val}</span>'

                        def _fmt_dist_penalty(val):
                            try:
                                v = float(val)
                                if v > 0:
                                    return f'<span style="background:#78350F;color:#FDE68A;font-weight:700;font-family:monospace;padding:1px 7px;border-radius:12px;font-size:12px">+{v:.2f} 🚛</span>'
                            except: pass
                            return f'<span style="color:#4B5563;font-family:monospace">0.00</span>'

                        def _fmt_risk(val):
                            try:
                                v = float(val)
                                if v > 1.15:
                                    return f'<span style="background:#7F1D1D;color:#FCA5A5;font-weight:700;font-family:monospace;padding:1px 7px;border-radius:12px;font-size:12px">×{v:.3f} 🔴</span>'
                                elif v > 1.08:
                                    return f'<span style="background:#78350F;color:#FDE68A;font-weight:700;font-family:monospace;padding:1px 7px;border-radius:12px;font-size:12px">×{v:.3f} 🟡</span>'
                                else:
                                    return f'<span style="color:#A3E635;font-family:monospace">×{v:.3f}</span>'
                            except: pass
                            return f'<span style="color:#4B5563;font-family:monospace">×1.000</span>'

                        cmp_rows = [{
                            "Supplier":      row["supplier"],
                            "Invoice":       f"{float(row['price_mxn_per_l']):.4f}",
                            "Freight":       f"{float(row['actual_freight_per_l']):.4f}",
                            "Surcharge":     f"{float(sur_col.loc[row.name] if row.name in sur_col.index else 0):.4f}",
                            "Dist Penalty":  f"{float(row.get('dist_penalty', 0)):.2f}",
                            "Risk ×":        f"{float(row.get('risk_mult', 1.0)):.3f}",
                            "Landed":        f"{float(row['actual_landed']):.4f}",
                            "Total MXN":     f"{int(row['total_cost']):,}",
                            "vs Best":       "✓ best" if float(row["actual_landed"])==best_lc else f"+MXN {(float(row['actual_landed'])-best_lc)*order_qty:,.0f}",
                        } for _,row in cands.iterrows()]

                        # Build HTML table with badges
                        _hdrs = ["Supplier","Invoice","Freight","Surcharge","Dist Penalty","Risk ×","Landed","Total MXN","vs Best"]
                        _tbl = '<table style="width:100%;border-collapse:collapse;font-size:13px">'
                        _tbl += '<thead><tr>' + ''.join(
                            f'<th style="text-align:left;padding:6px 10px;color:#6B7280;font-size:10px;'
                            f'text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #2E2E2A">{h}</th>'
                            for h in _hdrs) + '</tr></thead><tbody>'
                        for i, r in enumerate(cmp_rows):
                            _bg = "#1A2910" if i == 0 else "transparent"
                            _tbl += f'<tr style="background:{_bg}">'
                            for h in _hdrs:
                                if h == "Surcharge":
                                    _tbl += f'<td style="padding:6px 10px">{_fmt_surcharge(r[h])}</td>'
                                elif h == "Dist Penalty":
                                    _tbl += f'<td style="padding:6px 10px">{_fmt_dist_penalty(r[h])}</td>'
                                elif h == "Risk ×":
                                    _tbl += f'<td style="padding:6px 10px">{_fmt_risk(r[h])}</td>'
                                else:
                                    _c = "#A3E635" if h in ("Landed","vs Best") and i==0 else "#D1D5DB"
                                    _tbl += f'<td style="padding:6px 10px;color:{_c};font-family:monospace">{r[h]}</td>'
                            _tbl += '</tr>'
                        _tbl += '</tbody></table>'
                        st.markdown(_tbl, unsafe_allow_html=True)

                        # PEMEX compliance bar
                        if not pemex_applies:
                            st.markdown(f"""
                            <div style="margin-top:14px">
                              <div style="font-size:11px;font-weight:700;color:#4B5563;letter-spacing:0.14em;
                                          text-transform:uppercase;margin-bottom:8px">PEMEX Compliance</div>
                              <div style="font-size:13px;color:#6B7280">
                                PEMEX minimum does not apply to {product} at this station.
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            bc  = "#EF4444" if pemex_pct<50 else "#A3E635"
                            stx = "⚠ Below 50% — PEMEX order forced" if pemex_pct<50 else "✓ Monthly minimum met"
                            _month_label = pd.Timestamp(order_dt).strftime("%B %Y")
                            st.markdown(f"""
                            <div style="margin-top:14px">
                              <div style="font-size:11px;font-weight:700;color:#4B5563;letter-spacing:0.14em;
                                          text-transform:uppercase;margin-bottom:8px">PEMEX Compliance — {_month_label}</div>
                              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                                <span style="font-size:22px;font-weight:700;color:{bc};font-family:monospace">{pemex_pct:.1f}%</span>
                                <span style="font-size:12px;font-weight:600;color:{bc};margin-top:6px">{stx}</span>
                              </div>
                              <div style="background:#2E2E2A;border-radius:4px;height:8px;overflow:hidden">
                                <div style="height:8px;width:{min(pemex_pct*2,100):.0f}%;background:{bc};border-radius:4px"></div>
                              </div>
                              <div style="font-size:10px;color:#4B5563;margin-top:5px;font-family:monospace">
                                {product} · {sel_station} · 50% of monthly {product} volume must be PEMEX
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # AI explanation
                        st.markdown('<div style="font-size:11px;font-weight:700;color:#4B5563;letter-spacing:0.14em;text-transform:uppercase;margin:16px 0 8px">AI Explanation</div>', unsafe_allow_html=True)
                        api_key = os.environ.get("ANTHROPIC_API_KEY","")
                        if api_key:
                            import requests as _req
                            prompt = (
                                f"You are a procurement advisor for CapitalGas fuel stations in Mexico.\n"
                                f"Station: {sel_station} | Product: {product} | Qty: {order_qty:,}L | Date: {order_dt}\n"
                                f"Top supplier: {winner_sup} at {best_lc:.4f} MXN/L risk-adjusted landed cost.\n"
                                f"Cost breakdown: invoice {float(best.get('price_mxn_per_l',0)):.4f} + "
                                f"freight {float(best.get('actual_freight_per_l',0)):.4f} + "
                                f"surcharge {float(sur_col.loc[best.name] if best.name in sur_col.index else 0):.4f} + "
                                f"distance penalty {float(best.get('dist_penalty',0)):.2f} MXN/L "
                                f"(distance: {float(best.get('dist_km',0)):.0f} km), "
                                f"then × {float(best.get('risk_mult',1.0)):.3f} risk multiplier.\n"
                                f"Total: MXN {best_tot:,.0f} | Saves MXN {savings2:,.0f} vs next best.\n"
                                f"PEMEX month-to-date: {pemex_pct:.1f}% (50% min required by law).\n"
                                f"Explain in 3 sentences why this is the best choice, mention the distance/risk factors, and any compliance notes."
                            )
                            try:
                                resp = _req.post(
                                    "https://api.anthropic.com/v1/messages",
                                    headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
                                    json={"model":"claude-sonnet-4-20250514","max_tokens":300,
                                          "messages":[{"role":"user","content":prompt}]},
                                    timeout=30,
                                )
                                if resp.status_code == 200:
                                    st.markdown(f'<div style="background:#141E05;border:1px solid #365314;border-radius:8px;padding:14px 16px;font-size:13px;color:#9CA3AF;font-style:italic;line-height:1.7">{resp.json()["content"][0]["text"]}</div>', unsafe_allow_html=True)
                                else:
                                    st.warning(f"API error {resp.status_code}")
                            except Exception as e:
                                st.warning(f"Could not reach LLM API: {e}")
                        else:
                            st.info("Set ANTHROPIC_API_KEY in .env for AI explanations.")

                    # ── TAB: CHARTS ────────────────────────────────────────
                    with tab_charts:
                        try:
                            import plotly.graph_objects as go

                            sups_list = cands["supplier"].tolist()
                            sur_values = [float(sur_col.loc[r] if r in sur_col.index else 0) for r in cands.index]

                            _c_lime = "#A3E635"
                            _c_card = "#242420"
                            _c_gray = "#6B7280"
                            _c_text = "#D1D5DB"
                            _c_bg   = "rgba(0,0,0,0)"

                            _sup_colors = {
                                "Valero": "#A3E635", "Exxon": "#4A8FBF", "ExxonMobil": "#4A8FBF",
                                "Pemex": "#F59E0B", "Marathon": "#D946EF", "G500": "#6366F1",
                            }

                            # ═══ CHART 1: COST WATERFALL ═══════════════════════
                            st.markdown("##### 💧 Cost Waterfall — How each supplier\'s price builds up")
                            st.caption("Hover over any segment to see the exact MXN/L added at each step.")

                            for _wi, (_, _wr) in enumerate(cands.iterrows()):
                                _sup = _wr["supplier"]
                                _inv = float(_wr["price_mxn_per_l"])
                                _frt = float(_wr["actual_freight_per_l"])
                                _sur = float(sur_col.loc[_wr.name] if _wr.name in sur_col.index else 0)
                                _dpen = float(_wr.get("dist_penalty", 0))
                                _base = _inv + _frt + _sur + _dpen
                                _risk_uplift = float(_wr["actual_landed"]) - _base
                                _landed = float(_wr["actual_landed"])
                                _is_winner = (_wi == 0)

                                _wf = go.Figure(go.Waterfall(
                                    x=["Invoice", "Freight", "Surcharge", "Dist Penalty", "Risk Uplift", "Landed Cost"],
                                    y=[_inv, _frt, _sur, _dpen, _risk_uplift, 0],
                                    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                                    text=[f"{_inv:.4f}", f"+{_frt:.4f}",
                                          f"+{_sur:.4f}" if _sur > 0 else "—",
                                          f"+{_dpen:.2f}" if _dpen > 0 else "—",
                                          f"x{float(_wr.get('risk_mult',1)):.3f}",
                                          f"{_landed:.4f}"],
                                    textposition="outside",
                                    textfont=dict(size=11, color=_c_text),
                                    connector=dict(line=dict(color="#3A3A36", width=1)),
                                    increasing=dict(marker=dict(color="#EF4444")),
                                    decreasing=dict(marker=dict(color=_c_lime)),
                                    totals=dict(marker=dict(color=_c_lime if _is_winner else _c_gray)),
                                ))
                                _badge = "🥇 BEST" if _is_winner else f"#{_wi+1}"
                                _wf.update_layout(
                                    title=dict(text=f"{_badge} {_sup} — {_landed:.4f} MXN/L",
                                               font=dict(size=14, color=_c_lime if _is_winner else _c_text)),
                                    height=260,
                                    paper_bgcolor=_c_bg, plot_bgcolor=_c_card,
                                    font=dict(color=_c_text, size=11),
                                    xaxis=dict(color=_c_gray, gridcolor="#2E2E2A"),
                                    yaxis=dict(color=_c_gray, gridcolor="#2E2E2A", title="MXN/L"),
                                    margin=dict(t=40, b=30, l=50, r=20),
                                    showlegend=False,
                                )
                                st.plotly_chart(_wf, use_container_width=True)

                            st.divider()

                            # ═══ CHART 2: SMART SCATTER ════════════════════════
                            st.markdown("##### 🎯 Distance vs Landed Cost — Find the Sweet Spot")
                            st.caption("Bubble size = total order cost. Green zone = ideal. Hover for full breakdown.")

                            _dists = [float(cands.iloc[i].get("dist_km", 0)) for i in range(len(cands))]
                            _landeds = cands["actual_landed"].tolist()
                            _totals = cands["total_cost"].tolist()
                            _scores = cands["final_score"].tolist() if "final_score" in cands.columns else [0]*len(cands)
                            _rels = cands["reliability_score"].tolist() if "reliability_score" in cands.columns else [0.5]*len(cands)
                            _colors = [_sup_colors.get(s, "#9CA3AF") for s in sups_list]

                            fig_sc = go.Figure()
                            _max_d = max(_dists) if _dists else 100
                            _min_l = min(_landeds) if _landeds else 20
                            _max_l = max(_landeds) if _landeds else 25

                            fig_sc.add_shape(type="rect",
                                x0=0, x1=min(80, _max_d*0.4),
                                y0=_min_l-0.5, y1=_min_l+(_max_l-_min_l)*0.4,
                                fillcolor="rgba(163,230,53,0.08)",
                                line=dict(color="rgba(163,230,53,0.3)", dash="dot"))
                            fig_sc.add_annotation(x=min(40,_max_d*0.2), y=_min_l-0.3,
                                text="SWEET SPOT", showarrow=False,
                                font=dict(size=10, color="#A3E635"), opacity=0.6)

                            fig_sc.add_shape(type="rect",
                                x0=max(150,_max_d*0.6), x1=_max_d+20,
                                y0=_min_l+(_max_l-_min_l)*0.6, y1=_max_l+1,
                                fillcolor="rgba(239,68,68,0.06)",
                                line=dict(color="rgba(239,68,68,0.2)", dash="dot"))
                            fig_sc.add_annotation(x=max(180,_max_d*0.8), y=_max_l+0.5,
                                text="AVOID", showarrow=False,
                                font=dict(size=10, color="#EF4444"), opacity=0.5)

                            for i, sup in enumerate(sups_list):
                                _iv = float(cands.iloc[i]["price_mxn_per_l"])
                                _fr = float(cands.iloc[i]["actual_freight_per_l"])
                                _sv = sur_values[i]
                                _dp = float(cands.iloc[i].get("dist_penalty",0))
                                _rm = float(cands.iloc[i].get("risk_mult",1))
                                fig_sc.add_trace(go.Scatter(
                                    x=[_dists[i]], y=[_landeds[i]],
                                    mode="markers+text",
                                    marker=dict(size=max(25,min(60,_totals[i]/20000)),
                                               color=_colors[i], opacity=0.85,
                                               line=dict(width=2, color="#1C1C1A")),
                                    text=[sup], textposition="top center",
                                    textfont=dict(size=12, color=_colors[i]),
                                    name=sup,
                                    hovertemplate=(
                                        f"<b>{sup}</b><br>Distance: {_dists[i]:.0f} km<br>"
                                        f"Landed: {_landeds[i]:.4f} MXN/L<br>"
                                        f"Invoice: {_iv:.4f} | Freight: {_fr:.4f}<br>"
                                        f"Surcharge: {_sv:.4f} | Dist Pen: +{_dp:.2f}<br>"
                                        f"Risk x: {_rm:.3f} | Reliability: {_rels[i]:.3f}<br>"
                                        f"Score: {_scores[i]:.4f}<br>"
                                        f"Total: MXN {_totals[i]:,.0f}<extra></extra>"),
                                ))

                            fig_sc.update_layout(height=420,
                                paper_bgcolor=_c_bg, plot_bgcolor=_c_card,
                                font=dict(color=_c_text, size=11),
                                xaxis=dict(title="Distance (km)", color=_c_gray, gridcolor="#2E2E2A", zeroline=False),
                                yaxis=dict(title="Risk-adj landed cost (MXN/L)", color=_c_gray, gridcolor="#2E2E2A", zeroline=False),
                                margin=dict(t=20, b=50, l=60, r=20),
                                showlegend=False, hovermode="closest")
                            st.plotly_chart(fig_sc, use_container_width=True)

                        except ImportError:
                            st.info("pip install plotly to enable charts")


                    # ── TAB: HISTORY ───────────────────────────────────────
                    with tab_hist:
                        history = st.session_state.rec_history
                        if history:
                            st.dataframe(pd.DataFrame([{
                                "Date":     h["date"],
                                "Station":  h["station"],
                                "Product":  h["product"],
                                "Qty (L)":  f"{h['qty']:,}",
                                "Supplier": h["supplier"],
                                "Landed":   f"{h['landed']:.4f}",
                                "Total MXN":f"{int(h['total']):,}",
                                "Saved vs #2": f"MXN {int(h['savings2']):,}",
                                "PEMEX %":  f"{h['pemex_pct']:.1f}%",
                                "PEMEX Forced": "Yes" if h["pemex_forced"] else "No",
                            } for h in history]), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No recommendations made yet this session.")

                    # ── TAB: LIVE RISK ─────────────────────────────────────
                    with tab_risk:
                        st.markdown("#### 🌐 Live External Risk Multipliers")
                        st.caption(
                            "Each multiplier reflects a live external signal that can inflate the true landed cost "
                            "beyond invoice + freight + surcharge. Sourced from RBOB futures, OpenWeatherMap, "
                            "USD/MXN FX rate, NewsAPI terminal disruptions, and static regulatory scores."
                        )

                        _risk_factor_labels = {
                            "phi_market":     ("📈 Market",      "RBOB futures volatility"),
                            "phi_weather":    ("🌦 Weather",     "Storm / rain at origin state"),
                            "phi_finance":    ("💱 FX Rate",     "USD/MXN spike above 20.50"),
                            "phi_logistics":  ("🚛 Logistics",   "Terminal disruption news"),
                            "phi_regulatory": ("📋 Regulatory",  "ESG / PEMEX compliance score"),
                            "phi_infra":      ("🏗 Infra",       "Pipeline / rail status"),
                            "total_risk_mult":("⚡ TOTAL MULT",  "Combined risk multiplier"),
                        }

                        _risk_api = LiveRiskAPI()

                        with st.spinner("Fetching live risk signals..."):
                            _risk_rows = []
                            for _, row in cands.iterrows():
                                sup  = row.get("supplier", "?")
                                state_val   = str(row.get("state", "Nuevo Leon"))
                                terminal_val = str(row.get("terminal_name", sel_station))
                                coeffs = _risk_api.get_coefficients(
                                    supplier=sup, state=state_val,
                                    terminal=terminal_val, contract_type="Branded"
                                )
                                coeffs["Supplier"] = sup
                                _risk_rows.append(coeffs)

                        if _risk_rows:
                            # HTML risk table
                            _rhdrs = ["Supplier"] + list(_risk_factor_labels.keys())
                            _rtbl = '<table style="width:100%;border-collapse:collapse;font-size:12px">'
                            _rtbl += '<thead><tr>'
                            for h in _rhdrs:
                                lbl = _risk_factor_labels[h][0] if h in _risk_factor_labels else h
                                _rtbl += (f'<th style="text-align:left;padding:6px 10px;color:#6B7280;'
                                          f'font-size:10px;text-transform:uppercase;letter-spacing:0.1em;'
                                          f'border-bottom:1px solid #2E2E2A">{lbl}</th>')
                            _rtbl += '</tr></thead><tbody>'
                            for i, r in enumerate(_risk_rows):
                                _bg = "#1A2910" if r["Supplier"] == winner_sup else "transparent"
                                _rtbl += f'<tr style="background:{_bg}">'
                                for h in _rhdrs:
                                    val = r.get(h, "")
                                    if h == "Supplier":
                                        _rtbl += f'<td style="padding:6px 10px;color:#F5F5F0;font-weight:600">{val}</td>'
                                    elif h == "total_risk_mult":
                                        _tc = "#EF4444" if float(val) > 1.20 else "#F59E0B" if float(val) > 1.10 else "#A3E635"
                                        _rtbl += (f'<td style="padding:6px 10px;color:{_tc};font-weight:700;'
                                                  f'font-family:monospace">{val:.4f}</td>')
                                    else:
                                        _fc = "#F59E0B" if float(val) > 1.08 else "#9CA3AF"
                                        _rtbl += f'<td style="padding:6px 10px;color:{_fc};font-family:monospace">{val:.3f}</td>'
                                _rtbl += '</tr>'
                            _rtbl += '</tbody></table>'
                            st.markdown(_rtbl, unsafe_allow_html=True)

                            st.markdown("")
                            # Legend
                            _leg = '<div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:8px">'
                            for key, (lbl, desc) in _risk_factor_labels.items():
                                if key != "total_risk_mult":
                                    _leg += (f'<div style="font-size:11px;color:#4B5563">'
                                             f'<span style="color:#9CA3AF">{lbl}</span> — {desc}</div>')
                            _leg += '</div>'
                            st.markdown(_leg, unsafe_allow_html=True)

                            st.markdown("")
                            st.caption("🟡 Values above 1.08 are elevated (yellow). 🔴 Total multiplier above 1.20 is high risk (red). "
                                       "Multipliers reflect current real-world conditions — re-run Recommend to refresh.")


            elif get_rec:
                st.error("No normalized data found. Run stages 1–3 first.")

# ── STAGE 6: AGENTIC MONITOR ────────────────────────────────────────────
elif active == 6:
    st.header("🤖 Procurement Agent")
    st.markdown(
        "Ask anything about your procurement data, drop a surcharge email, "
        "or request a recommendation — the agent uses your pipeline data + LLM to answer."
    )

    # ── Chat history ─────────────────────────────────────────────────
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # ── File upload zone ─────────────────────────────────────────────
    with st.expander("📎 Attach files (surcharge notices, price files, etc.)", expanded=False):
        _agent_files = st.file_uploader("Drop files for the agent to process",
                                        key="agent_file_upload",
                                        accept_multiple_files=True,
                                        type=["csv","xlsx","xls","html","htm","txt","pdf"])
        if _agent_files:
            for _agent_file in _agent_files:
                _save_path = INBOX / _agent_file.name
                _save_path.write_bytes(_agent_file.getvalue())
                _ftype = detect_file_type(Path(_agent_file.name))
                _badge = "🔔 Surcharge/Notice" if _ftype == "surcharge" else "📊 Price Data"
                st.success(f"📎 {_agent_file.name} → {_badge} (saved to inbox)")

    # ── Display chat history ─────────────────────────────────────────
    for msg in st.session_state.agent_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"""
            <div style="background:#2A2A26;border:1px solid #3A3A36;border-radius:10px;
                        padding:12px 16px;margin:8px 0;margin-left:60px">
              <div style="font-size:10px;color:#6B7280;margin-bottom:4px">You</div>
              <div style="font-size:13px;color:#F5F5F0">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#1A2910;border:1px solid #365314;border-radius:10px;
                        padding:12px 16px;margin:8px 0;margin-right:60px">
              <div style="font-size:10px;color:#A3E635;margin-bottom:4px">🤖 Agent</div>
              <div style="font-size:13px;color:#D1D5DB">{content}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────────────
    _agent_input = st.text_input(
        "Ask the agent...",
        placeholder='e.g. "Best supplier for Regular at CapitalGas Floresta Plus today?" or "Parse the surcharge email I just uploaded"',
        key="agent_chat_input",
    )

    if _agent_input and st.button("Send", type="primary", use_container_width=True, key="agent_send"):
        st.session_state.agent_messages.append({"role": "user", "content": _agent_input})

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.session_state.agent_messages.append({
                "role": "agent",
                "content": "❌ ANTHROPIC_API_KEY not set in .env — I need LLM access to answer questions."
            })
            st.rerun()
        else:
            with st.spinner("Agent thinking..."):
                # Build context from pipeline data
                _context_parts = []

                # Normalized data summary
                _norm = load_normalized()
                if not _norm.empty:
                    _suppliers = _norm["supplier"].unique().tolist()
                    _stations_count = _norm["id_tienda"].nunique() if "id_tienda" in _norm.columns else 0
                    _date_range = f"{_norm['date'].min()} to {_norm['date'].max()}" if "date" in _norm.columns else "unknown"
                    _avg_landed = _norm["landed_cost"].mean() if "landed_cost" in _norm.columns else 0

                    _context_parts.append(
                        f"NORMALIZED DATA: {len(_norm)} rows, {_stations_count} stations, "
                        f"suppliers: {_suppliers}, date range: {_date_range}, "
                        f"avg landed cost: {_avg_landed:.4f} MXN/L"
                    )

                    # Station list for reference
                    if "id_tienda" in _norm.columns:
                        _station_ids = sorted(_norm["id_tienda"].unique().tolist())
                        _context_parts.append(f"AVAILABLE STATIONS: {_station_ids[:20]}{'...' if len(_station_ids)>20 else ''}")

                    # Per-supplier per-product average landed costs
                    _lc_col = "landed_cost" if "landed_cost" in _norm.columns else "actual_landed" if "actual_landed" in _norm.columns else None
                    if _lc_col and "product_type" in _norm.columns:
                        for prod in ["Regular", "Premium", "Diesel"]:
                            _prod_data = _norm[_norm["product_type"] == prod]
                            if not _prod_data.empty:
                                _best = _prod_data.groupby("supplier")[_lc_col].mean().sort_values().head(5)
                                _context_parts.append(
                                    f"  {prod} avg landed: " +
                                    ", ".join(f"{s}: {v:.4f} MXN/L" for s, v in _best.items())
                                )

                    # If the user message mentions a specific station, include detailed data for it
                    _user_msg = st.session_state.agent_messages[-1]["content"] if st.session_state.agent_messages else ""
                    _mentioned_stations = [sid for sid in _norm["id_tienda"].unique() if sid.lower() in _user_msg.lower()]
                    if not _mentioned_stations and "id_tienda" in _norm.columns:
                        # Try matching station names
                        _tiendas = load_tiendas()
                        if not _tiendas.empty and "nombre" in _tiendas.columns:
                            for _, _t in _tiendas.iterrows():
                                if str(_t.get("nombre","")).lower() in _user_msg.lower():
                                    _mentioned_stations.append(_t["id_tienda"])

                    for _ms in _mentioned_stations[:3]:
                        _st_data = _norm[_norm["id_tienda"] == _ms]
                        if not _st_data.empty:
                            _context_parts.append(f"\nDETAILED DATA FOR STATION {_ms}:")
                            for _prod in _st_data["product_type"].unique():
                                _pd = _st_data[_st_data["product_type"]==_prod].sort_values("date", ascending=False).drop_duplicates("supplier")
                                for _, _r in _pd.iterrows():
                                    _lc = _r.get(_lc_col, _r.get("price_mxn_per_l", 0))
                                    _dist = _r.get("dist_km", "?")
                                    _freight = _r.get("freight_mxn_per_l", _r.get("freight_cost_mxn", 0))
                                    _context_parts.append(
                                        f"  {_prod} | {_r['supplier']} | invoice={_r.get('price_mxn_per_l',0):.4f} | "
                                        f"freight={_freight} | landed={_lc:.4f} | dist={_dist}km | date={_r.get('date','?')}"
                                    )
                else:
                    _context_parts.append("NORMALIZED DATA: NOT AVAILABLE. User needs to run Stage 1 (Ingest) and Stage 2 (Normalize) first.")

                # Surcharges
                _sur_path = PROCESSED / "surcharges.csv"
                if _sur_path.exists():
                    _sur = pd.read_csv(_sur_path)
                    if not _sur.empty:
                        _context_parts.append(f"ACTIVE SURCHARGES: {len(_sur)} events")
                        for _, _sr in _sur.iterrows():
                            _context_parts.append(
                                f"  {_sr.get('supplier','?')} +{_sr.get('surcharge_per_l',0):.2f} MXN/L "
                                f"on {_sr.get('product','?')} at {_sr.get('terminal','?')} "
                                f"({_sr.get('effective_from','?')} → {_sr.get('effective_to','?')})"
                            )

                # PEMEX compliance summary — MONTHLY, not overall
                _hist = load_hist()
                if not _hist.empty and "es_pemex" in _hist.columns:
                    _hist["fecha_pedido"] = pd.to_datetime(_hist["fecha_pedido"], errors="coerce")
                    _total_orders = len(_hist)
                    _pemex_orders = len(_hist[_hist["es_pemex"] == True])
                    _pemex_overall = (_pemex_orders / _total_orders * 100) if _total_orders > 0 else 0
                    _context_parts.append(
                        f"PEMEX COMPLIANCE: Overall historical average is {_pemex_overall:.1f}% "
                        f"({_pemex_orders:,} / {_total_orders:,}). "
                        f"IMPORTANT: The 50% minimum is checked PER STATION, PER PRODUCT, PER MONTH. "
                        f"For future dates (like April 2026) where historico has no data, PEMEX compliance is 0% "
                        f"and PEMEX should be forced as the first order of the month. "
                        f"Historical data ends at {_hist['fecha_pedido'].max().strftime('%Y-%m-%d') if not _hist.empty else 'unknown'}."
                    )

                # Inbox files
                _inbox_files = [f.name for f in INBOX.iterdir() if f.is_file()] if INBOX.exists() else []
                if _inbox_files:
                    _context_parts.append(f"INBOX FILES: {_inbox_files}")

                # Check if user uploaded files
                _attached_file = None
                _attached_content = ""
                if _agent_files:
                    _file_names = [f.name for f in _agent_files]
                    _context_parts.append(f"ATTACHED FILES: {_file_names}")
                    # Use the first file for content preview
                    _agent_file = _agent_files[0]
                    _attached_file = _agent_file.name
                    try:
                        _attached_content = _agent_file.getvalue().decode("utf-8", errors="replace")[:3000]
                        _context_parts.append(f"First file content preview:\n{_attached_content}")
                    except Exception:
                        _context_parts.append(f"ATTACHED FILE: {_attached_file} (binary, cannot preview)")

                _context = "\n".join(_context_parts)

                # ── Auto-rank if the user seems to be asking for a recommendation ──
                # Try to extract station ID and product from the user message
                _user_msg = st.session_state.agent_messages[-1]["content"]
                _rank_result = None
                _tiendas = load_tiendas()

                # Find station ID in message
                _found_station = None
                _found_product = "Regular"  # default
                _found_date = date.today()

                # Match OXG-XXXXX pattern
                import re
                _station_match = re.search(r'(OXG-[A-Za-z0-9]+)', _user_msg)
                if _station_match:
                    _found_station = _station_match.group(1)

                # Match station name to ID
                if not _found_station and not _tiendas.empty and "nombre" in _tiendas.columns:
                    for _, _t in _tiendas.iterrows():
                        if str(_t.get("nombre", "")).lower() in _user_msg.lower():
                            _found_station = _t["id_tienda"]
                            break

                # Match product
                for _p in ["Regular", "Premium", "Diesel"]:
                    if _p.lower() in _user_msg.lower():
                        _found_product = _p
                        break

                # Match date
                _date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', _user_msg)
                if not _date_match:
                    _date_match = re.search(r'(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', _user_msg)
                if _date_match:
                    try:
                        _found_date = pd.to_datetime(_date_match.group(0), dayfirst=False).date()
                    except Exception:
                        pass

                # Run rank_live if we found a station and message looks like a recommendation request
                _rec_keywords = ["best", "recommend", "supplier", "cheapest", "order", "which", "rank"]
                if _found_station and any(k in _user_msg.lower() for k in _rec_keywords):
                    _rank_result, _rank_err = rank_live(_found_station, _found_product, 30000, _found_date)
                    if _rank_result:
                        _context += "\n\nLIVE RANKING RESULT (computed using the same engine as the Recommend page):\n"
                        _context += f"Station: {_found_station} | Product: {_found_product} | Qty: 30,000L | Date: {_found_date}\n"
                        _context += f"PEMEX compliance: {_rank_result['pemex_pct']}% | Forced: {_rank_result['pemex_forced']}\n"
                        if _rank_result["excluded"]:
                            _context += f"Excluded by constraints: {_rank_result['excluded']}\n"
                        _context += "Ranked suppliers (best first):\n"
                        for _i, _r in enumerate(_rank_result["ranked"], 1):
                            _context += (
                                f"  #{_i} {_r['supplier']}: landed={_r['landed']:.4f} MXN/L "
                                f"(invoice={_r['invoice']:.4f} + freight={_r['freight']:.4f} + "
                                f"surcharge={_r['surcharge']:.4f} + dist_penalty={_r['dist_penalty']:.2f}) "
                                f"× risk={_r['risk_mult']:.3f} | dist={_r['dist_km']}km | "
                                f"reliability={_r['reliability']:.3f} | score={_r['final_score']:.4f} | "
                                f"total={_r['total_cost']:,.0f} MXN\n"
                            )
                        _context += "\nIMPORTANT: Present these EXACT numbers to the user. Do NOT recalculate. These include distance penalty and risk multiplier."
                    elif _rank_err:
                        _context += f"\n\nRANKING ERROR: {_rank_err}"

                # Also add station name mapping
                if not _tiendas.empty and "nombre" in _tiendas.columns and "id_tienda" in _tiendas.columns:
                    _name_map = dict(zip(_tiendas["id_tienda"], _tiendas["nombre"]))
                    _context += f"\n\nSTATION NAME MAPPING (first 20): {dict(list(_name_map.items())[:20])}"

                _system_prompt = f"""You are the AI procurement agent for CapitalGas, a fuel station operator in Mexico with 617 stations.
You have access to the following live pipeline data:

{_context}

Your capabilities:
1. RECOMMEND — When a LIVE RANKING RESULT is provided above, present those EXACT numbers. Do NOT recalculate or estimate. The ranking already includes invoice + freight + surcharge + distance penalty × risk multiplier + weighted scoring (60% cost + 25% reliability + 15% volatility). Just explain the result clearly.
2. PARSE SURCHARGE — When the user shares or uploads a surcharge email/notice, extract: supplier, terminal, product, amount (MXN/L), effective dates, reason.
3. ANALYZE — Answer questions about PEMEX compliance, supplier comparisons, cost trends.
4. STATION LOOKUP — Use the station name mapping to resolve names to IDs.

Rules:
- For recommendations, ALWAYS use the LIVE RANKING RESULT if available — never do your own math
- Show the full breakdown: invoice + freight + surcharge + distance penalty, then × risk multiplier = landed cost
- PEMEX compliance is per-station, per-product, per-month (50% minimum)
- For April 2026 or any date beyond Jan 2026, PEMEX compliance will be 0% because historico_pedidos ends Jan 2026
- Never mention "Claude" — say "procurement agent" or "I"
- Be concise and direct"""

                try:
                    import requests as _req
                    _resp = _req.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1000,
                            "system": _system_prompt,
                            "messages": [
                                {"role": m["role"] if m["role"] == "user" else "assistant",
                                 "content": m["content"]}
                                for m in st.session_state.agent_messages
                            ],
                        },
                        timeout=60,
                    )

                    if _resp.status_code == 200:
                        _answer = _resp.json()["content"][0]["text"]

                        # If the agent parsed a surcharge, try to auto-save it
                        if _attached_file and ("surcharge" in _agent_input.lower() or "parse" in _agent_input.lower()):
                            try:
                                from src.ingestion.parse_llm import UnstructuredParser
                                _parser = UnstructuredParser()
                                if _attached_content:
                                    _events = _parser.parse_text(_attached_content, source_label=_attached_file)
                                else:
                                    _events = _parser.parse(str(INBOX / _attached_file))
                                if _events:
                                    _parser.save_events(_events)
                                    _answer += f"\n\n✅ **{len(_events)} surcharge event(s) saved to surcharges.csv.** Re-run Normalize (Stage 2) to apply them to landed costs."
                            except Exception as _parse_err:
                                _answer += f"\n\n⚠ Could not auto-save surcharge: {_parse_err}"

                        st.session_state.agent_messages.append({"role": "agent", "content": _answer})
                    else:
                        st.session_state.agent_messages.append({
                            "role": "agent",
                            "content": f"❌ LLM API error: {_resp.status_code}"
                        })
                except Exception as _e:
                    st.session_state.agent_messages.append({
                        "role": "agent",
                        "content": f"❌ Could not reach LLM API: {_e}"
                    })

            st.rerun()

    # ── Quick action buttons ─────────────────────────────────────────
    st.divider()
    st.markdown("**Quick actions:**")
    _qa1, _qa2, _qa3 = st.columns(3)
    with _qa1:
        if st.button("📊 Supplier comparison", use_container_width=True, key="qa_compare"):
            st.session_state.agent_messages.append({
                "role": "user",
                "content": "Compare all available suppliers for Regular fuel at Monterrey stations. Show avg landed cost, distance, and reliability for each."
            })
            st.session_state["agent_chat_input"] = ""
            st.rerun()
    with _qa2:
        if st.button("🏛️ PEMEX compliance check", use_container_width=True, key="qa_pemex"):
            st.session_state.agent_messages.append({
                "role": "user",
                "content": "What is our current PEMEX compliance status across all Monterrey stations? Are any stations at risk of falling below the 50% monthly minimum?"
            })
            st.rerun()
    with _qa3:
        if st.button("⚡ Active surcharges", use_container_width=True, key="qa_surcharges"):
            st.session_state.agent_messages.append({
                "role": "user",
                "content": "Show me all active surcharges and which stations/suppliers they affect. How much is each surcharge costing us per order?"
            })
            st.rerun()

    # Clear chat
    if st.session_state.agent_messages:
        if st.button("🗑️ Clear conversation", key="clear_agent_chat"):
            st.session_state.agent_messages = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  AGENT AUTO-POLL (non-blocking)
# ══════════════════════════════════════════════════════════════════════════
# Scans inbox for new files on each render when enabled
if st.session_state.agent_enabled:
    _last_scan_key = "_agent_last_scan"
    _now = time.time()
    _last = st.session_state.get(_last_scan_key, 0)
    _interval = st.session_state.agent_poll_interval

    if _now - _last >= _interval:
        st.session_state[_last_scan_key] = _now
        _scan_inbox()
        if st.session_state.agent_alerts:
            st.rerun()