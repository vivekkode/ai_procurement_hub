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
CAPITALGAS = ROOT / "data" / "capitalgas" / "outputs"
RAW        = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
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

# ── file-based unlock ────────────────────────────────────────────────────
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
    if n == 2: return f["ingest"]
    if n == 3: return f["normalize"]
    if n == 4: return f["engine"]
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
        {"n":3,"icon":"⚙️","title":"Run Engine",         "done":files["engine"]},
        {"n":4,"icon":"💡","title":"Recommend",          "done":False},
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
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.active_stage = 0
        st.rerun()

    st.caption("Globant · University at Buffalo")

    with st.expander("🐛 Debug", expanded=False):
        st.json({
            "active_stage": st.session_state.active_stage,
            "files_on_disk": check_files(),
            "ROOT": str(ROOT),
            "PROCESSED": str(PROCESSED),
        })

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
            if f["engine"]:
                if st.button("📊 Load Existing Results", use_container_width=True):
                    st.session_state.active_stage = 4
                    st.rerun()
            elif f["ingest"]:
                next_stage = 2 if not f["normalize"] else 3
                if st.button("↩ Resume Pipeline", use_container_width=True):
                    st.session_state.active_stage = next_stage
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
            with st.spinner("Sending to Claude for extraction..."):
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

                        st.info("💡 Now go to **Normalize** and re-run — the surcharge will be included in the landed cost calculation automatically.")

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

        if st.button("🔗 Run Normalization", type="primary", use_container_width=True):
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
                if st.button("→ Continue to Engine", type="primary", use_container_width=True):
                    st.session_state.active_stage = 3
                    st.rerun()
            else:
                st.error("Normalization did not produce normalized_suppliers.csv")
                with st.expander("stdout"): st.code(result.stdout[-3000:] or "(empty)")
                with st.expander("stderr"):  st.code(result.stderr[-3000:] or "(empty)")

        if check_files()["normalize"]:
            ndf = load_normalized()
            st.info(f"Previous normalization found: {len(ndf):,} rows on disk.")
            if st.button("→ Continue to Engine (use existing)", type="secondary"):
                st.session_state.active_stage = 3
                st.rerun()

# ── STAGE 3: ENGINE ──────────────────────────────────────────────────────
elif active == 3:
    st.header("⚙️ Stage 3 — Run Engine")
    st.markdown(
        "Run the procurement rules engine. Enforces PEMEX 50% minimum, MOQ, budget ceiling, "
        "and delivery deadlines. Ranking: 60% landed cost + 25% reliability + 15% volatility."
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
    st.header("💡 Stage 4 — Recommend")
    st.markdown(
        "Select a station and product. Get ranked suppliers with full landed cost "
        "breakdown, PEMEX compliance status, and an AI explanation."
    )

    if not stage_unlocked(4):
        st.warning("🔒 Complete Stage 3 first.")
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
                # ── Compute candidates (logic untouched) ──────────────────
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
                    sur_col = cands["surcharge_mxn_per_l"].fillna(0) if "surcharge_mxn_per_l" in cands.columns else pd.Series(0, index=cands.index)
                    cands["actual_freight_per_l"] = (cands["freight_cost_mxn"] / order_qty).round(6) if "freight_cost_mxn" in cands.columns else pd.Series(0, index=cands.index)
                    cands["actual_landed"]        = (cands["price_mxn_per_l"] + cands["actual_freight_per_l"] + sur_col).round(4)
                    cands["total_cost"]           = (cands["actual_landed"] * order_qty).round(0)
                    cands = cands.sort_values("actual_landed")

                    pemex_pct = 0.0
                    if not hist.empty and "id_tienda" in hist.columns:
                        ots  = pd.Timestamp(order_dt)
                        msta = ots.replace(day=1)
                        mo   = hist[
                            (hist["id_tienda"] == sel_station) &
                            (hist["fecha_pedido"] >= msta) &
                            (hist["fecha_pedido"] < ots) &
                            (hist.get("producto", pd.Series()) == product.lower() if "producto" in hist.columns else True)
                        ]
                        if len(mo) > 0:
                            tot_l = mo["litros_pedidos"].sum() if "litros_pedidos" in mo.columns else 0
                            pmx_l = mo[mo["es_pemex"]==True]["litros_pedidos"].sum() if "es_pemex" in mo.columns else 0
                            pemex_pct = (pmx_l/tot_l*100) if tot_l > 0 else 0.0

                    pemex_forced = pemex_pct < 50.0
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

                    # ── Three tabs: Results | Charts | History ─────────────
                    tab_res, tab_charts, tab_hist = st.tabs(["📋 Results", "📊 Charts", "🕘 History"])

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

                        # Cost breakdown table — compact
                        cmp = pd.DataFrame([{
                            "Supplier":  row["supplier"],
                            "Invoice":   f"{float(row['price_mxn_per_l']):.4f}",
                            "Freight":   f"{float(row['actual_freight_per_l']):.4f}",
                            "Surcharge": f"{float(sur_col.loc[row.name] if row.name in sur_col.index else 0):.4f}",
                            "Landed":    f"{float(row['actual_landed']):.4f}",
                            "Total MXN": f"{int(row['total_cost']):,}",
                            "vs Best":   "✓ best" if float(row["actual_landed"])==best_lc else f"+MXN {(float(row['actual_landed'])-best_lc)*order_qty:,.0f}",
                        } for _,row in cands.iterrows()])
                        st.dataframe(cmp, use_container_width=True, hide_index=True)

                        # PEMEX compliance bar
                        bc  = "#EF4444" if pemex_pct<50 else "#A3E635"
                        stx = "⚠ Below minimum — PEMEX order required" if pemex_pct<50 else "✓ Monthly minimum satisfied"
                        st.markdown(f"""
                        <div style="margin-top:14px">
                          <div style="font-size:11px;font-weight:700;color:#4B5563;letter-spacing:0.14em;
                                      text-transform:uppercase;margin-bottom:8px">PEMEX Compliance — This Month</div>
                          <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                            <span style="font-size:22px;font-weight:700;color:{bc};font-family:monospace">{pemex_pct:.1f}%</span>
                            <span style="font-size:12px;font-weight:600;color:{bc};margin-top:6px">{stx}</span>
                          </div>
                          <div style="background:#2E2E2A;border-radius:4px;height:8px;overflow:hidden">
                            <div style="height:8px;width:{min(pemex_pct*2,100):.0f}%;background:{bc};border-radius:4px"></div>
                          </div>
                          <div style="font-size:10px;color:#4B5563;margin-top:5px;font-family:monospace">
                            {product} · {sel_station} · 50% monthly minimum required by Mexican law
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
                                f"Top supplier: {winner_sup} at {best_lc:.4f} MXN/L landed "
                                f"(invoice {float(best.get('price_mxn_per_l',0)):.4f} + "
                                f"freight {float(best.get('actual_freight_per_l',0)):.4f} + "
                                f"surcharge {float(sur_col.loc[best.name] if best.name in sur_col.index else 0):.4f}).\n"
                                f"Total: MXN {best_tot:,.0f} | Saves MXN {savings2:,.0f} vs next best.\n"
                                f"PEMEX month-to-date: {pemex_pct:.1f}% (50% min required by law).\n"
                                f"Explain in 3 sentences why this is the best choice and any compliance notes."
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
                                st.warning(f"Could not reach Claude API: {e}")
                        else:
                            st.info("Set ANTHROPIC_API_KEY in .env for AI explanations.")

                    # ── TAB: CHARTS ────────────────────────────────────────
                    with tab_charts:
                        try:
                            import plotly.graph_objects as go

                            sups_list = cands["supplier"].tolist()

                            # Chart A — Cost composition stacked bar
                            st.markdown("##### How does each supplier's cost break down?")
                            st.caption("Stacked bars show invoice + freight + surcharge. The lime diamond marks the final landed cost. A low invoice does not always mean a low landed cost.")
                            fig_a = go.Figure()
                            fig_a.add_trace(go.Bar(
                                name="Invoice price", x=sups_list,
                                y=cands["price_mxn_per_l"].tolist(),
                                marker_color="#4A8FBF",
                                hovertemplate="<b>%{x}</b><br>Invoice: %{y:.4f} MXN/L<extra></extra>",
                            ))
                            fig_a.add_trace(go.Bar(
                                name="Freight cost", x=sups_list,
                                y=cands["actual_freight_per_l"].tolist(),
                                marker_color="#F59E0B",
                                hovertemplate="<b>%{x}</b><br>Freight: %{y:.4f} MXN/L<extra></extra>",
                            ))
                            fig_a.add_trace(go.Bar(
                                name="Surcharge", x=sups_list,
                                y=[float(sur_col.loc[r] if r in sur_col.index else 0) for r in cands.index],
                                marker_color="#EF4444",
                                hovertemplate="<b>%{x}</b><br>Surcharge: %{y:.4f} MXN/L<extra></extra>",
                            ))
                            fig_a.add_trace(go.Scatter(
                                name="Landed cost", x=sups_list,
                                y=cands["actual_landed"].tolist(),
                                mode="markers+text",
                                marker=dict(symbol="diamond", size=12, color="#A3E635",
                                           line=dict(color="#1C1C1A", width=1)),
                                text=[f"{v:.4f}" for v in cands["actual_landed"].tolist()],
                                textposition="top center",
                                textfont=dict(color="#A3E635", size=11, family="monospace"),
                                hovertemplate="<b>%{x}</b><br>Landed: %{y:.4f} MXN/L<extra></extra>",
                            ))
                            # Highlight winner bar with a border
                            winner_idx = sups_list.index(best.get("supplier","")) if best.get("supplier","") in sups_list else -1
                            fig_a.update_layout(
                                barmode="stack",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#9CA3AF", size=12),
                                height=340,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                           font=dict(color="#9CA3AF"), bgcolor="rgba(0,0,0,0)"),
                                xaxis=dict(gridcolor="#2E2E2A", color="#9CA3AF", showgrid=False,
                                          title="Supplier"),
                                yaxis=dict(gridcolor="#2E2E2A", color="#9CA3AF",
                                          title="Cost (MXN per litre)", zeroline=False),
                                margin=dict(t=50, b=40, l=50, r=20),
                            )

                            st.plotly_chart(fig_a, use_container_width=True)

                            st.divider()

                            # Chart C — Distance vs Landed Cost scatter
                            st.markdown("##### How does distance from the station affect the final cost?")
                            st.caption("Each bubble is one supplier. X = distance to your station. Y = landed cost. Bubble size = total order value. A cheap invoice from a distant supplier gets eroded by freight.")
                            dists   = [float(r.get("dist_km", 0)) for _, r in cands.iterrows()]
                            landeds = cands["actual_landed"].tolist()
                            totals  = cands["total_cost"].tolist()
                            winner  = best.get("supplier", "")
                            max_t   = max(totals) if totals else 1
                            fig_c   = go.Figure()
                            for sup, dist, lc, tot in zip(sups_list, dists, landeds, totals):
                                is_win = sup == winner
                                fig_c.add_trace(go.Scatter(
                                    x=[dist], y=[lc],
                                    mode="markers+text",
                                    name=sup,
                                    marker=dict(
                                        size=max(16, int(tot / max_t * 56)),
                                        color="#A3E635" if is_win else "#4A8FBF",
                                        opacity=0.9 if is_win else 0.7,
                                        line=dict(color="#1C1C1A", width=2),
                                    ),
                                    text=[sup],
                                    textposition="top center",
                                    textfont=dict(color="#F5F5F0" if is_win else "#9CA3AF", size=12),
                                    hovertemplate=(
                                        f"<b>{sup}</b><br>"
                                        f"Distance: {dist:.1f} km<br>"
                                        f"Landed cost: {lc:.4f} MXN/L<br>"
                                        f"Order total: MXN {int(tot):,}<extra></extra>"
                                    ),
                                ))

                            fig_c.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#9CA3AF", size=12),
                                showlegend=False, height=340,
                                xaxis=dict(gridcolor="#2E2E2A", color="#9CA3AF",
                                          title="Distance from station (km)", zeroline=False),
                                yaxis=dict(gridcolor="#2E2E2A", color="#9CA3AF",
                                          title="Landed cost (MXN/L)", zeroline=False),
                                margin=dict(t=20, b=50, l=50, r=20),
                            )
                            st.plotly_chart(fig_c, use_container_width=True)

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

            elif get_rec and norm.empty:
                st.error("No normalized data found. Run stages 1–3 first.")