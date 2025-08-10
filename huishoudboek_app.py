import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
import calendar

# ============================================================
# 🔧 Pagina-instellingen
# ============================================================
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ============================================================
# 📅 Maanden in het Nederlands
# ============================================================
MAANDEN_NL = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December"
]
maand_type = CategoricalDtype(categories=MAANDEN_NL, ordered=True)

# ============================================================
# 💶 Helpers
# ============================================================
def euro(x):
    try:
        return f"€ {x:,.2f}"
    except Exception:
        return "€ 0,00"


def pct(value, total, *, signed=False, absolute=False):
    if total is None or total == 0 or pd.isna(total):
        return "—"
    num = abs(value) if absolute else value
    p = (num / total) * 100
    return f"{p:+.1f}%" if signed else f"{p:.1f}%"


def _clamp(x, lo=0.0, hi=1.0):
    try:
        return float(min(max(x, lo), hi))
    except Exception:
        return np.nan

def _safe_div(a, b):
    return np.nan if (b is None or b == 0 or pd.isna(b)) else a / b


# ============================================================
# 📥 Data inladen (met optionele upload)
# ============================================================
with st.sidebar:
    upload = st.file_uploader("📥 Laad Excel (optioneel)", type=["xlsx", "xlsm"], key="upload_main")

@st.cache_data(show_spinner=False)
def laad_data(pad=None, file=None):
    src = file if file is not None else (pad or "huishoud.xlsx")
    df = pd.read_excel(src, sheet_name="Data", engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower()

    verplicht = ["datum", "bedrag", "categorie"]
    ontbreekt = [k for k in verplicht if k not in df.columns]
    if ontbreekt:
        raise ValueError(f"Ontbrekende kolommen: {', '.join(ontbreekt)}")

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df["bedrag"] = pd.to_numeric(df["bedrag"], errors="coerce")
    df["categorie"] = df["categorie"].astype(str).str.strip().str.title()

    if "vast/variabel" not in df.columns:
        df["vast/variabel"] = "Onbekend"
    df["vast/variabel"] = df["vast/variabel"].astype(str).str.strip().str.title()

    df["maand"] = df["datum"].dt.month
    df["jaar"] = df["datum"].dt.year
    df["maand_naam"] = df["maand"].apply(lambda m: MAANDEN_NL[int(m)-1] if pd.notnull(m) else "")
    df["maand_naam"] = df["maand_naam"].astype(maand_type)

    df = df.dropna(subset=["datum", "bedrag", "categorie"]).copy()
    df = df[df["categorie"].str.strip() != ""]
    return df

try:
    st.info("📁 Data laden…")
    df = laad_data(pad="huishoud.xlsx", file=upload)
    st.success("✅ Data geladen!")
    with st.expander("📄 Voorbeeld van de data"):
        st.write(df.head())
except Exception as e:
    st.error(f"❌ Fout bij het laden: {e}")
    st.stop()

# ============================================================
# 📅 Filters (met Reset)
# ============================================================
with st.sidebar:
    st.header("📅 Filter op periode")
    if "default_start" not in st.session_state:
        st.session_state.default_start = df["datum"].min().date()
        st.session_state.default_end = df["datum"].max().date()
        st.session_state.start_datum = st.session_state.default_start
        st.session_state.eind_datum = st.session_state.default_end

    c1, c2 = st.columns([3, 1])
    with c1:
        start_datum = st.date_input("Van", st.session_state.get("start_datum", st.session_state.default_start), key="date_from")
        eind_datum = st.date_input("Tot", st.session_state.get("eind_datum", st.session_state.default_end), key="date_to")
    with c2:
        if st.button("🔄 Reset", key="reset_btn"):
            st.session_state.start_datum = st.session_state.default_start
            st.session_state.eind_datum = st.session_state.default_end
            st.rerun()

# Filter toepassen
df_filtered = df[(df["datum"] >= pd.to_datetime(start_datum)) & (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_filtered["maand_naam"] = df_filtered["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))
if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# Maandkeuze
aanwezig = set(df_filtered["maand_naam"].dropna().astype(str).tolist())
beschikbare_maanden = [m for m in MAANDEN_NL if m in aanwezig]
default_maand = beschikbare_maanden[-1] if beschikbare_maanden else MAANDEN_NL[0]
with st.sidebar:
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        beschikbare_maanden,
        index=(beschikbare_maanden.index(default_maand) if beschikbare_maanden else 0),
        key="maand_select",
    )

# ============================================================
# 📅 Maand-metrics (saldo)
# ============================================================
st.subheader(f"📆 Overzicht voor {geselecteerde_maand}")
df_maand = df_filtered[df_filtered["maand_naam"] == geselecteerde_maand].copy()

is_loon = df_maand["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon_m = df_maand[is_loon]
df_vast_m = df_maand[df_maand["vast/variabel"] == "Vast"]
df_variabel_m = df_maand[df_maand["vast/variabel"] == "Variabel"]

inkomen_m = df_loon_m["bedrag"].sum()
vast_saldo_m = df_vast_m["bedrag"].sum()
variabel_saldo_m = df_variabel_m["bedrag"].sum()
totaal_saldo_m = inkomen_m + vast_saldo_m + variabel_saldo_m




# ============================================================
# 🔁 Trend t.o.v. vorige maand (delta in st.metric)
# ============================================================
st.caption("Trend t.o.v. vorige maand")

if not df_maand.empty:
    # Laatste datum in de geselecteerde maand
    ref = df_maand["datum"].max()

    # Vorige maand bepalen (met jaarwisseling)
    prev_year, prev_month = (ref.year - 1, 12) if ref.month == 1 else (ref.year, ref.month - 1)

    # Dataframe van de vorige maand binnen de huidige filterrange
    prev_mask = (df_filtered["datum"].dt.year == prev_year) & (df_filtered["datum"].dt.month == prev_month)
    df_prev = df_filtered[prev_mask].copy()

    # Helper om totals te pakken (zelfde logica als je KPI's)
    def total_of(dfin, *, cat=None, vv=None):
        d = dfin.copy()
        cat_col = d["categorie"].astype(str).str.strip().str.lower()
        if cat == "inkomsten":
            d = d[cat_col.eq("inkomsten loon")]
        elif cat == "uitgaven":
            d = d[~cat_col.eq("inkomsten loon")]
        if vv is not None:
            d = d[d["vast/variabel"].astype(str).str.strip().str.title().eq(vv)]
        return d["bedrag"].sum()

    # Waarden vorige maand
    prev_ink = total_of(df_prev, cat="inkomsten")
    prev_vast = total_of(df_prev, vv="Vast")
    prev_var  = total_of(df_prev, vv="Variabel")
    prev_net  = prev_ink + prev_vast + prev_var

    # Deltas (huidig - vorige maand)
    delta_ink = inkomen_m - prev_ink
    delta_vast = vast_saldo_m - prev_vast
    delta_var = variabel_saldo_m - prev_var
    delta_net = totaal_saldo_m - prev_net

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📈 Inkomen (trend)", euro(inkomen_m), delta=euro(delta_ink))
    c2.metric("📌 Vaste kosten (trend)", euro(vast_saldo_m), delta=euro(delta_vast))
    c3.metric("📎 Variabele kosten (trend)", euro(variabel_saldo_m), delta=euro(delta_var))
    c4.metric("💰 Netto saldo maand (trend)", euro(totaal_saldo_m), delta=euro(delta_net))
else:
    st.info("ℹ️ Geen data voor de geselecteerde maand om een trend te tonen.")


# ============================================================
# 📊 Gemiddelde score + Uitgaven/inkomen meter — naast elkaar (met uitleg)
# ============================================================

def _clamp(x, lo=0.0, hi=1.0):
    try:
        return float(min(max(x, lo), hi))
    except Exception:
        return np.nan

def _safe_div(a, b):
    return np.nan if (b is None or b == 0 or pd.isna(b)) else a / b

# ---------- Bereken GEMIDDELDE SIMPELE SCORE (alle maanden) ----------
avg_score = None
fig_avg = None
avg_spaar_pct = np.nan
avg_vaste_pct = np.nan

try:
    scores_all = []
    spaar_pct_list = []
    vaste_pct_list = []

    gb = df.groupby(df["datum"].dt.to_period("M"), sort=True)

    for ym, df_month in gb:
        if df_month.empty:
            continue

        cat = df_month["categorie"].astype(str).str.strip().str.lower()
        is_loon = cat.eq("inkomsten loon")
        inkomen = df_month[is_loon]["bedrag"].sum()
        uitgaven = df_month[~is_loon]["bedrag"].sum()  # meestal negatief

        saldo = inkomen + uitgaven
        sparen_pct = _safe_div(saldo, inkomen)  # 0..1 (kan NaN worden)
        spaar_pct_list.append(sparen_pct * 100 if not pd.isna(sparen_pct) else np.nan)

        vaste_ratio = np.nan
        if "vast/variabel" in df_month.columns:
            vaste_lasten = df_month[
                (df_month["vast/variabel"].astype(str).str.strip().str.title() == "Vast") & (~is_loon)
            ]["bedrag"].sum()
            vaste_ratio = _safe_div(abs(vaste_lasten), abs(inkomen) if inkomen != 0 else np.nan)
        vaste_pct_list.append(vaste_ratio * 100 if not pd.isna(vaste_ratio) else np.nan)

        # Component-scores
        score_sparen = _clamp(sparen_pct / 0.2 if not pd.isna(sparen_pct) else np.nan, 0, 1)
        score_vast = np.nan if pd.isna(vaste_ratio) else (1.0 - _clamp((vaste_ratio - 0.5) / 0.5, 0, 1))

        components = {"Sparen": (score_sparen, 0.5), "Vaste lasten": (score_vast, 0.5)}
        avail = {k: v for k, (v, w) in components.items() if not pd.isna(v)}
        if not avail:
            continue
        total_weight = sum([components[k][1] for k in avail.keys()])
        score_0_1 = sum([components[k][0] * components[k][1] for k in avail.keys()]) / total_weight
        scores_all.append(score_0_1)

    if scores_all:
        avg_score = int(round((sum(scores_all) / len(scores_all)) * 100))
        # Gemiddelden voor uitleg
        if any(not pd.isna(x) for x in spaar_pct_list):
            avg_spaar_pct = float(np.nanmean(spaar_pct_list))
        if any(not pd.isna(x) for x in vaste_pct_list):
            avg_vaste_pct = float(np.nanmean(vaste_pct_list))

        # Gauge
        fig_avg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            number={'suffix': "/100"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.3},
                'steps': [
                    {'range': [0, 50],  'color': '#fca5a5'},
                    {'range': [50, 65], 'color': '#fcd34d'},
                    {'range': [65, 80], 'color': '#a7f3d0'},
                    {'range': [80, 100],'color': '#86efac'},
                ],
            }
        ))
        fig_avg.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
except Exception:
    avg_score = None
    fig_avg = None

# ---------- Bereken UITGAVEN / INKOMEN (%) (alle data) ----------
perc_all = None
fig_exp_all = None

try:
    cat_all = df["categorie"].astype(str).str.strip().str.lower()
    is_loon_all = cat_all.eq("inkomsten loon")

    inkomen_all = df[is_loon_all]["bedrag"].sum()
    uitgaven_all = df[~is_loon_all]["bedrag"].sum()  # meestal negatief

    if not pd.isna(inkomen_all) and abs(inkomen_all) != 0:
        perc_all = float(abs(uitgaven_all) / abs(inkomen_all) * 100.0)
        axis_max = max(120, min(200, (int(perc_all // 10) + 2) * 10))
        fig_exp_all = go.Figure(go.Indicator(
            mode="gauge+number",
            value=perc_all,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, axis_max]},
                'bar': {'thickness': 0.3},
                'steps': [
                    {'range': [0, 33.33],      'color': '#86efac'},  # groen
                    {'range': [33.33, 100],    'color': '#fcd34d'},  # geel
                    {'range': [100, axis_max], 'color': '#fca5a5'},  # rood
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 2},
                    'thickness': 0.75,
                    'value': 100
                },
            }
        ))
        fig_exp_all.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
except Exception:
    perc_all = None
    fig_exp_all = None

# ---------- Toon NAAST ELKAAR ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Gemiddelde financiële gezondheid — alle maanden")
    if fig_avg is not None:
        st.plotly_chart(fig_avg, use_container_width=True)

        # Uitlegregel onder de meter
        uitleg_parts = []
        if not pd.isna(avg_spaar_pct):
            uitleg_parts.append(f"gemiddeld spaarpercentage: {avg_spaar_pct:.1f}%")
        if not pd.isna(avg_vaste_pct):
            uitleg_parts.append(f"gemiddeld aandeel vaste lasten: {avg_vaste_pct:.1f}% van inkomen")
        uitleg_txt = " — ".join(uitleg_parts) if uitleg_parts else "onvoldoende gegevens voor detailuitleg."

        if avg_score >= 80:
            st.success(f"Uitstekend ({avg_score}/100) — {uitleg_txt}")
        elif avg_score >= 65:
            st.info(f"Gezond ({avg_score}/100) — {uitleg_txt}")
        elif avg_score >= 50:
            st.warning(f"Aandacht nodig ({avg_score}/100) — {uitleg_txt}")
        else:
            st.error(f"Kwetsbaar ({avg_score}/100) — {uitleg_txt}")
    else:
        st.info("ℹ️ Geen voldoende gegevens voor de gemiddelde score.")

with col2:
    st.subheader("🎯 Uitgaven t.o.v. inkomen — alle data")
    if fig_exp_all is not None:
        st.plotly_chart(fig_exp_all, use_container_width=True)
        st.caption(
            f"Uitgaven zijn {perc_all:.1f}% van het inkomen. "
            "Groen < 33.3%, geel 33.3–100%, rood ≥ 100%."
        )
    else:
        st.info("ℹ️ Geen inkomen gevonden in alle data, kan geen percentage tonen.")










# ============================================================
# 🔮 Wat-als scenario — simulatie vaste lasten & inkomen
# ============================================================
st.header("🧪 Wat-als scenario")

with st.expander("Pas waarden aan om te simuleren", expanded=False):
    col_a, col_b = st.columns(2)

    # Schuivers voor maandelijkse wijziging
    with col_a:
        extra_vaste = st.slider(
            "Verandering vaste lasten per maand (€)", 
            min_value=-2000, max_value=2000, value=0, step=50,
            help="Negatief = minder kosten, positief = meer kosten"
        )
    with col_b:
        extra_inkomen = st.slider(
            "Verandering inkomen per maand (€)", 
            min_value=-2000, max_value=2000, value=0, step=50,
            help="Negatief = minder inkomen, positief = meer inkomen"
        )

# --- Huidige totale cijfers (uit ALLE data) ---
cat_all = df["categorie"].astype(str).str.strip().str.lower()
is_loon_all = cat_all.eq("inkomsten loon")

inkomen_all = abs(df[is_loon_all]["bedrag"].sum())   # positief
uitgaven_all = abs(df[~is_loon_all]["bedrag"].sum()) # positief

# --- Aantal maanden in dataset ---
maanden_count = df["datum"].dt.to_period("M").nunique()

# --- Simulatie: pas waarden aan ---
sim_inkomen = inkomen_all + (extra_inkomen * maanden_count)
sim_uitgaven = uitgaven_all + (extra_vaste * maanden_count)

# --- Vermijd negatieve inkomsten ---
if sim_inkomen <= 0:
    sim_ratio = None
else:
    sim_ratio = (sim_uitgaven / sim_inkomen) * 100

# --- Gauge grafiek ---
if sim_ratio is not None:
    axis_max = max(120, min(200, (int(sim_ratio // 10) + 2) * 10))
    fig_sim_ratio = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sim_ratio,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, axis_max]},
            'bar': {'thickness': 0.3},
            'steps': [
                {'range': [0, 33.33],      'color': '#86efac'},  # groen
                {'range': [33.33, 100],    'color': '#fcd34d'},  # geel
                {'range': [100, axis_max], 'color': '#fca5a5'},  # rood
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': 100
            },
        }
    ))
    fig_sim_ratio.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sim_ratio, use_container_width=True)

    # Delta t.o.v. huidige situatie
    huidige_ratio = (uitgaven_all / inkomen_all) * 100
    delta_ratio = sim_ratio - huidige_ratio
    st.caption(f"📊 Verandering t.o.v. huidige situatie: {delta_ratio:+.1f}%")
else:
    st.error("⚠️ Simulatie-inkomen is ≤ 0 — ratio niet berekenbaar.")
















# ============================================================
# 📊 Financiële metrics (gehele periode)
# ============================================================
is_loon_all = df_filtered["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_filtered[is_loon_all]
df_vast = df_filtered[df_filtered["vast/variabel"] == "Vast"]
df_variabel = df_filtered[df_filtered["vast/variabel"] == "Variabel"]

inkomen = df_loon["bedrag"].sum()
vast_saldo = df_vast["bedrag"].sum()
variabel_saldo = df_variabel["bedrag"].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Inkomen", euro(inkomen), "—")
c2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo), f"{pct(vast_saldo, inkomen, absolute=True)} van inkomen")
c3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo), f"{pct(variabel_saldo, inkomen, absolute=True)} van inkomen")
c4.metric("💰 Totaal saldo", euro(totaal_saldo), f"{pct(totaal_saldo, inkomen, signed=True)} van inkomen")

# ============================================================
# 🎯 Budgetdoelen per categorie — alleen VASTE kosten
# ============================================================
st.subheader(f"🎯 Budgetdoelen per categorie — {geselecteerde_maand}")

# Alle vaste categorieën (uit hele dataset)
vaste_cats = (
    df[df["vast/variabel"].astype(str).str.strip().str.title().eq("Vast")]["categorie"]
      .astype(str).str.strip().str.title().dropna().unique()
)

# Uitgaven deze maand (alleen vaste) als Series
uitgaven_mnd_ser = (
    df_filtered[
        (df_filtered["maand_naam"] == geselecteerde_maand) &
        (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")) &
        (df_filtered["vast/variabel"].astype(str).str.strip().str.title().eq("Vast"))
    ]
    .groupby("categorie")["bedrag"].sum().abs()
)

# 🔥 NIEUW: automatische budgetten = gemiddelde van voorgaande maanden
# Bepaal referentiemaand (gebruik laatste datum in de geselecteerde maand)
if df_maand.empty:
    gemiddelde_per_cat = pd.Series(dtype=float)
else:
    ref = df_maand["datum"].max()
    maand_start = pd.Timestamp(ref.year, ref.month, 1)

    # Historische vaste uitgaven vóór deze maand
    prev = df[(df["datum"] < maand_start) &
              (df["vast/variabel"].astype(str).str.strip().str.title() == "Vast") &
              (~df["categorie"].astype(str).str.lower().eq("inkomsten loon"))].copy()

    if prev.empty:
        gemiddelde_per_cat = pd.Series(dtype=float)
    else:
        # Som per maand & categorie → gemiddelde per categorie
        per_mnd_cat = prev.groupby([prev["datum"].dt.to_period("M"), "categorie"])['bedrag'].sum().abs()
        gemiddelde_per_cat = per_mnd_cat.groupby("categorie").mean()

# Serie met alle vaste categorieën voor huidige maand (ontbrekend = 0)
uitgaven_full = (
    uitgaven_mnd_ser.reindex(sorted(vaste_cats)).fillna(0.0).rename("uitgave")
)

# Budget-state initialiseren of bijwerken met defaults uit gemiddelde
if "budget_state" not in st.session_state:
    st.session_state.budget_state = pd.DataFrame({
        "categorie": sorted(vaste_cats),
        "budget": [float(gemiddelde_per_cat.get(cat, np.nan)) if not pd.isna(gemiddelde_per_cat.get(cat, np.nan)) else np.nan
                    for cat in sorted(vaste_cats)]
    })
else:
    prev_state = st.session_state.budget_state
    merged = pd.DataFrame({"categorie": sorted(vaste_cats)}).merge(prev_state, on="categorie", how="left")
    # Vul alleen lege budgetten met gemiddelde
    if not gemiddelde_per_cat.empty:
        merged.loc[merged["budget"].isna(), "budget"] = merged.loc[merged["budget"].isna(), "categorie"].map(gemiddelde_per_cat)
    st.session_state.budget_state = merged

with st.expander("✏️ Stel budgetten in (per categorie)", expanded=False):
    budget_df = st.data_editor(
        st.session_state.budget_state,
        num_rows="dynamic",
        hide_index=True,
        key="budget_editor",
        column_config={
            "categorie": st.column_config.TextColumn("Categorie", disabled=True),
            "budget": st.column_config.NumberColumn("Budget (€)", min_value=0.0, step=10.0, help="Auto = gemiddelde vorige maanden; aanpasbaar")
        }
    )
    st.session_state.budget_state = budget_df

# Join budget + uitgaven (index-based)
budget_join = (
    st.session_state.budget_state.set_index("categorie").join(uitgaven_full, how="left").reset_index()
)

budget_join["budget"] = pd.to_numeric(budget_join["budget"], errors="coerce")
budget_join["uitgave"] = pd.to_numeric(budget_join["uitgave"], errors="coerce").fillna(0)
budget_join["verschil"] = budget_join["budget"] - budget_join["uitgave"]

# Tabel weergeven
tabel = budget_join.assign(
    Budget=budget_join["budget"].apply(lambda x: euro(x) if pd.notna(x) else "—"),
    Uitgave=budget_join["uitgave"].apply(euro),
    **{"Δ (budget - uitgave)": budget_join["verschil"].apply(lambda x: euro(x) if pd.notna(x) else "—")},
    Status=np.where(
        budget_join["budget"].notna() & (budget_join["uitgave"] > budget_join["budget"]),
        "🚨 Over budget",
        np.where(budget_join["budget"].notna(), "✅ Binnen budget", "—")
    )
)
kolommen = [k for k in ["categorie", "Budget", "Uitgave", "Δ (budget - uitgave)", "Status"] if k in tabel.columns]
st.dataframe(tabel.loc[:, kolommen].rename(columns={"categorie": "Categorie"}), use_container_width=True)

# Grafiek: Uitgave vs Budget
b_plot = budget_join.dropna(subset=["budget"]).copy()
if not b_plot.empty:
    b_plot = b_plot.sort_values("uitgave", ascending=False)
    fig_b = px.bar(
        b_plot.melt(id_vars=["categorie"], value_vars=["uitgave", "budget"], var_name="type", value_name="€"),
        x="categorie", y="€", color="type", barmode="group",
        title=f"Uitgaven vs. Budget — {geselecteerde_maand}", labels={"categorie": "Categorie"}
    )
    st.plotly_chart(fig_b, use_container_width=True)

# 🔮 Prognose einde van de maand — alleen budgetten (geen dagen/tempo)
st.subheader("🔮 Prognose einde van de maand")

if not df_maand.empty:
    laatste_datum = df_maand["datum"].max()
    jaar, mnd = laatste_datum.year, laatste_datum.month

    # Alle uitgaven (excl. inkomsten) in de geselecteerde jaar-maand
    mask_ym = (
        (df_filtered["datum"].dt.year == jaar)
        & (df_filtered["datum"].dt.month == mnd)
        & (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
    )
    df_ym = df_filtered[mask_ym].copy()

    if not df_ym.empty:
        # Totaal uitgegeven tot en met vandaag
        uitg_tmv = abs(df_ym[df_ym["datum"] <= laatste_datum]["bedrag"].sum())

        # Reeds uitgegeven per categorie
        spent_per_cat = (
            df_ym[df_ym["datum"] <= laatste_datum]
            .groupby("categorie")["bedrag"].sum().abs()
        )

        # Budget per categorie uit editor
        budget_per_cat = (
            budget_join.set_index("categorie")["budget"].astype(float)
            if "budget" in budget_join.columns else pd.Series(dtype=float)
        )

        # Alleen budgetten: resterend = max(0, budget - spent); geen budget => 0
        resterend_per_cat = (budget_per_cat - spent_per_cat).clip(lower=0).fillna(0)

        # Prognose = al uitgegeven + wat er nog 'mag' volgens budget
        proj = float(uitg_tmv + resterend_per_cat.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Uitgaven tot en met vandaag", euro(uitg_tmv))
        c2.metric("Voorspelling maandtotaal", euro(proj))
        c3.metric("Nog te verwachten", euro(proj - uitg_tmv))

        # Vergelijk met totaalbudget
        totaal_budget = pd.to_numeric(budget_join["budget"], errors="coerce").sum(skipna=True)
        if not np.isnan(totaal_budget) and totaal_budget > 0:
            if proj > totaal_budget:
                st.error(f"⚠️ Verwachte uitgaven ({euro(proj)}) liggen boven totaalbudget ({euro(totaal_budget)}).")
            else:
                st.success(f"✅ Verwachte uitgaven ({euro(proj)}) liggen binnen totaalbudget ({euro(totaal_budget)}).")

        st.caption("Prognose gebaseerd uitsluitend op budgetten: resterend = max(0, budget − uitgegeven).")
    else:
        st.info("ℹ️ Geen uitgaven gevonden voor de gekozen jaar-maand.")
else:
    st.info("ℹ️ Geen data in de geselecteerde maand voor prognose.")
