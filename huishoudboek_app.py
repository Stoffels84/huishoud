import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
import calendar

# ----------------------------
# 🔧 Pagina-instellingen
# ----------------------------
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ----------------------------
# 📅 Maanden in het Nederlands
# ----------------------------
MAANDEN_NL = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December"
]
maand_type = CategoricalDtype(categories=MAANDEN_NL, ordered=True)

# ----------------------------
# 💶 Helper: euro-format
# ----------------------------
def euro(x):
    try:
        return f"€ {x:,.2f}"
    except Exception:
        return "€ 0,00"

# ----------------------------
# 🧮 Helper voor percentages
# ----------------------------
def pct(value, total, *, signed=False, absolute=False):
    """Format percentage. signed=True => +/-, absolute=True => |value|/total."""
    if total is None or total == 0 or pd.isna(total):
        return "—"
    num = abs(value) if absolute else value
    p = (num / total) * 100
    return f"{p:+.1f}%" if signed else f"{p:.1f}%"

# ----------------------------
# 📥 Data inladen (met optionele upload)
# ----------------------------
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

# ----------------------------
# 📅 Filters (met Reset)
# ----------------------------
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

# Update session state als user inputs wijzigen
st.session_state.start_datum = start_datum
st.session_state.eind_datum = eind_datum

# Filter toepassen
df_filtered = df[(df["datum"] >= pd.to_datetime(start_datum)) &
                 (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_filtered["maand_naam"] = df_filtered["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))
if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# Maandkeuze: chronologisch en alleen aanwezige maanden
present = set(df_filtered["maand_naam"].dropna().astype(str).tolist())
beschikbare_maanden = [m for m in MAANDEN_NL if m in present]
default_maand = beschikbare_maanden[-1] if beschikbare_maanden else MAANDEN_NL[0]

with st.sidebar:
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        beschikbare_maanden,
        index=(beschikbare_maanden.index(default_maand) if beschikbare_maanden else 0),
        key="maand_select"
    )

# ----------------------------
# 📅 Maand-metrics (saldo)
# ----------------------------
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

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📈 Inkomen", euro(inkomen_m), "—")
col_m2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo_m), f"{pct(vast_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo_m), f"{pct(variabel_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m4.metric("💰 Netto saldo maand", euro(totaal_saldo_m), f"{pct(totaal_saldo_m, inkomen_m, signed=True)} van inkomen")

# ----------------------------
# 📊 Financiële metrics (gehele periode)
# ----------------------------
is_loon_all = df_filtered["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_filtered[is_loon_all]
df_vast = df_filtered[df_filtered["vast/variabel"] == "Vast"]
df_variabel = df_filtered[df_filtered["vast/variabel"] == "Variabel"]

inkomen = df_loon["bedrag"].sum()
vast_saldo = df_vast["bedrag"].sum()
variabel_saldo = df_variabel["bedrag"].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", euro(inkomen), "—")
col2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo), f"{pct(vast_saldo, inkomen, absolute=True)} van inkomen")
col3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo), f"{pct(variabel_saldo, inkomen, absolute=True)} van inkomen")
col4.metric("💰 Totaal saldo", euro(totaal_saldo), f"{pct(totaal_saldo, inkomen, signed=True)} van inkomen")

# ----------------------------
# 💡 Financiële gezondheidsscore
# ----------------------------
st.subheader("💡 Financiële Gezondheid")
totale_uitgaven = abs(vast_saldo) + abs(variabel_saldo)
if inkomen > 0:
    gezondheid_score = 100 - ((totale_uitgaven / inkomen) * 100)
    gezondheid_score = max(0, min(100, gezondheid_score))
else:
    gezondheid_score = 0

st.metric("💚 Gezondheidsscore", f"{gezondheid_score:.0f} / 100", help="Gebaseerd op verhouding tussen uitgaven en inkomen")

fig_score = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gezondheid_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Financiële gezondheid"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green" if gezondheid_score >= 60 else "orange" if gezondheid_score >= 30 else "red"},
        'steps': [
            {'range': [0, 30], 'color': "#ffcccc"},
            {'range': [30, 60], 'color': "#ffe0b3"},
            {'range': [60, 100], 'color': "#ccffcc"}
        ],
    }
))
st.plotly_chart(fig_score, use_container_width=True)

st.caption(
    "De score schat hoeveel van je inkomen beschikbaar blijft: "
    "100 = alle uitgaven zijn 0; 0 = uitgaven ≥ inkomen. "
    "Richtwaarden: ≥60 goed, 30–60 aandacht, <30 risicovol."
)

# ----------------------------
# 📋 Draaitabellen
# ----------------------------

def _maak_pivot(data):
    data = data.copy()
    data["categorie"] = data["categorie"].astype(str).str.strip()
    data = data[data["categorie"].notna() & (data["categorie"] != "")]
    if data.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(
        data,
        index="categorie",
        columns="maand_naam",
        values="bedrag",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Totaal",
    )
    pivot = pivot.reindex(columns=[m for m in MAANDEN_NL if m in pivot.columns] + ["Totaal"]).sort_index()
    return pivot

st.subheader("📂 Overzicht per groep")
st.dataframe(_maak_pivot(df_loon).style.format("€ {:,.2f}"), use_container_width=True, height=300)
st.dataframe(_maak_pivot(df_vast).style.format("€ {:,.2f}"), use_container_width=True, height=300)
st.dataframe(_maak_pivot(df_variabel).style.format("€ {:,.2f}"), use_container_width=True, height=300)

# ----------------------------
# 📈 Grafieken
# ----------------------------
st.subheader("📈 Grafieken per maand en categorie")
inkomen_per_maand = df_loon.groupby("maand_naam")["bedrag"].sum().sort_index().fillna(0)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand, use_container_width=True)

kosten_per_maand = (
    df_filtered[df_filtered["vast/variabel"].isin(["Vast", "Variabel"])]
    .groupby(["maand_naam", "vast/variabel"])["bedrag"]
    .sum()
    .unstack()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📉 Vaste en variabele kosten per maand (saldo)")
st.line_chart(kosten_per_maand, use_container_width=True)

# ----------------------------
# 🎯 Budgetdoelen per categorie — alleen VASTE kosten
# ----------------------------
st.subheader(f"🎯 Budgetdoelen per categorie — {geselecteerde_maand}")

# 1) Alle vaste kosten-categorieën (altijd tonen, ook zonder uitgave deze maand)
vaste_cats = (
    df[df["vast/variabel"].astype(str).str.strip().str.title().eq("Vast")]["categorie"]
      .astype(str).str.strip().str.title().dropna().unique()
)

# 2) Werkelijke uitgaven (alleen vaste kosten) in geselecteerde maand
uitgaven_mnd_ser = (
    df_filtered[
        (df_filtered["maand_naam"] == geselecteerde_maand) &
        (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")) &
        (df_filtered["vast/variabel"].astype(str).str.strip().str.title().eq("Vast"))
    ]
    .groupby("categorie")["bedrag"].sum()
    .abs()
)

# 3) Maak volledige categorie-lijst met uitgaven=0 als ontbrekend
uitgaven_mnd = (
    pd.Series(0.0, index=sorted(vaste_cats), name="uitgave")
      .add(uitgaven_mnd_ser, fill_value=0)
      .rename_axis("categorie")
      .reset_index()
)

# 4) Budget-editor (bewaar in session_state)
if "budget_state" not in st.session_state:
    st.session_state.budget_state = pd.DataFrame({"categorie": sorted(vaste_cats), "budget": np.nan})
else:
    # sync: voeg nieuwe categorieën toe, behoud bestaande budgetten
    prev = st.session_state.budget_state
    st.session_state.budget_state = (
        pd.DataFrame({"categorie": sorted(vaste_cats)})
        .merge(prev, on="categorie", how="left")
    )

with st.expander("✏️ Stel budgetten in (per categorie)", expanded=False):
    budget_df = st.data_editor(
        st.session_state.budget_state,
        num_rows="dynamic",
        hide_index=True,
        key="budget_editor",
        column_config={
            "categorie": st.column_config.TextColumn("Categorie", disabled=True),
            "budget": st.column_config.NumberColumn("Budget (€)", min_value=0.0, step=10.0, help="Maandbudget per categorie")
        }
    )
    st.session_state.budget_state = budget_df

# 5) Combineer budgetten met uitgaven
budget_join = (
    budget_df.set_index("categorie")
             .join(uitgaven_mnd.set_index("categorie")["uitgave"], how="left")
             .reset_index()
    )
