import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

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
# 🧮 Helpers
# ----------------------------
def pct(value, total, *, signed=False, absolute=False):
    """
    Format percentage netjes.
    - signed=True -> +/-
    - absolute=True -> |value| / total
    """
    if total is None or total == 0 or pd.isna(total):
        return "—"
    num = abs(value) if absolute else value
    p = (num / total) * 100
    return f"{p:+.1f}%" if signed else f"{p:.1f}%"

# ----------------------------
# 📥 Data inladen
# ----------------------------
@st.cache_data(show_spinner=False)
def laad_data(pad="huishoud.xlsx"):
    df = pd.read_excel(pad, sheet_name="Data", engine="openpyxl")

    # Kolomnamen opschonen
    df.columns = df.columns.str.strip().str.lower()

    # Verplichte kolommen
    verplicht = ["datum", "bedrag", "categorie"]
    ontbreekt = [k for k in verplicht if k not in df.columns]
    if ontbreekt:
        raise ValueError(f"Ontbrekende kolommen: {', '.join(ontbreekt)}")

    # Types & schoonmaak
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df["bedrag"] = pd.to_numeric(df["bedrag"], errors="coerce")
    df["categorie"] = df["categorie"].astype(str).str.strip().str.title()

    if "vast/variabel" not in df.columns:
        df["vast/variabel"] = "Onbekend"
    df["vast/variabel"] = df["vast/variabel"].astype(str).str.strip().str.title()

    # Maandinfo (NL)
    df["maand"] = df["datum"].dt.month
    df["maand_naam"] = df["maand"].apply(lambda m: MAANDEN_NL[m-1] if pd.notnull(m) else "")
    df["maand_naam"] = df["maand_naam"].astype(maand_type)

    # Onvolledige rijen weg
    df = df.dropna(subset=["datum", "bedrag", "categorie"])
    df = df[df["categorie"].str.strip() != ""]

    return df

try:
    st.info("📁 Data laden…")
    df = laad_data()
    st.success("✅ Data geladen!")
    with st.expander("📄 Voorbeeld van de data"):
        st.write(df.head())
except Exception as e:
    st.error(f"❌ Fout bij het laden: {e}")
    st.stop()

# ----------------------------
# 📅 Filters
# ----------------------------
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df["datum"].min())
    eind_datum = st.date_input("Tot", df["datum"].max())

df_filtered = df[(df["datum"] >= pd.to_datetime(start_datum)) &
                 (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_filtered["maand_naam"] = df_filtered["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))
if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

with st.sidebar:
    beschikbare_maanden = [m for m in MAANDEN_NL if m in df_filtered["maand_naam"].dropna().unique()]
    default_maand = beschikbare_maanden[-1] if beschikbare_maanden else MAANDEN_NL[0]
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        beschikbare_maanden,
        index=(beschikbare_maanden.index(default_maand) if beschikbare_maanden else 0)
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
vast_saldo_m = df_vast_m["bedrag"].sum()            # saldo (meestal negatief)
variabel_saldo_m = df_variabel_m["bedrag"].sum()    # saldo (meestal negatief)
totaal_saldo_m = inkomen_m + vast_saldo_m + variabel_saldo_m

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📈 Inkomen", f"€ {inkomen_m:,.2f}", "100%")
col_m2.metric("📌 Vaste kosten (saldo)", f"€ {vast_saldo_m:,.2f}", f"{pct(vast_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m3.metric("📎 Variabele kosten (saldo)", f"€ {variabel_saldo_m:,.2f}", f"{pct(variabel_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m4.metric("💰 Netto saldo maand", f"€ {totaal_saldo_m:,.2f}", f"{pct(totaal_saldo_m, inkomen_m, signed=True)} van inkomen", delta_color="normal")

# ----------------------------
# 📊 Financiële metrics (gehele periode)
# ----------------------------
is_loon_all = df_filtered["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_filtered[is_loon_all].copy()
df_vast = df_filtered[df_filtered["vast/variabel"] == "Vast"].copy()
df_variabel = df_filtered[df_filtered["vast/variabel"] == "Variabel"].copy()

inkomen = df_loon["bedrag"].sum()
vast_saldo = df_vast["bedrag"].sum()
variabel_saldo = df_variabel["bedrag"].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}", "100%")
col2.metric("📌 Vaste kosten", f"€ {vast_saldo:,.2f}", f"{pct(vast_saldo, inkomen, absolute=True)} van inkomen")
col3.metric("📎 Variabele kosten", f"€ {variabel_saldo:,.2f}", f"{pct(variabel_saldo, inkomen, absolute=True)} van inkomen")
col4.metric("💰 Totaal saldo", f"€ {totaal_saldo:,.2f}", f"{pct(totaal_saldo, inkomen, signed=True)} van inkomen", delta_color="normal")

# ----------------------------
# 💡 Financiële gezondheidsscore
# ----------------------------
st.subheader("💡 Financiële Gezondheid")
totale_uitgaven = abs(vast_saldo) + abs(variabel_saldo)  # alleen uitgaande euro's
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

# ----------------------------
# 📋 Draaitabellen
# ----------------------------
def toon_draaitabel(data, titel):
    data = data.copy()
    data["categorie"] = data["categorie"].astype(str).str.strip()
    data = data[data["categorie"].notna() & (data["categorie"] != "")]
    if data.empty:
        st.info(f"ℹ️ Geen gegevens beschikbaar voor: {titel}")
        return
    st.markdown(f"### {titel}")
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
    pivot = pivot.reindex(columns=[m for m in MAANDEN_NL if m in pivot.columns] + ["Totaal"])
    st.dataframe(pivot.style.format("€ {:,.2f}"), use_container_width=True, height=400)

st.subheader("📂 Overzicht per groep")
toon_draaitabel(df_loon, "💼 Inkomsten: Loon")
toon_draaitabel(df_vast, "📌 Vaste kosten")
toon_draaitabel(df_variabel, "📎 Variabele kosten")

# ----------------------------
# 📈 Grafieken
# ----------------------------
st.subheader("📈 Grafieken per maand en categorie")

# 📈 Inkomen per maand (som)
inkomen_per_maand = df_loon.groupby("maand_naam")["bedrag"].sum().sort_index().fillna(0)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand, use_container_width=True)

# 📉 Vaste & variabele kosten per maand (SALDO, dus geen abs)
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

# 🍩 Donut: uitgaven per categorie in gekozen maand (excl. loon) – absolute uitgaven
st.subheader(f"🍩 Uitgaven per categorie in {geselecteerde_maand} (excl. 'Inkomsten Loon')")
df_donut = df_filtered[
    (df_filtered["maand_naam"] == geselecteerde_maand) &
    (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
]
if df_donut.empty:
    st.info("ℹ️ Geen uitgaven gevonden voor deze maand.")
else:
    donut_data = df_donut.groupby("categorie")["bedrag"].sum().abs().reset_index()
    fig = px.pie(donut_data, names="categorie", values="bedrag", hole=0.4,
                 title=f"Verdeling uitgaven in {geselecteerde_maand}")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 📊 Vergelijking & waarschuwing bij hoge uitgaven
# ----------------------------
st.subheader("📊 Vergelijking: Uitgaven per maand")

# Alleen uitgaven (geen loon) voor vergelijking => absolute uitgaven
df_alleen_uitgaven = df_filtered[~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
df_alleen_uitgaven["maand_naam"] = df_alleen_uitgaven["maand_naam"].astype(maand_type)

totaal_per_maand = (
    df_alleen_uitgaven
    .groupby("maand_naam")["bedrag"]
    .sum()
    .reindex(MAANDEN_NL)
    .dropna()
    .abs()
)

if geselecteerde_maand in totaal_per_maand.index:
    huidige_maand_bedrag = totaal_per_maand[geselecteerde_maand]
    overige_maanden = totaal_per_maand.drop(geselecteerde_maand)
    gemiddeld_bedrag = overige_maanden.mean() if not overige_maanden.empty else None

    if gemiddeld_bedrag:
        verschil_pct = ((huidige_maand_bedrag - gemiddeld_bedrag) / gemiddeld_bedrag) * 100
        if verschil_pct > 20:
            st.error(f"🔺 Je hebt deze maand **{verschil_pct:.1f}% meer** uitgegeven dan gemiddeld. Even opletten! 🔴")
        elif verschil_pct < -20:
            st.success(f"🔻 Goed bezig! Je gaf deze maand **{abs(verschil_pct):.1f}% minder** uit dan gemiddeld. 💚")
        else:
            st.info(f"⚖️ Je uitgaven liggen deze maand rond het gemiddelde ({verschil_pct:.1f}%).")
    else:
        st.info("ℹ️ Niet genoeg gegevens om het gemiddelde te berekenen.")

# Barplot: totale uitgaven per maand (absolute uitgaven)
uitgaven_per_maand = (
    df_alleen_uitgaven
    .groupby("maand_naam")["bedrag"]
    .sum()
    .reindex(MAANDEN_NL)
    .fillna(0)
    .abs()
)

fig_vergelijking = px.bar(
    uitgaven_per_maand.reset_index(),
    x="maand_naam",
    y="bedrag",
    labels={"maand_naam": "Maand", "bedrag": "Uitgaven (€)"},
    title="Totale uitgaven per maand",
    text_auto=".2s"
)
fig_vergelijking.update_layout(xaxis_title="Maand", yaxis_title="€")
st.plotly_chart(fig_vergelijking, use_container_width=True)

# ----------------------------
# 📅 Kalenderweergave van uitgaven
# ----------------------------
st.subheader("📅 Dagelijkse uitgaven (kalenderweergave)")
df_kalender = df_filtered[~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
dagelijkse_uitgaven = df_kalender.groupby(pd.to_datetime(df_kalender["datum"].dt.date))["bedrag"].sum().abs()
dagelijkse_uitgaven.index = pd.to_datetime(dagelijkse_uitgaven.index)

if dagelijkse_uitgaven.empty:
    st.info("ℹ️ Geen uitgaven om weer te geven in de kalender.")
else:
    try:
        import calplot
        fig, ax = calplot.calplot(dagelijkse_uitgaven, cmap="Reds", colorbar=True,
                                  suptitle="Uitgaven per dag", figsize=(10, 3))
        st.pyplot(fig)
    except Exception:
        st.info("📅 Calplot niet beschikbaar; val terug op Plotly heatmap.")
        heat = dagelijkse_uitgaven.rename("bedrag").reset_index(names="datum")
        heat["jaar"] = heat["datum"].dt.year
        heat["dag"] = heat["datum"].dt.dayofyear
        fig = px.density_heatmap(heat, x="dag", y="jaar", z="bedrag",
                                 title="Uitgaven per dag (heatmap)", nbinsx=53)
        st.plotly_chart(fig, use_container_width=True)
