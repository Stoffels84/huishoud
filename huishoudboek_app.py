import streamlit as st
import pandas as pd
import calendar
import plotly.express as px
import matplotlib.pyplot as plt
import calplot
from pandas.api.types import CategoricalDtype

# ----------------------------
# 🔧 Pagina-instellingen
# ----------------------------
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ----------------------------
# 📥 Data inladen
# ----------------------------
def laad_data():
    try:
        st.info("📁 Bestand gevonden, laden maar...")
        df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")

        # Kolomnamen opschonen
        df.columns = df.columns.str.strip().str.lower()

        # Verplichte kolommen controleren
        verplichte_kolommen = ['datum', 'bedrag', 'categorie']
        for kolom in verplichte_kolommen:
            if kolom not in df.columns:
                st.error(f"Kolom '{kolom}' ontbreekt in Excel-bestand.")
                st.stop()

        # Data opschonen
        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
        df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')
        df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
        df['vast/variabel'] = df.get('vast/variabel', 'Onbekend').astype(str).str.strip().str.title()

        # Maandnamen toevoegen
        df['maand'] = df['datum'].dt.month
        df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

        # Onvolledige rijen verwijderen
        df = df.dropna(subset=['datum', 'bedrag', 'categorie'])
        df = df[df['categorie'].str.strip() != ""]

        st.success("✅ Data geladen!")
        with st.expander("📄 Voorbeeld van de data"):
            st.write(df.head())

        return df

    except Exception as e:
        st.error(f"❌ Fout bij het laden van de data: {e}")
        st.stop()

df = laad_data()

# ----------------------------
# 🧭 Maanden sorteren op juiste volgorde
# ----------------------------
maand_volgorde = list(calendar.month_name)[1:]  # ['January', ..., 'December']
maand_type = CategoricalDtype(categories=maand_volgorde, ordered=True)

# ----------------------------
# 📅 Filter op periode en maand
# ----------------------------
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]
st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))

if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

with st.sidebar:
    unieke_maanden = df_filtered['maand_naam'].dropna().unique()
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        sorted(unieke_maanden, key=lambda x: maand_volgorde.index(x))
    )

# ----------------------------
# 📅 Metrics voor geselecteerde maand
# ----------------------------
st.subheader(f"📆 Overzicht voor {geselecteerde_maand}")

df_maand = df_filtered[df_filtered['maand_naam'] == geselecteerde_maand]

# Opnieuw categoriseren
df_loon_m = df_maand[df_maand['categorie'].str.lower() == 'inkomsten loon']
df_vast_m = df_maand[df_maand['vast/variabel'] == 'Vast']
df_variabel_m = df_maand[df_maand['vast/variabel'] == 'Variabel']

inkomen_m = df_loon_m['bedrag'].sum()
vast_saldo_m = df_vast_m['bedrag'].sum()
variabel_saldo_m = df_variabel_m['bedrag'].sum()
totaal_saldo_m = inkomen_m + vast_saldo_m + variabel_saldo_m

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📈 Inkomen", f"€ {inkomen_m:,.2f}", "100%")
col_m2.metric("📌 Vaste kosten", f"€ {vast_saldo_m:,.2f}", f"{pct(vast_saldo_m, inkomen_m)} van inkomen")
col_m3.metric("📎 Variabele kosten", f"€ {variabel_saldo_m:,.2f}", f"{pct(variabel_saldo_m, inkomen_m)} van inkomen")
col_m4.metric("💰 Totaal saldo", f"€ {totaal_saldo_m:,.2f}", f"{pct(totaal_saldo_m, inkomen_m)} van inkomen")


# ----------------------------
# 📊 Financiële metrics
# ----------------------------
df_filtered['maand_naam'] = df_filtered['maand_naam'].astype(maand_type)

df_loon = df_filtered[df_filtered['categorie'].str.lower() == 'inkomsten loon']
df_loon['maand_naam'] = df_loon['maand_naam'].astype(maand_type)

df_vast = df_filtered[df_filtered['vast/variabel'] == 'Vast']
df_variabel = df_filtered[df_filtered['vast/variabel'] == 'Variabel']

inkomen = df_loon['bedrag'].sum()
vast_saldo = df_vast['bedrag'].sum()
variabel_saldo = df_variabel['bedrag'].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

def pct(v, t): return f"{(v/t*100):.1f}%" if t != 0 else "0%"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}", "100%")
col2.metric("📌 Vaste kosten", f"€ {vast_saldo:,.2f}", f"{pct(vast_saldo, inkomen)} van inkomen")
col3.metric("📎 Variabele kosten", f"€ {variabel_saldo:,.2f}", f"{pct(variabel_saldo, inkomen)} van inkomen")
col4.metric("💰 Totaal saldo", f"€ {totaal_saldo:,.2f}", f"{pct(totaal_saldo, inkomen)} van inkomen")

# ----------------------------
# 💡 Financiële gezondheidsscore
# ----------------------------
st.subheader("💡 Financiële Gezondheid")

totale_uitgaven = abs(vast_saldo + variabel_saldo)
if inkomen > 0:
    gezondheid_score = 100 - ((totale_uitgaven / inkomen) * 100)
    gezondheid_score = max(0, min(100, gezondheid_score))
else:
    gezondheid_score = 0

st.metric("💚 Gezondheidsscore", f"{gezondheid_score:.0f} / 100", help="Gebaseerd op verhouding tussen uitgaven en inkomen")

# Alternatief: visuele meter via plotly
import plotly.graph_objects as go

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
    data['categorie'] = data['categorie'].astype(str).str.strip()
    data = data[data['categorie'].notna() & (data['categorie'] != "")]

    if data.empty:
        st.info(f"ℹ️ Geen gegevens beschikbaar voor: {titel}")
        return

    st.markdown(f"### {titel}")
    pivot = pd.pivot_table(
        data,
        index='categorie',
        columns='maand_naam',
        values='bedrag',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Totaal'
    )
    pivot = pivot.reindex(columns=[m for m in maand_volgorde if m in pivot.columns] + ['Totaal'])
    st.dataframe(pivot.style.format("€ {:,.2f}"), use_container_width=True, height=400)

st.subheader("📂 Overzicht per groep")
toon_draaitabel(df_loon, "💼 Inkomsten: Loon")
toon_draaitabel(df_vast, "📌 Vaste kosten")
toon_draaitabel(df_variabel, "📎 Variabele kosten")

# ----------------------------
# 📊 Grafieken
# ----------------------------
st.subheader("📈 Grafieken per maand en categorie")

# 📈 Inkomen per maand
inkomen_per_maand = (
    df_loon.groupby('maand_naam')['bedrag']
    .sum()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand, use_container_width=True)

# 📉 Vaste & variabele kosten per maand
kosten_per_maand = (
    df_filtered[df_filtered['vast/variabel'].isin(['Vast', 'Variabel'])]
    .groupby(['maand_naam', 'vast/variabel'])['bedrag']
    .sum()
    .unstack()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📉 Vaste en variabele kosten per maand")
st.line_chart(kosten_per_maand, use_container_width=True)

# 🍩 Donutgrafiek per maand (zonder inkomsten loon)
st.subheader(f"🍩 Uitgaven per categorie in {geselecteerde_maand} (excl. 'Inkomsten Loon')")

df_donut = df_filtered[
    (df_filtered['maand_naam'] == geselecteerde_maand) &
    (df_filtered['categorie'].str.lower() != 'inkomsten loon')
]

if df_donut.empty:
    st.info("ℹ️ Geen uitgaven gevonden voor deze maand.")
else:
    donut_data = df_donut.groupby('categorie')['bedrag'].sum().reset_index()
    donut_data['bedrag'] = donut_data['bedrag'].abs()  # Alle bedragen positief

    fig = px.pie(
        donut_data,
        names='categorie',
        values='bedrag',
        hole=0.4,
        title=f"Verdeling uitgaven in {geselecteerde_maand}"
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 📊 Vergelijking huidige maand met vorige
# ----------------------------
st.subheader("📊 Vergelijking: Uitgaven per maand")

# ----------------------------
# 🔔 Automatische waarschuwing bij hoge uitgaven
# ----------------------------

# Filter alleen uitgaven (geen inkomen)
df_alleen_uitgaven = df_filtered[df_filtered['categorie'].str.lower() != 'inkomsten loon'].copy()
df_alleen_uitgaven['maand_naam'] = df_alleen_uitgaven['maand_naam'].astype(maand_type)

# Totaal per maand berekenen
totaal_per_maand = (
    df_alleen_uitgaven
    .groupby('maand_naam')['bedrag']
    .sum()
    .reindex(maand_volgorde)
    .dropna()
    .abs()
)

if geselecteerde_maand in totaal_per_maand.index:
    huidige_maand_bedrag = totaal_per_maand[geselecteerde_maand]

    # Gemiddelde van alle andere maanden (excl. huidige)
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


# Alleen uitgaven (geen loon)
df_uitgaven = df_filtered[df_filtered['categorie'].str.lower() != 'inkomsten loon'].copy()
df_uitgaven['maand_naam'] = df_uitgaven['maand_naam'].astype(maand_type)

# Groeperen per maand
uitgaven_per_maand = (
    df_uitgaven
    .groupby('maand_naam')['bedrag']
    .sum()
    .reindex(maand_volgorde)
    .fillna(0)
    .abs()
)

fig_vergelijking = px.bar(
    uitgaven_per_maand.reset_index(),
    x='maand_naam',
    y='bedrag',
    labels={'maand_naam': 'Maand', 'bedrag': 'Uitgaven (€)'},
    title="Totale uitgaven per maand",
    text_auto='.2s'
)

fig_vergelijking.update_layout(xaxis_title="Maand", yaxis_title="€")
st.plotly_chart(fig_vergelijking, use_container_width=True)
# ----------------------------
# 🔍 Automatische analyse huidig vs. vorige maand (visueel met pijlen)
# ----------------------------
aanwezige_maanden = [m for m in maand_volgorde if m in uitgaven_per_maand.index and uitgaven_per_maand[m] > 0]

if geselecteerde_maand in aanwezige_maanden:
    huidig_index = aanwezige_maanden.index(geselecteerde_maand)

    if huidig_index > 0:
        vorige_maand = aanwezige_maanden[huidig_index - 1]

        huidig_bedrag = uitgaven_per_maand[geselecteerde_maand]
        vorig_bedrag = uitgaven_per_maand[vorige_maand]

        if vorig_bedrag > 0:
            verschil_pct = ((huidig_bedrag - vorig_bedrag) / vorig_bedrag) * 100

            if verschil_pct < -5:
                st.success(f"🔻 **{abs(verschil_pct):.1f}% minder uitgaven** dan in **{vorige_maand}**. Goed gedaan! 💚")
            elif verschil_pct > 5:
                st.error(f"🔺 **{verschil_pct:.1f}% meer uitgaven** dan in **{vorige_maand}**. Let op! 🔴")
            else:
                st.info(f"⚖️ Uitgaven zijn vergelijkbaar met **{vorige_maand}** ({verschil_pct:.1f}%).")
        else:
            st.info(f"ℹ️ Geen uitgaven in {vorige_maand} om mee te vergelijken.")
    else:
        st.info("ℹ️ Geen eerdere maand beschikbaar om mee te vergelijken.")

# ----------------------------
# 📅 Kalenderweergave van uitgaven
# ----------------------------
st.subheader("📅 Dagelijkse uitgaven (kalenderweergave)")

df_kalender = df_filtered.copy()
df_kalender = df_kalender[df_kalender['categorie'].str.lower() != 'inkomsten loon']
df_kalender = df_kalender.dropna(subset=['datum', 'bedrag'])

# Som per dag berekenen
dagelijkse_uitgaven = (
    df_kalender
    .groupby(pd.to_datetime(df_kalender['datum'].dt.date))['bedrag']
    .sum()
    .abs()
)

dagelijkse_uitgaven.index = pd.to_datetime(dagelijkse_uitgaven.index)  # Belangrijk!

if dagelijkse_uitgaven.empty:
    st.info("ℹ️ Geen uitgaven om weer te geven in de kalender.")
else:
    fig, ax = calplot.calplot(
        dagelijkse_uitgaven,
        cmap='Reds',
        colorbar=True,
        suptitle='Uitgaven per dag',
        figsize=(10, 3)
    )
    st.pyplot(fig)
