import streamlit as st
import pandas as pd
import calendar
from pandas.api.types import CategoricalDtype
import plotly.express as px


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
# 📅 Filter op periode
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
    geselecteerde_maand = st.selectbox("📆 Kies een maand voor uitgavenanalyse", sorted(unieke_maanden, key=lambda x: maand_volgorde.index(x)))

# ----------------------------
# 🧭 Maanden sorteren op volgorde
# ----------------------------

maand_volgorde = list(calendar.month_name)[1:]  # ['January', ..., 'December']
maand_type = CategoricalDtype(categories=maand_volgorde, ordered=True)

df_filtered['maand_naam'] = df_filtered['maand_naam'].astype(maand_type)
df_loon = df_filtered[df_filtered['categorie'].str.lower() == 'inkomsten loon']
df_loon['maand_naam'] = df_loon['maand_naam'].astype(maand_type)

# ----------------------------
# 📊 Metrics met saldi
# ----------------------------

df_vast = df_filtered[df_filtered['vast/variabel'] == 'Vast']
df_variabel = df_filtered[df_filtered['vast/variabel'] == 'Variabel']

inkomen = df_loon['bedrag'].sum()
vast_saldo = df_vast['bedrag'].sum()
variabel_saldo = df_variabel['bedrag'].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

def pct(v, t): return f"{(v/t*100):.1f}%" if t != 0 else "0%"

pct_vast = pct(vast_saldo, inkomen)
pct_variabel = pct(variabel_saldo, inkomen)
pct_totaal = pct(totaal_saldo, inkomen)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}", "100%")
col2.metric("📌 Vaste kosten", f"€ {vast_saldo:,.2f}", f"{pct_vast} van inkomen")
col3.metric("📎 Variabele kosten", f"€ {variabel_saldo:,.2f}", f"{pct_variabel} van inkomen")
col4.metric("💰 Totaal saldo", f"€ {totaal_saldo:,.2f}", f"{pct_totaal} van inkomen")

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

# 📈 Inkomen per maand (chronologisch)
inkomen_per_maand = (
    df_loon.groupby('maand_naam')['bedrag']
    .sum()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand)

# 📉 Vaste & variabele kosten per maand (chronologisch)
kosten_per_maand = (
    df_filtered[df_filtered['vast/variabel'].isin(['Vast', 'Variabel'])]
    .groupby(['maand_naam', 'vast/variabel'])['bedrag']
    .sum()
    .unstack()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📉 Vaste en variabele kosten per maand")
st.line_chart(kosten_per_maand)

st.subheader(f"🍩 Uitgaven per categorie in {geselecteerde_maand} (excl. 'Inkomsten Loon')")

df_donut = df_filtered[
    (df_filtered['maand_naam'] == geselecteerde_maand) &
    (df_filtered['categorie'].str.lower() != 'inkomsten loon')
]

if df_donut.empty:
    st.info("ℹ️ Geen uitgaven gevonden voor deze maand.")
else:
    donut_data = df_donut.groupby('categorie')['bedrag'].sum().reset_index()
    donut_data['bedrag'] = donut_data['bedrag'].abs()  # Zorg dat alle waarden positief zijn

    fig = px.pie(
        donut_data,
        names='categorie',
        values='bedrag',
        hole=0.4,
        title=f"Verdeling uitgaven in {geselecteerde_maand}"
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


