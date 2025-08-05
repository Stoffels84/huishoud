import streamlit as st
import pandas as pd
import calendar
import plotly.express as px
from pandas.api.types import CategoricalDtype

# ----------------------------
# 🔧 Pagina-instellingen en opmaak
# ----------------------------
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.markdown("<style>div.block-container { padding-top: 1rem; padding-bottom: 1rem; }</style>", unsafe_allow_html=True)
st.title("📊 Huishoudboekje Dashboard")

# ----------------------------
# 📂 Bestand uploaden
# ----------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("📁 Upload je Excel-bestand (.xlsx)", type="xlsx")

# ----------------------------
# 📥 Data inladen
# ----------------------------
def laad_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Data", engine="openpyxl")
        df.columns = df.columns.str.strip().str.lower()

        verplichte_kolommen = ['datum', 'bedrag', 'categorie']
        for kolom in verplichte_kolommen:
            if kolom not in df.columns:
                st.error(f"❌ Kolom '{kolom}' ontbreekt in het Excel-bestand.")
                st.stop()

        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
        df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')
        df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
        df['vast/variabel'] = df.get('vast/variabel', 'Onbekend').astype(str).str.strip().str.title()

        df['maand'] = df['datum'].dt.month
        df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

        df = df.dropna(subset=['datum', 'bedrag', 'categorie'])
        df = df[df['categorie'].str.strip() != ""]

        st.success("✅ Data succesvol geladen!")
        with st.expander("📄 Bekijk voorbeeld van de data"):
            st.write(df.head())

        return df

    except Exception as e:
        st.error(f"❌ Fout bij het laden van de data: {e}")
        st.stop()

if uploaded_file:
    df = laad_data(uploaded_file)
else:
    st.stop()

# ----------------------------
# 📅 Filters in sidebar
# ----------------------------
maand_volgorde = list(calendar.month_name)[1:]
maand_type = CategoricalDtype(categories=maand_volgorde, ordered=True)

with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

if df_filtered.empty:
    st.warning("⚠️ Geen data in de gekozen periode.")
    st.stop()

with st.sidebar:
    st.header("📆 Selecteer een maand")
    beschikbare_maanden = df_filtered['maand_naam'].dropna().unique()
    geselecteerde_maand = st.selectbox(
        "Maand voor uitgavenanalyse",
        sorted(beschikbare_maanden, key=lambda x: maand_volgorde.index(x))
    )

# ----------------------------
# 📊 Financiële metrics (in 2 rijen)
# ----------------------------
df_filtered['maand_naam'] = df_filtered['maand_naam'].astype(maand_type)

df_loon = df_filtered[df_filtered['categorie'].str.lower() == 'inkomsten loon']
df_vast = df_filtered[df_filtered['vast/variabel'] == 'Vast']
df_variabel = df_filtered[df_filtered['vast/variabel'] == 'Variabel']

inkomen = df_loon['bedrag'].sum()
vast_saldo = df_vast['bedrag'].sum()
variabel_saldo = df_variabel['bedrag'].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

def pct(v, t): return f"{(v/t*100):.1f}%" if t != 0 else "0%"

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}", "100%")
col2.metric("📌 Vaste kosten", f"€ {vast_saldo:,.2f}", pct(vast_saldo, inkomen))
col3.metric("📎 Variabele kosten", f"€ {variabel_saldo:,.2f}", pct(variabel_saldo, inkomen))
col4.metric("💰 Totaal saldo", f"€ {totaal_saldo:,.2f}", pct(totaal_saldo, inkomen))

# ----------------------------
# 📂 Tabs: Overzicht – Grafieken – Donut
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📂 Overzicht", "📈 Grafieken", "🍩 Donut"])

# 📋 Overzicht (draaitabellen)
def toon_draaitabel(data, titel):
    if data.empty:
        st.info(f"ℹ️ Geen gegevens beschikbaar voor: {titel}")
        return

    st.markdown(f"### {titel}")
    draaitabel = pd.pivot_table(
        data,
        index='categorie',
        columns='maand_naam',
        values='bedrag',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Totaal'
    )
    draaitabel = draaitabel.reindex(columns=[m for m in maand_volgorde if m in draaitabel.columns] + ['Totaal'])
    st.dataframe(draaitabel.style.format("€ {:,.2f}"), use_container_width=True, height=400)

with tab1:
    toon_draaitabel(df_loon, "💼 Inkomsten: Loon")
    toon_draaitabel(df_vast, "📌 Vaste kosten")
    toon_draaitabel(df_variabel, "📎 Variabele kosten")

# 📈 Grafieken per maand
with tab2:
    st.markdown("#### 📈 Inkomen per maand")
    inkomen_per_maand = df_loon.groupby('maand_naam')['bedrag'].sum().sort_index()
    st.line_chart(inkomen_per_maand, use_container_width=True)

    st.markdown("#### 📉 Vaste en variabele kosten per maand")
    kosten_per_maand = df_filtered[df_filtered['vast/variabel'].isin(['Vast', 'Variabel'])]
    kosten_grafiek = kosten_per_maand.groupby(['maand_naam', 'vast/variabel'])['bedrag'].sum().unstack().sort_index()
    st.line_chart(kosten_grafiek.fillna(0), use_container_width=True)

# 🍩 Donutgrafiek van uitgaven in geselecteerde maand
with tab3:
    st.subheader(f"🍩 Uitgaven in {geselecteerde_maand} (excl. 'Inkomsten Loon')")
    df_donut = df_filtered[
        (df_filtered['maand_naam'] == geselecteerde_maand) &
        (df_filtered['categorie'].str.lower() != 'inkomsten loon')
    ]

    if df_donut.empty:
        st.info("ℹ️ Geen uitgaven gevonden voor deze maand.")
    else:
        donut_data = df_donut.groupby('categorie')['bedrag'].sum().reset_index()
        donut_data['bedrag'] = donut_data['bedrag'].abs()

        fig = px.pie(
            donut_data,
            names='categorie',
            values='bedrag',
            hole=0.4,
            title=f"Verdeling uitgaven in {geselecteerde_maand}"
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
