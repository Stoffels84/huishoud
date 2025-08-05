import streamlit as st
import pandas as pd

# Pagina-instellingen
st.set_page_config(page_title="Huishoudboekje", layout="wide")

# Titel
st.title("📊 Huishoudboekje Dashboard")

# ---------- Data inladen ----------
@st.cache_data
def laad_data():
    df = pd.read_excel("huishoud.xlsx", sheet_name="data")
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df = df.dropna(subset=['Datum', 'Bedrag'])
    return df

df = laad_data()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['Datum'].min())
    eind_datum = st.date_input("Tot", df['Datum'].max())

# Filter toepassen op datum
df_filtered = df[(df['Datum'] >= pd.to_datetime(start_datum)) & (df['Datum'] <= pd.to_datetime(eind_datum))]

# ---------- Kerncijfers ----------
st.subheader("💼 Overzicht")

totaal = df_filtered['Bedrag'].sum()
inkomen = df_filtered[df_filtered['Bedrag'] > 0]['Bedrag'].sum()
uitgaven = df_filtered[df_filtered['Bedrag'] < 0]['Bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# ---------- Grafiek per categorie ----------
if 'Categorie' in df_filtered.columns:
    st.subheader("📂 Bedragen per categorie")
    bedrag_per_categorie = df_filtered.groupby('Categorie')['Bedrag'].sum().sort_values()
    st.bar_chart(bedrag_per_categorie)

# ---------- Grafiek per maand ----------
st.subheader("📅 Saldo per maand")
df_filtered['Maand'] = df_filtered['Datum'].dt.to_period('M').astype(str)
saldo_per_maand = df_filtered.groupby('Maand')['Bedrag'].sum()
st.line_chart(saldo_per_maand)

# ---------- Gegevens tabel ----------
st.subheader("📄 Detailgegevens")
st.dataframe(df_filtered.sort_values(by="Datum", ascending=False), use_container_width=True)
