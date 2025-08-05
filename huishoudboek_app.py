import streamlit as st
import pandas as pd

# Pagina instellingen
st.set_page_config(page_title="Huishoudboekje", layout="wide")

st.title("📊 Dashboard Huishoudboekje")

# Laad Excel-data
@st.cache_data
def load_data():
    df = pd.read_excel("huishoud.xlsx", sheet_name="data")
    df['Datum'] = pd.to_datetime(df['Datum'])  # Zorg dat 'Datum' datetime is
    return df

df = load_data()

# Filters
with st.sidebar:
    st.header("📅 Filter op periode")
    start_date = st.date_input("Startdatum", df['Datum'].min())
    end_date = st.date_input("Einddatum", df['Datum'].max())

# Filter op datum
filtered_df = df[(df['Datum'] >= pd.to_datetime(start_date)) & (df['Datum'] <= pd.to_datetime(end_date))]

# ✅ Totalen
totaal = filtered_df['Bedrag'].sum()
inkomen = filtered_df[filtered_df['Bedrag'] > 0]['Bedrag'].sum()
uitgaven = filtered_df[filtered_df['Bedrag'] < 0]['Bedrag'].sum()

st.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col1, col2 = st.columns(2)
col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col2.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# 📊 Grafiek per categorie
if 'Categorie' in filtered_df.columns:
    st.subheader("📂 Uitgaven per categorie")
    categorie_data = filtered_df.groupby('Categorie')['Bedrag'].sum().sort_values()
    st.bar_chart(categorie_data)

# 📅 Grafiek per maand
st.subheader("🕓 Uitgaven per maand")
df_maand = filtered_df.copy()
df_maand['Maand'] = df_maand['Datum'].dt.to_period('M').astype(str)
maand_data = df_maand.groupby('Maand')['Bedrag'].sum()
st.line_chart(maand_data)

# 📋 Tabel
st.subheader(
