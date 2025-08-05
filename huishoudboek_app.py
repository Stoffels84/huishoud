import streamlit as st
import pandas as pd

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

def laad_data():
    st.write("📁 Bestand gevonden, laden maar...")
    df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")
    st.write("✅ Data geladen!")
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum', 'bedrag'])
    return df

df = laad_data()

# Filters
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['Datum'] >= pd.to_datetime(start_datum)) & (df['Datum'] <= pd.to_datetime(eind_datum))]

# Totalen
totaal = df_filtered['Bedrag'].sum()
inkomen = df_filtered[df_filtered['Bedrag'] > 0]['Bedrag'].sum()
uitgaven = df_filtered[df_filtered['Bedrag'] < 0]['Bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# Categorie grafiek
if 'Categorie' in df_filtered.columns:
    st.subheader("📂 Bedragen per categorie")
    categorie_data = df_filtered.groupby("Categorie")["Bedrag"].sum().sort_values()
    st.bar_chart(categorie_data)

# Maand grafiek
st.subheader("📅 Saldo per maand")
df_filtered['Maand'] = df_filtered['Datum'].dt.to_period('M').astype(str)
maand_data = df_filtered.groupby("Maand")["Bedrag"].sum()
st.line_chart(maand_data)

# Tabel
st.subheader("📄 Transacties")
st.dataframe(df_filtered.sort_values(by="Datum", ascending=False), use_container_width=True)
