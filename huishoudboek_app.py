import streamlit as st
import pandas as pd

# Pagina-instellingen
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# Nederlandse maandnamen handmatig
maanden_nl = {
    1: "januari", 2: "februari", 3: "maart", 4: "april",
    5: "mei", 6: "juni", 7: "juli", 8: "augustus",
    9: "september", 10: "oktober", 11: "november", 12: "december"
}

# Laad data uit Excel
def laad_data():
    st.write("📁 Bestand gevonden, laden maar...")
    df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")
    st.success("✅ Data geladen!")

    # Opschonen kolommen
    df.columns = df.columns.str.strip().str.lower()
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
    df['vast/variabel'] = df['vast/variabel'].astype(str).str.strip().str.title()
    df = df.dropna(subset=['datum', 'bedrag'])

    # Voeg maand toe
    df['maand'] = df['datum'].dt.month.map(maanden_nl)

    return df

df = laad_data()

# Sidebar filters
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

# Filter op datum
df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

# Basis totalen
totaal = df_filtered['bedrag'].sum()
inkomen = df_filtered[df_filtered['bedrag'] > 0]['bedrag'].sum()
uitgaven = df_filtered[df_filtered['bedrag'] < 0]['bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# 📂 Categorie-grafiek
if 'categorie' in df_filtered.columns:
    st.subheader("📂 Bedragen per categorie")
    categorie_data = df_filtered.groupby("categorie")["bedrag"].sum().sort_values()
    st.bar_chart(categorie_data)

# 📅 Maand-grafiek
st.subheader("📅 Saldo per maand")
maand_data = df_filtered.groupby("maand")["bedrag"].sum().reindex(maanden_nl.values())
st.line_chart(maand_data)

# 📄 Uitgaven per maand (zoals draaitabel)
st.subheader("📉 Uitgaven per maand")
uitgaven_df = df_filtered[df_filtered['bedrag'] < 0].copy()
uitgaven_pivot = pd.pivot_table(
    uitgaven_df,
    index=["vast/variabel", "categorie"],
    columns="maand",
    values="bedrag",
    aggfunc="sum",
    fill_value=0
)
uitgaven_pivot["Totaal"] = uitgaven_pivot.sum(axis=1)
uitgaven_pivot = uitgaven_pivot.reset_index()
uitgaven_pivot = uitgaven_pivot[["vast/variabel", "categorie"] + list(maanden_nl.values()) + ["Totaal"]]
st.dataframe(uitgaven_pivot, use_container_width=True)

# 📄 Inkomsten per maand
st.subheader("📈 Inkomsten per maand")
inkomen_df = df_filtered[df_filtered['bedrag'] > 0].copy()
inkomen_pivot = pd.pivot_table(
    inkomen_df,
    index=["categorie"],
    columns="maand",
    values="bedrag",
    aggfunc="sum",
    fill_value=0
)
inkomen_pivot["Totaal"] = inkomen_pivot.sum(axis=1)
inkomen_pivot = inkomen_pivot.reset_index()
inkomen_pivot = inkomen_pivot[["categorie"] + list(maanden_nl.values()) + ["Totaal"]]
st.dataframe(inkomen_pivot, use_container_width=True)

# 📄 Transactielijst
st.subheader("📋 Transacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)
