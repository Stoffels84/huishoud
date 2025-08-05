import streamlit as st
import pandas as pd
import calendar

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# 📁 Data inladen
def laad_data():
    st.write("📁 Bestand gevonden, laden maar...")
    df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")
    st.success("✅ Data geladen!")

    # Kolomnamen normaliseren
    df.columns = df.columns.str.strip().str.lower()
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
    df['vast/variabel'] = df['vast/variabel'].astype(str).str.strip().str.title()

    # Drop lege rijen
    df = df.dropna(subset=['datum', 'bedrag'])

    # Voeg maand toe
    df['maand'] = df['datum'].dt.month
    df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

    return df

df = laad_data()

# 📅 Filter
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

# 📊 Totalen
totaal = df_filtered['bedrag'].sum()
inkomen = df_filtered[df_filtered['bedrag'] > 0]['bedrag'].sum()
uitgaven = df_filtered[df_filtered['bedrag'] < 0]['bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# 📂 Bedragen per categorie (positief én negatief)
st.subheader("📂 Som per categorie")
categorie_data = df_filtered.groupby("categorie")["bedrag"].sum().sort_values()
st.bar_chart(categorie_data)

# 📅 Saldo per maand
st.subheader("📅 Saldo per maand")
df_filtered['maand_str'] = df_filtered['datum'].dt.to_period('M').astype(str)
maand_data = df_filtered.groupby("maand_str")["bedrag"].sum()
st.line_chart(maand_data)

# 🧾 Transacties
st.subheader("📄 Transacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)

# 🧮 Uitgaven per maand per categorie (zowel positief als negatief)
st.subheader("📉 Uitgaven per maand")

pivot = (
    df_filtered
    .groupby(["vast/variabel", "categorie", "maand_naam"])["bedrag"]
    .sum()
    .reset_index()
)

# Zet maanden in juiste volgorde
maanden_volgorde = list(calendar.month_name)[1:]  # ['January', ..., 'December']
pivot["maand_naam"] = pd.Categorical(pivot["maand_naam"], categories=maanden_volgorde, ordered=True)

# Draaitabel met maanden als kolommen
uitgaven_pivot = pivot.pivot_table(
    index=["vast/variabel", "categorie"],
    columns="maand_naam",
    values="bedrag",
    aggfunc="sum",
    fill_value=0
)

# Voeg totaalkolom toe
uitgaven_pivot["Totaal"] = uitgaven_pivot.sum(axis=1)

# Toon als tabel
st.dataframe(uitgaven_pivot, use_container_width=True)
