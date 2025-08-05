import streamlit as st
import pandas as pd
import locale


# 📋 Pagina setup
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# 📁 Data inladen
def laad_data():
    st.write("📁 Bestand gevonden, laden maar...")
    df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")
    st.write("✅ Data geladen!")

    # Kolomnamen schoonmaken (kleine letters)
    df.columns = df.columns.str.strip().str.lower()
    
    # Zorg voor correcte datums en verwijder lege regels
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum', 'bedrag'])

    return df

df = laad_data()

# 📅 Sidebar: datumbereik filter
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

# 🔎 Filter toepassen
df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

# ✅ Totalen berekenen
totaal = df_filtered['bedrag'].sum()
inkomen = df_filtered[df_filtered['bedrag'] > 0]['bedrag'].sum()
uitgaven = df_filtered[df_filtered['bedrag'] < 0]['bedrag'].sum()

# 🔢 Toon kerncijfers
col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# 📊 Bar chart: uitgaven per categorie
if 'categorie' in df_filtered.columns:
    st.subheader("📂 Uitgaven per categorie")
    categorie_data = df_filtered[df_filtered['bedrag'] < 0].groupby("categorie")["bedrag"].sum().sort_values()
    st.bar_chart(categorie_data)

# 📉 Lijngrafiek: saldo per maand
st.subheader("📅 Saldo per maand")
df_filtered['maand'] = df_filtered['datum'].dt.to_period('M').astype(str)
maand_saldo = df_filtered.groupby('maand')['bedrag'].sum()
st.line_chart(maand_saldo)

# 📅 Draaitabellen per maand
st.subheader("📋 Draaitabellen per maand")

# 📆 Maandnaam in Nederlands
maanden_nl = {
    1: 'januari', 2: 'februari', 3: 'maart', 4: 'april',
    5: 'mei', 6: 'juni', 7: 'juli', 8: 'augustus',
    9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
}
df_filtered['maand'] = df_filtered['datum'].dt.month.map(maanden_nl)


# Maand volgorde vastleggen
maand_volgorde = [
    'januari', 'februari', 'maart', 'april', 'mei', 'juni',
    'juli', 'augustus', 'september', 'oktober', 'november', 'december'
]
df_filtered['maand'] = pd.Categorical(df_filtered['maand'], categories=maand_volgorde, ordered=True)

# 🔄 Opsplitsen in inkomsten en uitgaven
inkomsten_df = df_filtered[df_filtered['bedrag'] > 0]
uitgaven_df = df_filtered[df_filtered['bedrag'] < 0]

# 📈 Inkomsten draaitabel
st.markdown("### 📈 Inkomsten per maand")
pivot_inkomen = pd.pivot_table(
    inkomsten_df,
    index='categorie',
    columns='maand',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
).reset_index()
st.dataframe(pivot_inkomen, use_container_width=True)

# 📉 Uitgaven draaitabel
st.markdown("### 📉 Uitgaven per maand")
pivot_uitgaven = pd.pivot_table(
    uitgaven_df,
    index=['vast/variabel', 'categorie'],
    columns='maand',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
).reset_index()
st.dataframe(pivot_uitgaven, use_container_width=True)

# 📄 Detailtabel
st.subheader("📄 Detailtransacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)
