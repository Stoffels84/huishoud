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

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

# Totalen
totaal = df_filtered['bedrag'].sum()
inkomen = df_filtered[df_filtered['bedrag'] > 0]['bedrag'].sum()
uitgaven = df_filtered[df_filtered['bedrag'] < 0]['bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# Categorie grafiek
if 'Categorie' in df_filtered.columns:
    st.subheader("📂 Bedragen per categorie")
    categorie_data = df_filtered.groupby("categorie")["bedrag"].sum().sort_values()
    st.bar_chart(categorie_data)

# Maand grafiek
st.subheader("📅 Saldo per maand")
df_filtered['maand'] = df_filtered['datum'].dt.to_period('M').astype(str)
maand_data = df_filtered.groupby("maand")["bedrag"].sum()
st.line_chart(maand_data)

# 📊 Draaitabel per maand en categorie
st.subheader("📅 Draaitabel: Som van bedragen per maand en categorie")

# Zorg dat er een 'maand' kolom is
df_filtered['maand'] = df_filtered['datum'].dt.strftime('%B')  # maand als naam (januari, februari, ...)

# Sorteer op maandvolgorde
maand_volgorde = [
    'januari', 'februari', 'maart', 'april', 'mei', 'juni',
    'juli', 'augustus', 'september', 'oktober', 'november', 'december'
]
df_filtered['maand'] = df_filtered['maand'].str.lower()
df_filtered['maand'] = pd.Categorical(df_filtered['maand'], categories=maand_volgorde, ordered=True)

# Draaitabel maken
pivot = pd.pivot_table(
    df_filtered,
    index=['vast/variabel', 'categorie'],
    columns='maand',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
)

# Reset index zodat het leesbaar is in de Streamlit-tabel
pivot = pivot.reset_index()

# Tabel tonen
st.dataframe(pivot, use_container_width=True)


# Tabel
st.subheader("📄 Transacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)
