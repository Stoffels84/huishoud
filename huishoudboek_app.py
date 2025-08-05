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

    # Kolommen opschonen
    df.columns = df.columns.str.strip().str.lower()
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum', 'bedrag'])

    df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
    df['vast/variabel'] = df['vast/variabel'].astype(str).str.strip().str.title()

    # Maandnaam toevoegen
    df['maand'] = df['datum'].dt.month
    df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

    return df

df = laad_data()

# 📅 Filter op datum
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]

# 🔄 Draaitabel per categorie & maand
st.subheader("📊 Uitgaven & inkomsten per categorie en maand")

# Juiste maandvolgorde
maanden = list(calendar.month_name)[1:]  # Januari -> December
df_filtered['maand_naam'] = pd.Categorical(df_filtered['maand_naam'], categories=maanden, ordered=True)

# Draaitabel
pivot = df_filtered.pivot_table(
    index=['vast/variabel', 'categorie'],
    columns='maand_naam',
    values='bedrag',
    aggfunc='sum',
    fill_value=0
)

# Totale kolom toevoegen
pivot["Totaal"] = pivot.sum(axis=1)

# Tabel weergeven
st.dataframe(pivot, use_container_width=True)

# 📄 Transacties
st.subheader("📄 Alle transacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)
