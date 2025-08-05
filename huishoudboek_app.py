import streamlit as st
import pandas as pd
import calendar

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# 📁 Data inladen
def laad_data():
    st.write("📁 Bestand gevonden, laden maar...")
    df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")

    # Check kolomnamen
    st.write("🧾 Kolomnamen in bestand:", df.columns.tolist())

    df.columns = df.columns.str.strip().str.lower()
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')

    # Check type van 'bedrag'
    st.write("📊 Unieke types in 'bedrag':", df['bedrag'].map(type).unique())

    df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')  # heel belangrijk!

    df = df.dropna(subset=['datum', 'bedrag'])

    df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
    df['vast/variabel'] = df['vast/variabel'].astype(str).str.strip().str.title()

    df['maand'] = df['datum'].dt.month
    df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

    # Laat 5 rijen zien om te checken
    st.write("📄 Voorbeeld data:", df.head())

    return df


df = laad_data()

# 📅 Filter op datum
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]
st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))


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
