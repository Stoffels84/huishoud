import streamlit as st
import pandas as pd
import calendar

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

def laad_data():
    try:
        st.info("📁 Bestand gevonden, laden maar...")
        df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")

        # Schoonmaak kolomnamen
        df.columns = df.columns.str.strip().str.lower()

        # Check verplichte kolommen
        verplichte_kolommen = ['datum', 'bedrag', 'categorie']
        for kolom in verplichte_kolommen:
            if kolom not in df.columns:
                st.error(f"Kolom '{kolom}' ontbreekt in Excel-bestand.")
                st.stop()

        # Converteer types
        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
        df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')
        df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
        df['vast/variabel'] = df.get('vast/variabel', 'Onbekend').astype(str).str.strip().str.title()

        # Extra kolommen
        df['maand'] = df['datum'].dt.month
        df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

        # Filter lege
        df = df.dropna(subset=['datum', 'bedrag'])

        st.success("✅ Data geladen!")
        st.write("📄 Voorbeeld data:", df.head())

        return df

    except Exception as e:
        st.error(f"❌ Fout bij het laden van de data: {e}")
        st.stop()


# 📥 Data inladen
df = laad_data()

# 📅 Filters
with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]
st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))

if len(df_filtered) == 0:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# 📊 Totalen
totaal = df_filtered['bedrag'].sum()
inkomen = df_filtered[df_filtered['bedrag'] > 0]['bedrag'].sum()
uitgaven = df_filtered[df_filtered['bedrag'] < 0]['bedrag'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Totaal saldo", f"€ {totaal:,.2f}")
col2.metric("📈 Inkomen", f"€ {inkomen:,.2f}")
col3.metric("📉 Uitgaven", f"€ {uitgaven:,.2f}")

# 📌 Draaitabel per categorie per maand
st.subheader("📂 Inkomsten & uitgaven per categorie per maand")

pivot = pd.pivot_table(
    df_filtered,
    index=['vast/variabel', 'categorie'],
    columns='maand_naam',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
)

# Sorteer maanden
maand_volgorde = list(calendar.month_name)[1:] + ['Totaal']
pivot = pivot.reindex(columns=[m for m in maand_volgorde if m in pivot.columns])

# 📋 Toon draaitabel
st.dataframe(pivot, use_container_width=True)
