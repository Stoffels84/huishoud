import streamlit as st
import pandas as pd
import calendar

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

def laad_data():
    try:
        st.info("📁 Bestand gevonden, laden maar...")
        df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")

        # Kolommen opschonen
        df.columns = df.columns.str.strip().str.lower()
        verplicht = ['datum', 'bedrag', 'categorie']
        for kolom in verplicht:
            if kolom not in df.columns:
                st.error(f"Kolom '{kolom}' ontbreekt in het Excel-bestand.")
                st.stop()

        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
        df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')
        df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
        df['vast/variabel'] = df.get('vast/variabel', 'Onbekend').astype(str).str.strip().str.title()

        df = df.dropna(subset=['datum', 'bedrag'])
        df['maand'] = df['datum'].dt.month
        df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

        st.success("✅ Data geladen!")
        st.write("📄 Voorbeeld data:", df.head())

        return df

    except Exception as e:
        st.error(f"❌ Fout bij het laden van data: {e}")
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

# 📌 Draaitabellen INKOMSTEN en UITGAVEN
maand_volgorde = list(calendar.month_name)[1:] + ['Totaal']
df_filtered['maand_naam'] = pd.Categorical(df_filtered['maand_naam'], categories=maand_volgorde[:-1], ordered=True)

# ➕ INKOMSTEN: enkel 'Inkomsten Loon'
df_inkomen = df_filtered[df_filtered['categorie'] == 'Inkomsten Loon']
pivot_inkomen = pd.pivot_table(
    df_inkomen,
    index=['vast/variabel', 'categorie'],
    columns='maand_naam',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
)
pivot_inkomen = pivot_inkomen.reindex(columns=[m for m in maand_volgorde if m in pivot_inkomen.columns])

st.subheader("📈 Inkomsten")
st.dataframe(pivot_inkomen, use_container_width=True)

# ➖ UITGAVEN: alles behalve 'Inkomsten Loon'
df_uitgaven = df_filtered[df_filtered['categorie'] != 'Inkomsten Loon']
pivot_uitgaven = pd.pivot_table(
    df_uitgaven,
    index=['vast/variabel', 'categorie'],
    columns='maand_naam',
    values='bedrag',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Totaal'
)
pivot_uitgaven = pivot_uitgaven.reindex(columns=[m for m in maand_volgorde if m in pivot_uitgaven.columns])

st.subheader("📉 Uitgaven")
st.dataframe(pivot_uitgaven, use_container_width=True)

# 📄 Transacties
st.subheader("📋 Alle transacties")
st.dataframe(df_filtered.sort_values(by="datum", ascending=False), use_container_width=True)
