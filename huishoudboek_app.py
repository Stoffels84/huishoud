import streamlit as st
import pandas as pd
import calendar

st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ----------------------------
# 📥 Data inladen
# ----------------------------

def laad_data():
    try:
        st.info("📁 Bestand gevonden, laden maar...")
        df = pd.read_excel("huishoud.xlsx", sheet_name="Data", engine="openpyxl")

        # Kolomnamen opschonen
        df.columns = df.columns.str.strip().str.lower()

        # Verplichte kolommen controleren
        verplichte_kolommen = ['datum', 'bedrag', 'categorie']
        for kolom in verplichte_kolommen:
            if kolom not in df.columns:
                st.error(f"Kolom '{kolom}' ontbreekt in Excel-bestand.")
                st.stop()

        # Types converteren
        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
        df['bedrag'] = pd.to_numeric(df['bedrag'], errors='coerce')
        df['categorie'] = df['categorie'].astype(str).str.strip().str.title()
        df['vast/variabel'] = df.get('vast/variabel', 'Onbekend').astype(str).str.strip().str.title()

        # Extra kolommen
        df['maand'] = df['datum'].dt.month
        df['maand_naam'] = df['datum'].dt.month.apply(lambda x: calendar.month_name[x])

        # Verwijder lege rijen
        df = df.dropna(subset=['datum', 'bedrag', 'categorie'])
        df = df[df['categorie'].str.strip() != ""]

        st.success("✅ Data geladen!")
        with st.expander("📄 Voorbeeld van de data"):
            st.write(df.head())

        return df

    except Exception as e:
        st.error(f"❌ Fout bij het laden van de data: {e}")
        st.stop()

df = laad_data()

# ----------------------------
# 📅 Filter op periode
# ----------------------------

with st.sidebar:
    st.header("📅 Filter op periode")
    start_datum = st.date_input("Van", df['datum'].min())
    eind_datum = st.date_input("Tot", df['datum'].max())

df_filtered = df[(df['datum'] >= pd.to_datetime(start_datum)) & (df['datum'] <= pd.to_datetime(eind_datum))]
st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))

if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# ----------------------------
# 📊 Metrics met correcte saldi
# ----------------------------

# Filters
df_loon = df_filtered[df_filtered['categorie'].str.lower() == 'inkomsten loon']
df_vast = df_filtered[df_filtered['vast/variabel'] == 'Vast']
df_variabel = df_filtered[df_filtered['vast/variabel'] == 'Variabel']

# Netto saldi (inkomsten + uitgaven)
inkomen = df_loon['bedrag'].sum()
vast_saldo = df_vast['bedrag'].sum()
variabel_saldo = df_variabel['bedrag'].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

# Percentages t.o.v. inkomen
def pct(v, t): return f"{(v/t*100):.1f}%" if t != 0 else "0%"

pct_vast = pct(vast_saldo, inkomen)
pct_variabel = pct(variabel_saldo, inkomen)
pct_totaal = pct(totaal_saldo, inkomen)

# 📈 Vier kerngetallen tonen
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", f"€ {inkomen:,.2f}", "100%")
col2.metric("📌 Vaste kosten", f"€ {vast_saldo:,.2f}", f"{pct_vast} van inkomen")
col3.metric("📎 Variabele kosten", f"€ {variabel_saldo:,.2f}", f"{pct_variabel} van inkomen")
col4.metric("💰 Totaal saldo", f"€ {totaal_saldo:,.2f}", f"{pct_totaal} van inkomen")

# ----------------------------
# 📋 Draaitabellen
# ----------------------------

def toon_draaitabel(data, titel):
    # Opschonen: geen lege of foute categorieën
    data = data.copy()
    data['categorie'] = data['categorie'].astype(str).str.strip()
    data = data[data['categorie'].notna() & (data['categorie'] != "")]

    if data.empty:
        st.info(f"ℹ️ Geen gegevens beschikbaar voor: {titel}")
        return

    st.markdown(f"### {titel}")

    pivot = pd.pivot_table(
        data,
        index='categorie',
        columns='maand_naam',
        values='bedrag',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Totaal'
    )

    # Maandvolgorde sorteren
    maand_volgorde = list(calendar.month_name)[1:] + ['Totaal']
    pivot = pivot.reindex(columns=[m for m in maand_volgorde if m in pivot.columns])

    # Opmaken als euro's
    pivot = pivot.applymap(lambda x: f"€ {x:,.2f}")
    st.dataframe(pivot, use_container_width=True, height=400)

# ----------------------------
# 📂 Draaitabellen tonen
# ----------------------------

st.subheader("📂 Overzicht per groep")
toon_draaitabel(df_loon, "💼 Inkomsten: Loon")
toon_draaitabel(df_vast, "📌 Vaste kosten")
toon_draaitabel(df_variabel, "📎 Variabele kosten")
