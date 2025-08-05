import streamlit as st
import pandas as pd

# Pagina-instellingen
st.set_page_config(page_title="Huishoudboekje", layout="wide")

st.title("📒 Mijn Huishoudboekje")

# Upload je Excel-bestand
uploaded_file = st.file_uploader("Upload je Excel-bestand", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📋 Tabeloverzicht")
    st.dataframe(df, use_container_width=True)

    # Voorbeeld: totaal per categorie
    if 'Categorie' in df.columns and 'Bedrag' in df.columns:
        st.subheader("📊 Uitgaven per categorie")
        categorie_totaal = df.groupby('Categorie')['Bedrag'].sum()
        st.bar_chart(categorie_totaal)
    else:
        st.warning("Zorg dat je Excel de kolommen 'Categorie' en 'Bedrag' bevat.")
else:
    st.info("Upload een Excel-bestand om te beginnen.")
