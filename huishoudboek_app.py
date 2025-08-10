import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
import calendar

# ============================================================
# 🔧 Pagina-instellingen
# ============================================================
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ============================================================
# 📅 Maanden in het Nederlands
# ============================================================
MAANDEN_NL = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December"
]
maand_type = CategoricalDtype(categories=MAANDEN_NL, ordered=True)

# ============================================================
# 💶 Helpers
# ============================================================
def euro(x):
    """Eenvoudige euro-notatie zonder extra pakketten."""
    try:
        return f"€ {x:,.2f}"
    except Exception:
        return "€ 0,00"


def pct(value, total, *, signed=False, absolute=False):
    """Percentage formatter. signed=True => +/-; absolute=True => |value|/total."""
    if total is None or total == 0 or pd.isna(total):
        return "—"
    num = abs(value) if absolute else value
    p = (num / total) * 100
    return f"{p:+.1f}%" if signed else f"{p:.1f}%"

# ============================================================
# 📥 Data inladen (met optionele upload)
# ============================================================
with st.sidebar:
    upload = st.file_uploader("📥 Laad Excel (optioneel)", type=["xlsx", "xlsm"], key="upload_main")


@st.cache_data(show_spinner=False)
def laad_data(pad=None, file=None):
    src = file if file is not None else (pad or "huishoud.xlsx")
    df = pd.read_excel(src, sheet_name="Data", engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower()

    verplicht = ["datum", "bedrag", "categorie"]
    ontbreekt = [k for k in verplicht if k not in df.columns]
    if ontbreekt:
        raise ValueError(f"Ontbrekende kolommen: {', '.join(ontbreekt)}")

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df["bedrag"] = pd.to_numeric(df["bedrag"], errors="coerce")
    df["categorie"] = df["categorie"].astype(str).str.strip().str.title()

    if "vast/variabel" not in df.columns:
        df["vast/variabel"] = "Onbekend"
    df["vast/variabel"] = df["vast/variabel"].astype(str).str.strip().str.title()

    df["maand"] = df["datum"].dt.month
    df["maand_naam"] = df["maand"].apply(lambda m: MAANDEN_NL[int(m)-1] if pd.notnull(m) else "")
    df["maand_naam"] = df["maand_naam"].astype(maand_type)

    df = df.dropna(subset=["datum", "bedrag", "categorie"]).copy()
    df = df[df["categorie"].str.strip() != ""]

    return df

try:
    st.info("📁 Data laden…")
    df = laad_data(pad="huishoud.xlsx", file=upload)
    st.success("✅ Data geladen!")
    with st.expander("📄 Voorbeeld van de data"):
        st.write(df.head())
except Exception as e:
    st.error(f"❌ Fout bij het laden: {e}")
    st.stop()

# ============================================================
# 📅 Filters (met Reset)
# ============================================================
with st.sidebar:
    st.header("📅 Filter op periode")
    if "default_start" not in st.session_state:
        st.session_state.default_start = df["datum"].min().date()
        st.session_state.default_end = df["datum"].max().date()
        st.session_state.start_datum = st.session_state.default_start
        st.session_state.eind_datum = st.session_state.default_end

    c1, c2 = st.columns([3, 1])
    with c1:
        start_datum = st.date_input("Van", st.session_state.get("start_datum", st.session_state.default_start), key="date_from")
        eind_datum = st.date_input("Tot", st.session_state.get("eind_datum", st.session_state.default_end), key="date_to")
    with c2:
        if st.button("🔄 Reset", key="reset_btn"):
            st.session_state.start_datum = st.session_state.default_start
            st.session_state.eind_datum = st.session_state.default_end
            st.rerun()

# Sla keuzes op en filter
df_gefilterd = df[(df["datum"] >= pd.to_datetime(start_datum)) & (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_gefilterd["maand_naam"] = df_gefilterd["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_gefilterd))
if df_gefilterd.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# Maandselectie (alleen aanwezige maanden, chronologisch)
aanwezig = set(df_gefilterd["maand_naam"].dropna().astype(str).tolist())
beschikbare_maanden = [m for m in MAANDEN_NL if m in aanwezig]
default_maand = beschikbare_maanden[-1] if beschikbare_maanden else MAANDEN_NL[0]
with st.sidebar:
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        beschikbare_maanden,
        index=(beschikbare_maanden.index(default_maand) if beschikbare_maanden else 0),
        key="maand_select",
    )

# ============================================================
# 📅 Maand-metrics (saldo)
# ============================================================
st.subheader(f"📆 Overzicht voor {geselecteerde_maand}")
df_maand = df_gefilterd[df_gefilterd["maand_naam"] == geselecteerde_maand].copy()

is_loon = df_maand["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon_m = df_maand[is_loon]
df_vast_m = df_maand[df_maand["vast/variabel"] == "Vast"]
df_variabel_m = df_maand[df_maand["vast/variabel"] == "Variabel"]

inkomen_m = df_loon_m["bedrag"].sum()
vast_saldo_m = df_vast_m["bedrag"].sum()
variabel_saldo_m = df_variabel_m["bedrag"].sum()
totaal_saldo_m = inkomen_m + vast_saldo_m + variabel_saldo_m

c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Inkomen", euro(inkomen_m), "—")
c2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo_m), f"{pct(vast_saldo_m, inkomen_m, absolute=True)} van inkomen")
c3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo_m), f"{pct(variabel_saldo_m, inkomen_m, absolute=True)} van inkomen")
c4.metric("💰 Netto saldo maand", euro(totaal_saldo_m), f"{pct(totaal_saldo_m, inkomen_m, signed=True)} van inkomen")

# ============================================================
# 📊 Financiële metrics (gehele periode)
# ============================================================
is_loon_all = df_gefilterd["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_gefilterd[is_loon_all]
df_vast = df_gefilterd[df_gefilterd["vast/variabel"] == "Vast"]
df_variabel = df_gefilterd[df_gefilterd["vast/variabel"] == "Variabel"]

inkomen = df_loon["bedrag"].sum()
vast_saldo = df_vast["bedrag"].sum()
variabel_saldo = df_variabel["bedrag"].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Inkomen", euro(inkomen), "—")
c2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo), f"{pct(vast_saldo, inkomen, absolute=True)} van inkomen")
c3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo), f"{pct(variabel_saldo, inkomen, absolute=True)} van inkomen")
c4.metric("💰 Totaal saldo", euro(totaal_saldo), f"{pct(totaal_saldo, inkomen, signed=True)} van inkomen")

# ============================================================
# 💡 Financiële gezondheidsscore
# ============================================================
st.subheader("💡 Financiële Gezondheid")
totale_uitgaven = abs(vast_saldo) + abs(variabel_saldo)
gezondheid_score = 0 if inkomen <= 0 else max(0, min(100, 100 - ((totale_uitgaven / inkomen) * 100)))
st.metric("💚 Gezondheidsscore", f"{gezondheid_score:.0f} / 100", help="Gebaseerd op verhouding tussen uitgaven en inkomen")

fig_score = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gezondheid_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Financiële gezondheid"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green" if gezondheid_score >= 60 else "orange" if gezondheid_score >= 30 else "red"},
        'steps': [
            {'range': [0, 30], 'color': "#ffcccc"},
            {'range': [30, 60], 'color': "#ffe0b3"},
            {'range': [60, 100], 'color': "#ccffcc"}
        ],
    }
))
st.plotly_chart(fig_score, use_container_width=True)

st.caption(
    "De score schat hoeveel van je inkomen beschikbaar blijft: 100 = alle uitgaven zijn 0; 0 = uitgaven ≥ inkomen. "
    "Richtwaarden: ≥60 goed, 30–60 aandacht, <30 risicovol."
)

# ============================================================
# 📋 Draaitabellen
# ============================================================

def _maak_pivot(data):
    data = data.copy()
    data["categorie"] = data["categorie"].astype(str).str.strip()
    data = data[data["categorie"].notna() & (data["categorie"] != "")]
    if data.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(
        data,
        index="categorie",
        columns="maand_naam",
        values="bedrag",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Totaal",
    )
    pivot = pivot.reindex(columns=[m for m in MAANDEN_NL if m in pivot.columns] + ["Totaal"]).sort_index()
    return pivot

st.subheader("📂 Overzicht per groep")
st.dataframe(_maak_pivot(df_loon).style.format("€ {:,.2f}"), use_container_width=True, height=300)
st.dataframe(_maak_pivot(df_vast).style.format("€ {:,.2f}"), use_container_width=True, height=300)
st.dataframe(_maak_pivot(df_variabel).style.format("€ {:,.2f}"), use_container_width=True, height=300)

# ============================================================
# 📈 Grafieken
# ============================================================
st.subheader("📈 Grafieken per maand en categorie")
inkomen_per_maand = df_loon.groupby("maand_naam")["bedrag"].sum().sort_index().fillna(0)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand, use_container_width=True)

kosten_per_maand = (
    df_gefilterd[df_gefilterd["vast/variabel"].isin(["Vast", "Variabel"])]
    .groupby(["maand_naam", "vast/variabel"])["bedrag"]
    .sum()
    .unstack()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📉 Vaste en variabele kosten per maand (saldo)")
st.line_chart(kosten_per_maand, use_container_width=True)

# ============================================================
# 🎯 Budgetdoelen per categorie — alleen VASTE kosten
# ============================================================
st.subheader(f"🎯 Budgetdoelen per categorie — {geselecteerde_maand}")

# 1) Alle vaste kosten-categorieën (altijd tonen, ook zonder uitgave deze maand)
vaste_cats = (
    df[df["vast/variabel"].astype(str).str.strip().str.title().eq("Vast")]["categorie"]
      .astype(str).str.strip().str.title().dropna().unique()
)

# 2) Werkelijke uitgaven (alleen vaste kosten) in geselecteerde maand -> Series met index=categorie
uitgaven_mnd_ser = (
    df_gefilterd[
        (df_gefilterd["maand_naam"] == geselecteerde_maand) &
        (~df_gefilterd["categorie"].astype(str).str.lower().eq("inkomsten loon")) &
        (df_gefilterd["vast/variabel"].astype(str).str.strip().str.title().eq("Vast"))
    ]
    .groupby("categorie")["bedrag"].sum().abs()
)

# 3) Robuuste serie met alle vaste categorieën (ontbrekend = 0)
uitgaven_full = (
    uitgaven_mnd_ser.reindex(sorted(vaste_cats))
                   .fillna(0.0)
                   .rename("uitgave")
)

# 4) Budget-editor (bewaar in session_state)
if "budget_state" not in st.session_state:
    st.session_state.budget_state = pd.DataFrame({"categorie": sorted(vaste_cats), "budget": np.nan})
else:
    prev = st.session_state.budget_state
    st.session_state.budget_state = (
        pd.DataFrame({"categorie": sorted(vaste_cats)})
        .merge(prev, on="categorie", how="left")
    )

with st.expander("✏️ Stel budgetten in (per categorie)", expanded=False):
    budget_df = st.data_editor(
        st.session_state.budget_state,
        num_rows="dynamic",
        hide_index=True,
        key="budget_editor",
        column_config={
            "categorie": st.column_config.TextColumn("Categorie", disabled=True),
            "budget": st.column_config.NumberColumn("Budget (€)", min_value=0.0, step=10.0, help="Maandbudget per categorie")
        }
    )
    st.session_state.budget_state = budget_df

# 5) Combineer budgetten met uitgaven (index-based join → geen KeyError)
budget_join = (
    st.session_state.budget_state.set_index("categorie")
                                 .join(uitgaven_full, how="left")
                                 .reset_index()
)

# 6) Tabel met status
budget_join["budget"] = pd.to_numeric(budget_join["budget"], errors="coerce")
budget_join["uitgave"] = pd.to_numeric(budget_join["uitgave"], errors="coerce").fillna(0)
budget_join["verschil"] = budget_join["budget"] - budget_join["uitgave"]

tabel = budget_join.assign(
    Budget=budget_join["budget"].apply(lambda x: euro(x) if pd.notna(x) else "—"),
    Uitgave=budget_join["uitgave"].apply(euro),
    **{"Δ (budget - uitgave)": budget_join["verschil"].apply(lambda x: euro(x) if pd.notna(x) else "—")},
    Status=np.where(
        budget_join["budget"].notna() & (budget_join["uitgave"] > budget_join["budget"]),
        "🚨 Over budget",
        np.where(budget_join["budget"].notna(), "✅ Binnen budget", "—")
    )
)
kolommen = [k for k in ["categorie", "Budget", "Uitgave", "Δ (budget - uitgave)", "Status"] if k in tabel.columns]
st.dataframe(tabel.loc[:, kolommen].rename(columns={"categorie": "Categorie"}), use_container_width=True)

# 7) Grafiek: Uitgave vs Budget (alleen rijen met budget ingevuld)
b_plot = budget_join.dropna(subset=["budget"]).copy()
if not b_plot.empty:
    b_plot = b_plot.sort_values("uitgave", ascending=False)
    fig_b = px.bar(
        b_plot.melt(id_vars=["categorie"], value_vars=["uitgave", "budget"], var_name="type", value_name="€"),
        x="categorie", y="€", color="type", barmode="group",
        title=f"Uitgaven vs. Budget — {geselecteerde_maand}", labels={"categorie": "Categorie"}
    )
    st.plotly_chart(fig_b, use_container_width=True)

# 8) Meldingen bij overschrijding
overs = budget_join[budget_join["budget"].notna() & (budget_join["uitgave"] > budget_join["budget"])].copy()
if not overs.empty:
    st.error(f"🚨 {len(overs)} categorie(ën) boven budget: " + ", ".join(overs["categorie"].tolist()))

# ============================================================
# 🔮 Prognose: einde van de maand (standaard tempo)
# ============================================================
st.subheader("🔮 Prognose einde van de maand")
if not df_maand.empty:
    laatste_datum = df_maand["datum"].max()
    jaar, mnd = laatste_datum.year, laatste_datum.month
    mask_ym = (
        (df_gefilterd["datum"].dt.year == jaar) &
        (df_gefilterd["datum"].dt.month == mnd) &
        (~df_gefilterd["categorie"].astype(str).str.lower().eq("inkomsten loon"))
    )
    df_ym = df_gefilterd[mask_ym].copy()

    if not df_ym.empty:
        uitg_tmv = abs(df_ym[df_ym["datum"] <= laatste_datum]["bedrag"].sum())
        dag_nr = int(laatste_datum.day)
        dagen_in_maand = int(calendar.monthrange(jaar, mnd)[1])
        proj = (uitg_tmv / max(dag_nr, 1)) * dagen_in_maand

        c1, c2, c3 = st.columns(3)
        c1.metric("Uitgaven tot en met vandaag", euro(uitg_tmv))
        c2.metric("Voorspelling maandtotaal", euro(proj))
        c3.metric("Nog te verwachten", euro(proj - uitg_tmv))

        totaal_budget = pd.to_numeric(budget_join["budget"], errors="coerce").sum(skipna=True)
        if not np.isnan(totaal_budget) and totaal_budget > 0:
            if proj > totaal_budget:
                st.error(f"⚠️ Verwachte uitgaven ({euro(proj)}) liggen boven totaalbudget ({euro(totaal_budget)}).")
            else:
                st.success(f"✅ Verwachte uitgaven ({euro(proj)}) liggen binnen totaalbudget ({euro(totaal_budget)}).")
    else:
        st.info("ℹ️ Geen uitgaven gevonden voor de gekozen jaar-maand.")
else:
    st.info("ℹ️ Geen data in de geselecteerde maand voor prognose.")

# ============================================================
# 📊 Vergelijking: uitgaven per maand
# ============================================================
st.subheader("📊 Vergelijking: Uitgaven per maand")
df_alleen_uitgaven = df_gefilterd[~df_gefilterd["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
df_alleen_uitgaven["maand_naam"] = df_alleen_uitgaven["maand_naam"].astype(maand_type)

totaal_per_maand = df_alleen_uitgaven.groupby("maand_naam")["bedrag"].sum().reindex(MAANDEN_NL).dropna().abs()

if geselecteerde_maand in totaal_per_maand.index:
    huidige_maand_bedrag = totaal_per_maand[geselecteerde_maand]
    overige_maanden = totaal_per_maand.drop(geselecteerde_maand)
    gemiddeld_bedrag = overige_maanden.mean() if not overige_maanden.empty else None
    if (gemiddeld_bedrag is not None) and (not np.isnan(gemiddeld_bedrag)) and (gemiddeld_bedrag != 0):
        verschil_pct = ((huidige_maand_bedrag - gemiddeld_bedrag) / gemiddeld_bedrag) * 100
        if verschil_pct > 20:
            st.error(f"🔺 Deze maand **{verschil_pct:.1f}% meer** uitgegeven dan gemiddeld. 🔴")
        elif verschil_pct < -20:
            st.success(f"🔻 Goed bezig! **{abs(verschil_pct):.1f}% minder** dan gemiddeld. 💚")
        else:
            st.info(f"⚖️ Uitgaven liggen rond het gemiddelde ({verschil_pct:.1f}%).")
    else:
        st.info("ℹ️ Niet genoeg of geen variatie om te vergelijken.")

uitgaven_per_maand = df_alleen_uitgaven.groupby("maand_naam")["bedrag"].sum().reindex(MAANDEN_NL).fillna(0).abs()
fig_vergelijking = px.bar(
    uitgaven_per_maand.reset_index(),
    x="maand_naam", y="bedrag",
    labels={"maand_naam": "Maand", "bedrag": "Uitgaven (€)"},
    title="Totale uitgaven per maand",
    text_auto=".2s",
)
fig_vergelijking.update_layout(xaxis_title="Maand", yaxis_title="€")
st.plotly_chart(fig_vergelijking, use_container_width=True)

# ============================================================
# 📅 Kalenderweergave
# ============================================================
st.subheader("📅 Dagelijkse uitgaven (kalenderweergave)")
df_kalender = df_gefilterd[~df_gefilterd["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
dagelijkse_uitgaven = df_kalender.groupby(pd.to_datetime(df_kalender["datum"].dt.date))["bedrag"].sum().abs()
dagelijkse_uitgaven.index = pd.to_datetime(dagelijkse_uitgaven.index)

if dagelijkse_uitgaven.empty:
    st.info("ℹ️ Geen uitgaven om weer te geven.")
else:
    try:
        import calplot
        fig, ax = calplot.calplot(dagelijkse_uitgaven, cmap="Reds", colorbar=True, suptitle="Uitgaven per dag", figsize=(10, 3))
        st.pyplot(fig)
    except Exception:
        st.info("📅 Calplot niet beschikbaar; fallback naar Plotly.")
        heat = dagelijkse_uitgaven.rename("bedrag").reset_index(names="datum")
        fig = px.density_heatmap(heat, x="datum", y=heat["datum"].dt.year.astype(str), z="bedrag", nbinsx=53, title="Uitgaven per dag (heatmap)")
        fig.update_yaxes(title="Jaar")
        fig.update_xaxes(title="Datum")
        st.plotly_chart(fig, use_container_width=True)
