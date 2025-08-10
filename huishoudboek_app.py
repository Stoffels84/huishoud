import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
import calendar

# ----------------------------
# 🔧 Pagina-instellingen
# ----------------------------
st.set_page_config(page_title="Huishoudboekje", layout="wide")
st.title("📊 Huishoudboekje Dashboard")

# ----------------------------
# 📅 Maanden in het Nederlands
# ----------------------------
MAANDEN_NL = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December"
]
maand_type = CategoricalDtype(categories=MAANDEN_NL, ordered=True)

# ----------------------------
# 💶 Helper: euro-format
# ----------------------------
def euro(x):
    try:
        return f"€ {x:,.2f}"
    except Exception:
        return "€ 0,00"

# ----------------------------
# 🧮 Helper voor percentages
# ----------------------------
def pct(value, total, *, signed=False, absolute=False):
    """Format percentage. signed=True => +/-, absolute=True => |value|/total."""
    if total is None or total == 0 or pd.isna(total):
        return "—"
    num = abs(value) if absolute else value
    p = (num / total) * 100
    return f"{p:+.1f}%" if signed else f"{p:.1f}%"

# ----------------------------
# 📥 Data inladen (met optionele upload)
# ----------------------------
with st.sidebar:
    upload = st.file_uploader("📥 Laad Excel (optioneel)", type=["xlsx", "xlsm"])

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
    df["maand_naam"] = df["maand"].apply(lambda m: MAANDEN_NL[m-1] if pd.notnull(m) else "")
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

# ----------------------------
# 📅 Filters (met Reset)
# ----------------------------
with st.sidebar:
    st.header("📅 Filter op periode")
    if "default_start" not in st.session_state:
        st.session_state.default_start = df["datum"].min().date()
        st.session_state.default_end = df["datum"].max().date()
        st.session_state.start_datum = st.session_state.default_start
        st.session_state.eind_datum = st.session_state.default_end

    c1, c2 = st.columns([3, 1])
    with c1:
        start_datum = st.date_input("Van", st.session_state.get("start_datum", st.session_state.default_start))
        eind_datum = st.date_input("Tot", st.session_state.get("eind_datum", st.session_state.default_end))
    with c2:
        if st.button("🔄 Reset"):
            st.session_state.start_datum = st.session_state.default_start
            st.session_state.eind_datum = st.session_state.default_end
            st.rerun()

# Update session state als user inputs wijzigt
st.session_state.start_datum = start_datum
st.session_state.eind_datum = eind_datum

# Filter toepassen
df_filtered = df[(df["datum"] >= pd.to_datetime(start_datum)) &
                 (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_filtered["maand_naam"] = df_filtered["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))
if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# Maandkeuze: chronologisch en alleen aanwezige maanden
present = set(df_filtered["maand_naam"].dropna().astype(str).tolist())
beschikbare_maanden = [m for m in MAANDEN_NL if m in present]
default_maand = beschikbare_maanden[-1] if beschikbare_maanden else MAANDEN_NL[0]

with st.sidebar:
    geselecteerde_maand = st.selectbox(
        "📆 Kies een maand voor uitgavenanalyse",
        beschikbare_maanden,
        index=(beschikbare_maanden.index(default_maand) if beschikbare_maanden else 0)
    )

# ----------------------------
# 📅 Maand-metrics (saldo)
# ----------------------------
st.subheader(f"📆 Overzicht voor {geselecteerde_maand}")
df_maand = df_filtered[df_filtered["maand_naam"] == geselecteerde_maand].copy()

is_loon = df_maand["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon_m = df_maand[is_loon]
df_vast_m = df_maand[df_maand["vast/variabel"] == "Vast"]
df_variabel_m = df_maand[df_maand["vast/variabel"] == "Variabel"]

inkomen_m = df_loon_m["bedrag"].sum()
vast_saldo_m = df_vast_m["bedrag"].sum()
variabel_saldo_m = df_variabel_m["bedrag"].sum()
totaal_saldo_m = inkomen_m + vast_saldo_m + variabel_saldo_m

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📈 Inkomen", euro(inkomen_m), "—")
col_m2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo_m), f"{pct(vast_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo_m), f"{pct(variabel_saldo_m, inkomen_m, absolute=True)} van inkomen")
col_m4.metric("💰 Netto saldo maand", euro(totaal_saldo_m), f"{pct(totaal_saldo_m, inkomen_m, signed=True)} van inkomen")

# ----------------------------
# 📊 Financiële metrics (gehele periode)
# ----------------------------
is_loon_all = df_filtered["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_filtered[is_loon_all]
df_vast = df_filtered[df_filtered["vast/variabel"] == "Vast"]
df_variabel = df_filtered[df_filtered["vast/variabel"] == "Variabel"]

inkomen = df_loon["bedrag"].sum()
vast_saldo = df_vast["bedrag"].sum()
variabel_saldo = df_variabel["bedrag"].sum()
totaal_saldo = inkomen + vast_saldo + variabel_saldo

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Inkomen", euro(inkomen), "—")
col2.metric("📌 Vaste kosten (aandeel)", euro(vast_saldo), f"{pct(vast_saldo, inkomen, absolute=True)} van inkomen")
col3.metric("📎 Variabele kosten (aandeel)", euro(variabel_saldo), f"{pct(variabel_saldo, inkomen, absolute=True)} van inkomen")
col4.metric("💰 Totaal saldo", euro(totaal_saldo), f"{pct(totaal_saldo, inkomen, signed=True)} van inkomen")

# ----------------------------
# 💡 Financiële gezondheidsscore
# ----------------------------
st.subheader("💡 Financiële Gezondheid")
totale_uitgaven = abs(vast_saldo) + abs(variabel_saldo)
if inkomen > 0:
    gezondheid_score = 100 - ((totale_uitgaven / inkomen) * 100)
    gezondheid_score = max(0, min(100, gezondheid_score))
else:
    gezondheid_score = 0

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
    "De score schat hoeveel van je inkomen beschikbaar blijft: "
    "100 = alle uitgaven zijn 0; 0 = uitgaven ≥ inkomen. "
    "Richtwaarden: ≥60 goed, 30–60 aandacht, <30 risicovol."
)

# ----------------------------
# 📋 Draaitabellen
# ----------------------------
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

def toon_draaitabel(data, titel):
    pivot = _maak_pivot(data)
    if pivot.empty:
        st.info(f"ℹ️ Geen gegevens beschikbaar voor: {titel}")
        return
    st.markdown(f"### {titel}")
    st.dataframe(pivot.style.format("€ {:,.2f}"), use_container_width=True, height=400)

st.subheader("📂 Overzicht per groep")
toon_draaitabel(df_loon, "💼 Inkomsten: Loon")
toon_draaitabel(df_vast, "📌 Vaste kosten")
toon_draaitabel(df_variabel, "📎 Variabele kosten")

# ----------------------------
# 📈 Grafieken
# ----------------------------
st.subheader("📈 Grafieken per maand en categorie")
inkomen_per_maand = df_loon.groupby("maand_naam")["bedrag"].sum().sort_index().fillna(0)
st.markdown("#### 📈 Inkomen per maand")
st.line_chart(inkomen_per_maand, use_container_width=True)

kosten_per_maand = (
    df_filtered[df_filtered["vast/variabel"].isin(["Vast", "Variabel"])]
    .groupby(["maand_naam", "vast/variabel"])["bedrag"]
    .sum()
    .unstack()
    .sort_index()
    .fillna(0)
)
st.markdown("#### 📉 Vaste en variabele kosten per maand (saldo)")
st.line_chart(kosten_per_maand, use_container_width=True)

# Donut: uitgaven per categorie (absolute bedragen)
st.subheader(f"🍩 Uitgaven per categorie in {geselecteerde_maand} (excl. 'Inkomsten Loon')")
df_donut = df_filtered[
    (df_filtered["maand_naam"] == geselecteerde_maand) &
    (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
]
if df_donut.empty:
    st.info("ℹ️ Geen uitgaven gevonden voor deze maand.")
else:
    donut_data = df_donut.groupby("categorie")["bedrag"].sum().abs().reset_index()
    donut_data = donut_data.sort_values("bedrag", ascending=False)
    top_n = 8
    if len(donut_data) > top_n:
        rest = donut_data.iloc[top_n:]["bedrag"].sum()
        donut_data = pd.concat([
            donut_data.iloc[:top_n],
            pd.DataFrame([{"categorie": "Overig", "bedrag": rest}])
        ], ignore_index=True)
    fig = px.pie(donut_data, names="categorie", values="bedrag", hole=0.4,
                 title=f"Verdeling uitgaven in {geselecteerde_maand}")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 🎯 Budgetdoelen per categorie (voor geselecteerde maand)
# ----------------------------
st.subheader(f"🎯 Budgetdoelen per categorie — {geselecteerde_maand}")
# Bereken uitgaven per categorie (absolute bedragen) voor de geselecteerde maand
uitgaven_mnd = (
    df_filtered[
        (df_filtered["maand_naam"] == geselecteerde_maand) &
        (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
    ]
    .groupby("categorie")["bedrag"].sum().abs().sort_values(ascending=False)
)

cats = list(uitgaven_mnd.index)
if "budget_state" not in st.session_state:
    st.session_state.budget_state = pd.DataFrame({"categorie": cats, "budget": np.nan})

# Sync categorieën (voeg nieuwe toe, behoud bestaande budgetten)
prev = st.session_state.budget_state
merged = (
    pd.DataFrame({"categorie": cats})
    .merge(prev, on="categorie", how="left")
)
st.session_state.budget_state = merged

with st.expander("✏️ Stel budgetten in (per categorie)", expanded=False):
    budget_df = st.data_editor(
        st.session_state.budget_state,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "categorie": st.column_config.TextColumn("Categorie", disabled=True),
            "budget": st.column_config.NumberColumn("Budget (€)", min_value=0.0, step=10.0, help="Maandbudget per categorie")
        }
    )
    st.session_state.budget_state = budget_df

# Combineer met werkelijke uitgaven
budget_join = (
    budget_df.set_index("categorie").join(uitgaven_mnd.rename("uitgave"), how="outer")
    .reset_index()
)
budget_join["budget"] = pd.to_numeric(budget_join["budget"], errors="coerce")
budget_join["uitgave"] = pd.to_numeric(budget_join["uitgave"], errors="coerce").fillna(0)
budget_join["verschil"] = budget_join["budget"] - budget_join["uitgave"]

# Tabel met status
st.dataframe(
    budget_join.assign(
        **{
            "Budget": budget_join["budget"].apply(lambda x: euro(x) if pd.notna(x) else "—"),
            "Uitgave": budget_join["uitgave"].apply(euro),
            "Δ (budget - uitgave)": budget_join["verschil"].apply(lambda x: euro(x) if pd.notna(x) else "—"),
            "Status": np.where(
                budget_join["budget"].notna() & (budget_join["uitgave"] > budget_join["budget"]),
                "🚨 Over budget",
                np.where(budget_join["budget"].notna(), "✅ Binnen budget", "—")
            ),
        }
    )[["categorie", "Budget", "Uitgave", "Δ (budget - uitgave)", "Status"]]
    .rename(columns={"categorie": "Categorie"}),
    use_container_width=True,
)

# Grafiek: Uitgave vs Budget (alleen rijen met budget ingevuld)
b_plot = budget_join.dropna(subset=["budget"]).copy()
if not b_plot.empty:
    b_plot = b_plot.sort_values("uitgave", ascending=False)
    fig_b = px.bar(
        b_plot.melt(id_vars=["categorie"], value_vars=["uitgave", "budget"], var_name="type", value_name="€"),
        x="categorie", y="€", color="type", barmode="group",
        title=f"Uitgaven vs. Budget — {geselecteerde_maand}", labels={"categorie": "Categorie"}
    )
    st.plotly_chart(fig_b, use_container_width=True)

# Meldingen bij overschrijding
overs = budget_join[budget_join["budget"].notna() & (budget_join["uitgave"] > budget_join["budget"])].copy()
if not overs.empty:
    st.error(f"🚨 {len(overs)} categorie(ën) boven budget: " + ", ".join(overs["categorie"].tolist()))

# ----------------------------
# 🔮 Prognose: einde van de maand (op basis van tempo tot nu toe)
# ----------------------------
st.subheader("🔮 Prognose einde van de maand")
# Kies de meest recente kalendermaand (jaar+maand) die overeenkomt met de geselecteerde maand
if not df_maand.empty:
    laatste_datum = df_maand["datum"].max()
    jaar, mnd = laatste_datum.year, laatste_datum.month
    # Filter alleen uitgaven (excl. inkomsten) voor die specifieke jaar-maand
    mask_ym = (df_filtered["datum"].dt.year == jaar) & (df_filtered["datum"].dt.month == mnd) & (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
    df_ym = df_filtered[mask_ym].copy()

    if not df_ym.empty:
        uitg_tmv = df_ym[df_ym["datum"] <= laatste_datum]["bedrag"].sum()
        uitg_tmv = abs(uitg_tmv)
        dag_nr = laatste_datum.day
        dagen_in_maand = calendar.monthrange(jaar, mnd)[1]
        proj = (uitg_tmv / max(dag_nr, 1)) * dagen_in_maand

        c1, c2, c3 = st.columns(3)
        c1.metric("Uitgaven tot en met vandaag", euro(uitg_tmv))
        c2.metric("Voorspelling maandtotaal", euro(proj))
        diff = proj - uitg_tmv
        c3.metric("Nog te verwachten", euro(diff))

        # Als er totaalbudget (som van ingevulde budgetten) is, vergelijk
        totaal_budget = budget_join["budget"].sum(skipna=True)
        if not np.isnan(totaal_budget) and totaal_budget > 0:
            if proj > totaal_budget:
                st.error(f"⚠️ Verwachte uitgaven ({euro(proj)}) liggen boven totaalbudget ({euro(totaal_budget)}).")
            else:
                st.success(f"✅ Verwachte uitgaven ({euro(proj)}) liggen binnen totaalbudget ({euro(totaal_budget)}).")
    else:
        st.info("ℹ️ Geen uitgaven gevonden voor de gekozen jaar-maand.")
else:
    st.info("ℹ️ Geen data in de geselecteerde maand voor prognose.")

# ----------------------------
# 📊 Vergelijking: uitgaven per maand
# ----------------------------
st.subheader("📊 Vergelijking: Uitgaven per maand")
df_alleen_uitgaven = df_filtered[~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
df_alleen_uitgaven["maand_naam"] = df_alleen_uitgaven["maand_naam"].astype(maand_type)

totaal_per_maand = (
    df_alleen_uitgaven.groupby("maand_naam")["bedrag"].sum().reindex(MAANDEN_NL).dropna().abs()
)

if geselecteerde_maand in totaal_per_maand.index:
    huidige_maand_bedrag = totaal_per_maand[geselecteerde_maand]
    overige_maanden = totaal_per_maand.drop(geselecteerde_maand)
    gemiddeld_bedrag = overige_maanden.mean() if not overige_maanden.empty else None
    if (gemiddeld_bedrag is not None) and (not np.isnan(gemiddeld_bedrag)) and (gemiddeld_bedrag != 0):
        verschil_pct = ((huidige_maand_bedrag - gemiddeld_bedrag) / gemiddeld_bedrag) * 100
        if verschil_pct > 20:
            st.error(f"🔺 Je hebt deze maand **{verschil_pct:.1f}% meer** uitgegeven dan gemiddeld. 🔴")
        elif verschil_pct < -20:
            st.success(f"🔻 Goed bezig! Je gaf deze maand **{abs(verschil_pct):.1f}% minder** uit dan gemiddeld. 💚")
        else:
            st.info(f"⚖️ Uitgaven liggen rond het gemiddelde ({verschil_pct:.1f}%).")
    else:
        st.info("ℹ️ Niet genoeg of geen variatie om te vergelijken.")

uitgaven_per_maand = (
    df_alleen_uitgaven.groupby("maand_naam")["bedrag"].sum().reindex(MAANDEN_NL).fillna(0).abs()
)
fig_vergelijking = px.bar(
    uitgaven_per_maand.reset_index(),
    x="maand_naam", y="bedrag",
    labels={"maand_naam": "Maand", "bedrag": "Uitgaven (€)"},
    title="Totale uitgaven per maand",
    text_auto=".2s"
)
fig_vergelijking.update_layout(xaxis_title="Maand", yaxis_title="€")
st.plotly_chart(fig_vergelijking, use_container_width=True)

# ----------------------------
# 📅 Kalenderweergave
# ----------------------------
st.subheader("📅 Dagelijkse uitgaven (kalenderweergave)")
df_kalender = df_filtered[~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")].copy()
dagelijkse_uitgaven = df_kalender.groupby(pd.to_datetime(df_kalender["datum"].dt.date))["bedrag"].sum().abs()
dagelijkse_uitgaven.index = pd.to_datetime(dagelijkse_uitgaven.index)

if dagelijkse_uitgaven.empty:
    st.info("ℹ️ Geen uitgaven om weer te geven.")
else:
    try:
        import calplot
        fig, ax = calplot.calplot(dagelijkse_uitgaven, cmap="Reds", colorbar=True,
                                  suptitle="Uitgaven per dag", figsize=(10, 3))
        st.pyplot(fig)
    except Exception:
        st.info("📅 Calplot niet beschikbaar; fallback naar Plotly.")
        heat = dagelijkse_uitgaven.rename("bedrag").reset_index(names="datum")
        fig = px.density_heatmap(
            heat, x="datum", y=heat["datum"].dt.year.astype(str), z="bedrag",
            nbinsx=53, title="Uitgaven per dag (heatmap)"
        )
        fig.update_yaxes(title="Jaar")
        fig.update_xaxes(title="Datum")
        st.plotly_chart(fig, use_container_width=True)
