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
    try:
        return f"€ {x:,.2f}"
    except Exception:
        return "€ 0,00"


def pct(value, total, *, signed=False, absolute=False):
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
    df["jaar"] = df["datum"].dt.year
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

# Filter toepassen
df_filtered = df[(df["datum"] >= pd.to_datetime(start_datum)) & (df["datum"] <= pd.to_datetime(eind_datum))].copy()
df_filtered["maand_naam"] = df_filtered["maand_naam"].astype(maand_type)

st.write("🔍 Aantal gefilterde rijen:", len(df_filtered))
if df_filtered.empty:
    st.warning("⚠️ Geen data in deze periode.")
    st.stop()

# Maandkeuze
aanwezig = set(df_filtered["maand_naam"].dropna().astype(str).tolist())
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
# 🔁 Trend t.o.v. vorige maand (delta in st.metric)
# ============================================================
st.caption("Trend t.o.v. vorige maand")

if not df_maand.empty:
    # Laatste datum in de geselecteerde maand
    ref = df_maand["datum"].max()

    # Vorige maand bepalen (met jaarwisseling)
    prev_year, prev_month = (ref.year - 1, 12) if ref.month == 1 else (ref.year, ref.month - 1)

    # Dataframe van de vorige maand binnen de huidige filterrange
    prev_mask = (df_filtered["datum"].dt.year == prev_year) & (df_filtered["datum"].dt.month == prev_month)
    df_prev = df_filtered[prev_mask].copy()

    # Helper om totals te pakken (zelfde logica als je KPI's)
    def total_of(dfin, *, cat=None, vv=None):
        d = dfin.copy()
        cat_col = d["categorie"].astype(str).str.strip().str.lower()
        if cat == "inkomsten":
            d = d[cat_col.eq("inkomsten loon")]
        elif cat == "uitgaven":
            d = d[~cat_col.eq("inkomsten loon")]
        if vv is not None:
            d = d[d["vast/variabel"].astype(str).str.strip().str.title().eq(vv)]
        return d["bedrag"].sum()

    # Waarden vorige maand
    prev_ink = total_of(df_prev, cat="inkomsten")
    prev_vast = total_of(df_prev, vv="Vast")
    prev_var  = total_of(df_prev, vv="Variabel")
    prev_net  = prev_ink + prev_vast + prev_var

    # Deltas (huidig - vorige maand)
    delta_ink = inkomen_m - prev_ink
    delta_vast = vast_saldo_m - prev_vast
    delta_var = variabel_saldo_m - prev_var
    delta_net = totaal_saldo_m - prev_net

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📈 Inkomen (trend)", euro(inkomen_m), delta=euro(delta_ink))
    c2.metric("📌 Vaste kosten (trend)", euro(vast_saldo_m), delta=euro(delta_vast))
    c3.metric("📎 Variabele kosten (trend)", euro(variabel_saldo_m), delta=euro(delta_var))
    c4.metric("💰 Netto saldo maand (trend)", euro(totaal_saldo_m), delta=euro(delta_net))
else:
    st.info("ℹ️ Geen data voor de geselecteerde maand om een trend te tonen.")







# ============================================================
# 📊 Financiële metrics (gehele periode)
# ============================================================
is_loon_all = df_filtered["categorie"].astype(str).str.strip().str.lower().eq("inkomsten loon")
df_loon = df_filtered[is_loon_all]
df_vast = df_filtered[df_filtered["vast/variabel"] == "Vast"]
df_variabel = df_filtered[df_filtered["vast/variabel"] == "Variabel"]

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
# 🎯 Budgetdoelen per categorie — alleen VASTE kosten
# ============================================================
st.subheader(f"🎯 Budgetdoelen per categorie — {geselecteerde_maand}")

# Alle vaste categorieën (uit hele dataset)
vaste_cats = (
    df[df["vast/variabel"].astype(str).str.strip().str.title().eq("Vast")]["categorie"]
      .astype(str).str.strip().str.title().dropna().unique()
)

# Uitgaven deze maand (alleen vaste) als Series
uitgaven_mnd_ser = (
    df_filtered[
        (df_filtered["maand_naam"] == geselecteerde_maand) &
        (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon")) &
        (df_filtered["vast/variabel"].astype(str).str.strip().str.title().eq("Vast"))
    ]
    .groupby("categorie")["bedrag"].sum().abs()
)

# 🔥 NIEUW: automatische budgetten = gemiddelde van voorgaande maanden
# Bepaal referentiemaand (gebruik laatste datum in de geselecteerde maand)
if df_maand.empty:
    gemiddelde_per_cat = pd.Series(dtype=float)
else:
    ref = df_maand["datum"].max()
    maand_start = pd.Timestamp(ref.year, ref.month, 1)

    # Historische vaste uitgaven vóór deze maand
    prev = df[(df["datum"] < maand_start) &
              (df["vast/variabel"].astype(str).str.strip().str.title() == "Vast") &
              (~df["categorie"].astype(str).str.lower().eq("inkomsten loon"))].copy()

    if prev.empty:
        gemiddelde_per_cat = pd.Series(dtype=float)
    else:
        # Som per maand & categorie → gemiddelde per categorie
        per_mnd_cat = prev.groupby([prev["datum"].dt.to_period("M"), "categorie"])['bedrag'].sum().abs()
        gemiddelde_per_cat = per_mnd_cat.groupby("categorie").mean()

# Serie met alle vaste categorieën voor huidige maand (ontbrekend = 0)
uitgaven_full = (
    uitgaven_mnd_ser.reindex(sorted(vaste_cats)).fillna(0.0).rename("uitgave")
)

# Budget-state initialiseren of bijwerken met defaults uit gemiddelde
if "budget_state" not in st.session_state:
    st.session_state.budget_state = pd.DataFrame({
        "categorie": sorted(vaste_cats),
        "budget": [float(gemiddelde_per_cat.get(cat, np.nan)) if not pd.isna(gemiddelde_per_cat.get(cat, np.nan)) else np.nan
                    for cat in sorted(vaste_cats)]
    })
else:
    prev_state = st.session_state.budget_state
    merged = pd.DataFrame({"categorie": sorted(vaste_cats)}).merge(prev_state, on="categorie", how="left")
    # Vul alleen lege budgetten met gemiddelde
    if not gemiddelde_per_cat.empty:
        merged.loc[merged["budget"].isna(), "budget"] = merged.loc[merged["budget"].isna(), "categorie"].map(gemiddelde_per_cat)
    st.session_state.budget_state = merged

with st.expander("✏️ Stel budgetten in (per categorie)", expanded=False):
    budget_df = st.data_editor(
        st.session_state.budget_state,
        num_rows="dynamic",
        hide_index=True,
        key="budget_editor",
        column_config={
            "categorie": st.column_config.TextColumn("Categorie", disabled=True),
            "budget": st.column_config.NumberColumn("Budget (€)", min_value=0.0, step=10.0, help="Auto = gemiddelde vorige maanden; aanpasbaar")
        }
    )
    st.session_state.budget_state = budget_df

# Join budget + uitgaven (index-based)
budget_join = (
    st.session_state.budget_state.set_index("categorie").join(uitgaven_full, how="left").reset_index()
)

budget_join["budget"] = pd.to_numeric(budget_join["budget"], errors="coerce")
budget_join["uitgave"] = pd.to_numeric(budget_join["uitgave"], errors="coerce").fillna(0)
budget_join["verschil"] = budget_join["budget"] - budget_join["uitgave"]

# Tabel weergeven
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

# Grafiek: Uitgave vs Budget
b_plot = budget_join.dropna(subset=["budget"]).copy()
if not b_plot.empty:
    b_plot = b_plot.sort_values("uitgave", ascending=False)
    fig_b = px.bar(
        b_plot.melt(id_vars=["categorie"], value_vars=["uitgave", "budget"], var_name="type", value_name="€"),
        x="categorie", y="€", color="type", barmode="group",
        title=f"Uitgaven vs. Budget — {geselecteerde_maand}", labels={"categorie": "Categorie"}
    )
    st.plotly_chart(fig_b, use_container_width=True)

# 🔮 Prognose einde van de maand — alleen budgetten (geen dagen/tempo)
st.subheader("🔮 Prognose einde van de maand")

if not df_maand.empty:
    laatste_datum = df_maand["datum"].max()
    jaar, mnd = laatste_datum.year, laatste_datum.month

    # Alle uitgaven (excl. inkomsten) in de geselecteerde jaar-maand
    mask_ym = (
        (df_filtered["datum"].dt.year == jaar)
        & (df_filtered["datum"].dt.month == mnd)
        & (~df_filtered["categorie"].astype(str).str.lower().eq("inkomsten loon"))
    )
    df_ym = df_filtered[mask_ym].copy()

    if not df_ym.empty:
        # Totaal uitgegeven tot en met vandaag
        uitg_tmv = abs(df_ym[df_ym["datum"] <= laatste_datum]["bedrag"].sum())

        # Reeds uitgegeven per categorie
        spent_per_cat = (
            df_ym[df_ym["datum"] <= laatste_datum]
            .groupby("categorie")["bedrag"].sum().abs()
        )

        # Budget per categorie uit editor
        budget_per_cat = (
            budget_join.set_index("categorie")["budget"].astype(float)
            if "budget" in budget_join.columns else pd.Series(dtype=float)
        )

        # Alleen budgetten: resterend = max(0, budget - spent); geen budget => 0
        resterend_per_cat = (budget_per_cat - spent_per_cat).clip(lower=0).fillna(0)

        # Prognose = al uitgegeven + wat er nog 'mag' volgens budget
        proj = float(uitg_tmv + resterend_per_cat.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Uitgaven tot en met vandaag", euro(uitg_tmv))
        c2.metric("Voorspelling maandtotaal", euro(proj))
        c3.metric("Nog te verwachten", euro(proj - uitg_tmv))

        # Vergelijk met totaalbudget
        totaal_budget = pd.to_numeric(budget_join["budget"], errors="coerce").sum(skipna=True)
        if not np.isnan(totaal_budget) and totaal_budget > 0:
            if proj > totaal_budget:
                st.error(f"⚠️ Verwachte uitgaven ({euro(proj)}) liggen boven totaalbudget ({euro(totaal_budget)}).")
            else:
                st.success(f"✅ Verwachte uitgaven ({euro(proj)}) liggen binnen totaalbudget ({euro(totaal_budget)}).")

        st.caption("Prognose gebaseerd uitsluitend op budgetten: resterend = max(0, budget − uitgegeven).")
    else:
        st.info("ℹ️ Geen uitgaven gevonden voor de gekozen jaar-maand.")
else:
    st.info("ℹ️ Geen data in de geselecteerde maand voor prognose.")
