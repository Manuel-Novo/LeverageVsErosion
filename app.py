import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass

# --- CONFIG & STYLE ---
st.set_page_config(page_title="Credit Simulator - Inflation Comparison", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { color: #1f1f1f !important; }
    [data-testid="stMetricLabel"] { color: #4b4b4b !important; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
    }
    .box-positive { background-color: #e8f8f0; border-left: 5px solid #00CC96; }
    .box-neutral { background-color: #fef9e7; border-left: 5px solid #f1c40f; }
    .box-negative { background-color: #fdedec; border-left: 5px solid #e74c3c; }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# --- HELPERS ---
def fmt(value: float) -> str:
    """Format number with space as thousands separator."""
    return f"{round(value, 2):,}".replace(",", " ")


@dataclass
class AmortResult:
    df: pd.DataFrame
    monthly_payment: float
    total_int_nom: float
    total_int_act: float
    total_cap_act: float
    total_rent_nom: float
    total_rent_act: float
    total_charges_nom: float
    total_charges_act: float


# --- CALCULATION FUNCTIONS ---
@st.cache_data
def calculate_amortization(
    initial_property_value: float,
    loan_amount: float,
    annual_rate: float,
    loan_years: int,
    projection_years: int,
    inflation_rate: float,
    real_estate_inflation: float,
    initial_monthly_rent: float,
    initial_property_tax: float,
    occupancy_rate: float,
    stock_return: float,
    down_payment: float,
) -> AmortResult:
    """Calculate amortization table with projection beyond loan term."""
    monthly_rate = (annual_rate / 100) / 12
    nb_months = loan_years * 12

    if annual_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** nb_months) / ((1 + monthly_rate) ** nb_months - 1)
    else:
        monthly_payment = loan_amount / nb_months

    # Pre-compute factors
    inf_factor = 1 + inflation_rate / 100
    immo_factor = 1 + real_estate_inflation / 100
    stock_monthly = 1 + (stock_return / 100) / 12
    occ_factor = occupancy_rate / 100

    remaining_principal = loan_amount
    cumul_interest = cumul_principal = 0.0
    cumul_interest_act = cumul_principal_act = 0.0
    cumul_rent_nom = cumul_rent_act = 0.0
    cumul_charges_nom = cumul_charges_act = 0.0

    stock_value_nom = down_payment
    total_invested_stock = down_payment

    data = []

    for year in range(1, projection_years + 1):
        interest_year = principal_year = 0.0
        is_loan_active = year <= loan_years

        for _ in range(12):
            if is_loan_active and remaining_principal > 0:
                i = remaining_principal * monthly_rate
                p = min(monthly_payment - i, remaining_principal)
                interest_year += i
                principal_year += p
                remaining_principal -= p

                stock_value_nom = stock_value_nom * stock_monthly + monthly_payment
                total_invested_stock += monthly_payment
            else:
                stock_value_nom = stock_value_nom * stock_monthly

        discount_factor = 1 / (inf_factor ** year)

        nominal_property_value = initial_property_value * (immo_factor ** year)
        actualized_property_value = nominal_property_value * discount_factor

        annual_rent_nom = (initial_monthly_rent * 12) * (inf_factor ** (year - 1)) * occ_factor
        annual_charge_nom = initial_property_tax * (inf_factor ** (year - 1))

        cumul_interest += interest_year
        cumul_principal += principal_year
        cumul_interest_act += interest_year * discount_factor
        cumul_principal_act += principal_year * discount_factor
        cumul_rent_nom += annual_rent_nom
        cumul_rent_act += annual_rent_nom * discount_factor
        cumul_charges_nom += annual_charge_nom
        cumul_charges_act += annual_charge_nom * discount_factor

        total_cost_act = cumul_interest_act + cumul_principal_act
        total_cost_nom = loan_amount + cumul_interest

        # Gains calculations
        net_gain_nom = nominal_property_value - total_cost_nom
        net_gain_act = actualized_property_value - total_cost_act
        net_net_immo_act = (actualized_property_value + cumul_rent_act) - (total_cost_act + cumul_charges_act)
        net_net_immo_nom = (nominal_property_value + cumul_rent_nom) - (total_cost_nom + cumul_charges_nom)

        # Year-over-year changes for inflection analysis
        yoy_property_change_act = actualized_property_value * (immo_factor / inf_factor - 1) if year > 1 else 0
        yoy_net_net_change = 0  # Will be calculated after

        stock_value_act = stock_value_nom * discount_factor
        stock_gain_act = stock_value_act - (total_invested_stock * discount_factor)
        stock_gain_nom = stock_value_nom - total_invested_stock

        data.append({
            "Year": year,
            "Loan Active": "Yes" if is_loan_active else "No",
            "Remaining Principal": round(max(0, remaining_principal), 2),
            # Property values
            "Nominal Property Value": round(nominal_property_value, 2),
            "Actualized Property Value": round(actualized_property_value, 2),
            # Costs
            "Cumul. Repayment (Actualized)": round(total_cost_act, 2),
            "Cumul. Repayment (Nominal)": round(total_cost_nom, 2),
            "Cumul. Rent (Actualized)": round(cumul_rent_act, 2),
            "Cumul. Rent (Nominal)": round(cumul_rent_nom, 2),
            "Cumul. Charges (Actualized)": round(cumul_charges_act, 2),
            "Cumul. Charges (Nominal)": round(cumul_charges_nom, 2),
            # Net gains (property - cost)
            "Net Gain (Nominal)": round(net_gain_nom, 2),
            "Net Gain (Actualized)": round(net_gain_act, 2),
            # Net Net gains (property + rent - cost - charges)
            "Net Net Real Estate (Act.)": round(net_net_immo_act, 2),
            "Net Net Real Estate (Nom.)": round(net_net_immo_nom, 2),
            # Stock
            "Stock Gain (Act.)": round(stock_gain_act, 2),
            "Stock Gain (Nom.)": round(stock_gain_nom, 2),
            "Stock Value (Nominal)": round(stock_value_nom, 2),
            "Stock Value (Actualized)": round(stock_value_act, 2),
            "Total Invested Stock (Nominal)": round(total_invested_stock, 2),
            # For inflection analysis
            "Discount Factor": round(discount_factor, 6),
        })

    df = pd.DataFrame(data)

    # Calculate YoY changes
    df["YoY Property Change (Act.)"] = df["Actualized Property Value"].diff().fillna(0).round(2)
    df["YoY Net Net Change (Act.)"] = df["Net Net Real Estate (Act.)"].diff().fillna(0).round(2)
    df["YoY Property Change (Nom.)"] = df["Nominal Property Value"].diff().fillna(0).round(2)
    df["YoY Net Net Change (Nom.)"] = df["Net Net Real Estate (Nom.)"].diff().fillna(0).round(2)

    # Annualized return rates
    df["Actualized Annual Return (%)"] = (df["YoY Property Change (Act.)"] / df["Actualized Property Value"].shift(1) * 100).fillna(0).round(2)
    df["Nominal Annual Return (%)"] = (df["YoY Property Change (Nom.)"] / df["Nominal Property Value"].shift(1) * 100).fillna(0).round(2)

    return AmortResult(
        df=df,
        monthly_payment=monthly_payment,
        total_int_nom=cumul_interest,
        total_int_act=cumul_interest_act,
        total_cap_act=cumul_principal_act,
        total_rent_nom=cumul_rent_nom,
        total_rent_act=cumul_rent_act,
        total_charges_nom=cumul_charges_nom,
        total_charges_act=cumul_charges_act,
    )


@st.cache_data
def find_optimal_exit(df: pd.DataFrame, loan_years: int) -> dict:
    """Find optimal exit points based on various metrics."""
    # Peak actualized property value
    peak_act_idx = df["Actualized Property Value"].idxmax()
    peak_act_year = df.loc[peak_act_idx, "Year"]
    peak_act_value = df.loc[peak_act_idx, "Actualized Property Value"]

    # Peak net net actualized
    peak_net_net_idx = df["Net Net Real Estate (Act.)"].idxmax()
    peak_net_net_year = df.loc[peak_net_net_idx, "Year"]
    peak_net_net_value = df.loc[peak_net_net_idx, "Net Net Real Estate (Act.)"]

    # First year where actualized value decreases
    df_after_loan = df[df["Year"] > loan_years]
    declining_years = df_after_loan[df_after_loan["YoY Property Change (Act.)"] < 0]
    first_decline_year = declining_years["Year"].iloc[0] if not declining_years.empty else None

    # Years where actualized return goes negative
    negative_return_years = df[df["Actualized Annual Return (%)"] < 0]
    first_negative_return = negative_return_years["Year"].iloc[0] if not negative_return_years.empty else None

    return {
        "peak_act_year": peak_act_year,
        "peak_act_value": peak_act_value,
        "peak_net_net_year": peak_net_net_year,
        "peak_net_net_value": peak_net_net_value,
        "first_decline_year": first_decline_year,
        "first_negative_return": first_negative_return,
    }


@st.cache_data
def compute_sensitivity_rate(
    property_value: float,
    loan_amount: float,
    loan_years: int,
    inflation_rate: float,
    real_estate_inflation: float,
    initial_rent: float,
    property_tax: float,
    occupancy_rate: float,
    stock_return: float,
    down_payment: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate_range = np.linspace(0, 15, 100)
    net_vals, net_net_vals = [], []

    final_act_value = property_value * ((1 + real_estate_inflation / 100) ** loan_years) / ((1 + inflation_rate / 100) ** loan_years)

    for r in rate_range:
        res = calculate_amortization(
            property_value, loan_amount, r, loan_years, loan_years, inflation_rate,
            real_estate_inflation, initial_rent, property_tax,
            occupancy_rate, stock_return, down_payment
        )
        net_vals.append(final_act_value - (res.total_int_act + res.total_cap_act))
        net_net_vals.append((final_act_value + res.total_rent_act) - (res.total_int_act + res.total_cap_act + res.total_charges_act))

    return rate_range, np.array(net_vals), np.array(net_net_vals)


@st.cache_data
def compute_sensitivity_market(
    property_value: float,
    loan_years: int,
    inflation_rate: float,
    total_cost_act: float,
    total_rent_act: float,
    total_charges_act: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    market_range = np.linspace(-5, 15, 100)
    inf_denom = (1 + inflation_rate / 100) ** loan_years

    final_act_value = property_value * ((1 + market_range / 100) ** loan_years) / inf_denom
    net_vals = final_act_value - total_cost_act
    net_net_vals = (final_act_value + total_rent_act) - (total_cost_act + total_charges_act)

    return market_range, net_vals, net_net_vals


@st.cache_data
def compute_sensitivity_inflation(
    property_value: float,
    loan_amount: float,
    annual_rate: float,
    loan_years: int,
    real_estate_inflation: float,
    initial_rent: float,
    property_tax: float,
    occupancy_rate: float,
    stock_return: float,
    down_payment: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inf_range = np.linspace(0, 10, 100)
    net_vals, net_net_vals = [], []

    immo_growth = (1 + real_estate_inflation / 100) ** loan_years

    for i in inf_range:
        res = calculate_amortization(
            property_value, loan_amount, annual_rate, loan_years, loan_years, i,
            real_estate_inflation, initial_rent, property_tax,
            occupancy_rate, stock_return, down_payment
        )
        final_act_value = property_value * immo_growth / ((1 + i / 100) ** loan_years)
        net_vals.append(final_act_value - (res.total_int_act + res.total_cap_act))
        net_net_vals.append((final_act_value + res.total_rent_act) - (res.total_int_act + res.total_cap_act + res.total_charges_act))

    return inf_range, np.array(net_vals), np.array(net_net_vals)


def create_donut(labels: list, values: list, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker_colors=["#00CC96", "#EF553B"])])
    fig.update_layout(
        title=dict(text=title, x=0.5),
        margin=dict(t=50, b=20),
        height=350,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_sensitivity_chart(x: np.ndarray, y_net: np.ndarray, y_net_net: np.ndarray, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_net, name="Net", line=dict(color=color, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=y_net_net, name="Net Net", line=dict(color=color, width=4)))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="(%)",
        height=400,
        legend=dict(orientation="h", y=-0.2),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.header("🏠 Your Project")
    property_value = st.number_input("Property Value (€)", value=250000, step=5000)
    down_payment = st.number_input("Down Payment (€)", value=50000, step=1000)
    loan_amount = property_value - down_payment
    st.info(f"Loan Amount: **{fmt(loan_amount)} €**")

    st.divider()
    st.header("💳 Loan Parameters")
    annual_rate = st.slider("Annual Interest Rate (%)", 0.0, 10.0, 3.5, step=0.1)
    loan_years = st.number_input("Loan Duration (years)", value=20, min_value=1, max_value=40)

    st.divider()
    st.header("📊 Projection Horizon")
    projection_years = st.slider(
        "Projection Duration (years)",
        min_value=int(loan_years),
        max_value=40,
        value=min(int(loan_years) + 15, 40),
        help="Extend beyond loan term to see how inflation erodes value over time"
    )

    st.divider()
    st.header("💰 Income & Expenses")
    initial_rent = st.number_input("Initial Monthly Rent (€)", value=1000, step=50)
    occupancy_rate = st.slider("Occupancy Rate (%)", 0, 100, 95)
    property_tax = st.number_input("Annual Property Tax (€)", value=1200, step=100)

    st.divider()
    st.header("📈 Stock Market Return")
    stock_return = st.slider("Annual Stock Return (%)", 0.0, 15.0, 7.0, step=0.1, help="Expected average annual return.")

    st.divider()
    st.header("📊 Economic Context")
    inflation_rate = st.slider("Expected Annual Inflation (%)", 0.0, 10.0, 2.0, step=0.1)
    real_estate_inflation = st.slider("Real Estate Appreciation (%/year)", -5.0, 10.0, 1.5, step=0.1)

# --- MAIN PAGE ---
st.title("🚀 Credit Analysis: Nominal vs Purchasing Power")

if loan_amount > 0:
    result = calculate_amortization(
        property_value, loan_amount, annual_rate, int(loan_years), projection_years,
        inflation_rate, real_estate_inflation, initial_rent, property_tax,
        occupancy_rate, stock_return, down_payment
    )
    df = result.df
    exit_analysis = find_optimal_exit(df, int(loan_years))

    # Values at loan end
    df_at_loan_end = df[df["Year"] == loan_years].iloc[0]
    final_nom_at_loan = df_at_loan_end["Nominal Property Value"]
    final_act_at_loan = df_at_loan_end["Actualized Property Value"]
    total_cost_act = result.total_cap_act + result.total_int_act

    # Values at projection end
    final_nom = df["Nominal Property Value"].iloc[-1]
    final_act = df["Actualized Property Value"].iloc[-1]

    net_balance_at_loan = final_act_at_loan - total_cost_act
    net_net_at_loan = df_at_loan_end["Net Net Real Estate (Act.)"]

    # Value erosion after loan
    value_erosion = final_act_at_loan - final_act if projection_years > loan_years else 0

    # SECTION 1: QUICK SUMMARY
    st.subheader("📊 Financial Summary")

    box_class = "box-positive" if net_net_at_loan > 0 else "box-negative"
    st.markdown(f"""
    <div class="highlight-box {box_class}">
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div>
                <h3 style='margin:0; color:#4b4b4b; font-size: 1.1em;'>Net Balance (Equity)</h3>
                <h2 style='margin:0; color:#1f1f1f;'>{round(net_balance_at_loan, 0):,} €</h2>
            </div>
            <div style="width: 2px; height: 50px; background-color: #d0d0d0;"></div>
            <div>
                <h3 style='margin:0; color:#4b4b4b; font-size: 1.1em;'>Net Net Gain (Total)</h3>
                <h2 style='margin:0; color:#1f1f1f;'>{round(net_net_at_loan, 0):,} €</h2>
            </div>
        </div>
        <p style='margin-top:15px; color:#4b4b4b; font-style: italic;'>
            <b>Net Net</b> includes rental income ({occupancy_rate}% occupancy) and expenses over {loan_years} years (actualized at loan end).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Monthly Payment", f"{fmt(result.monthly_payment)} €")
    col_b.metric("Total Interest", f"{fmt(result.total_int_nom)} €")
    col_c.metric("Total Nominal Cost", f"{fmt(loan_amount + result.total_int_nom)} €")
    col_d.metric("Property Value (Nominal)", f"{fmt(final_nom_at_loan)} €")
    col_e.metric("Inflation Gain", f"{fmt((loan_amount + result.total_int_nom) - total_cost_act)} €")

    st.markdown("---")
    st.write("💡 **Actualized View at Loan End (Today's Purchasing Power):**")
    col_f, col_g, col_h, col_i, col_j = st.columns(5)
    col_f.metric("Real Credit Cost", f"{fmt(total_cost_act)} €")
    col_g.metric("Real Property Value", f"{fmt(final_act_at_loan)} €")
    col_h.metric("Net Balance (Equity)", f"{fmt(net_balance_at_loan)} €")
    col_i.metric("Net Net Gain (Total)", f"{fmt(net_net_at_loan)} €")

    df_crossover = df[df["Cumul. Repayment (Actualized)"] > df["Actualized Property Value"]]
    crossover_year = df_crossover["Year"].iloc[0] if not df_crossover.empty else "Never"
    col_j.metric("Crossover Year", f"Year {crossover_year}" if isinstance(crossover_year, int) else crossover_year)

    # SECTION 2: VISUALIZATIONS
    st.divider()

    tab_inflection, tab_nominal_vs_real, tab_global, tab_annual, tab_rent, tab_waterfall, tab_stock, tab_sensitivity, tab_data = st.tabs([
        "🎯 Inflection Point",
        "⚖️ Nominal vs Real",
        "🌎 Global View",
        "📅 Annual Evolution",
        "💰 Rent & Expenses",
        "🏆 Net Net Breakdown",
        "📈 Stock vs Real Estate",
        "🔍 Sensitivity Analysis",
        "📑 Data Table",
    ])

    # NEW TAB: INFLECTION POINT ANALYSIS
    with tab_inflection:
        st.subheader("🎯 The Inflection Point: When Real Value Starts Declining")

        st.markdown("""
        <div class="insight-box">
        <b>Key Insight:</b> During your loan, you benefit from <b>leverage</b> — you're repaying debt with money that's worth less each year.
        Once the loan ends, this advantage disappears. If real estate appreciation is lower than inflation, your property's 
        <b>real purchasing power decreases every year</b>.
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        col_peak1, col_peak2, col_peak3, col_peak4 = st.columns(4)

        col_peak1.metric(
            "📍 Peak Real Value Year",
            f"Year {exit_analysis['peak_act_year']}",
            help="Year when actualized property value reaches maximum"
        )
        col_peak2.metric(
            "💰 Peak Real Value",
            f"{fmt(exit_analysis['peak_act_value'])} €",
            help="Maximum actualized property value"
        )
        col_peak3.metric(
            "🏆 Peak Net Net Year",
            f"Year {exit_analysis['peak_net_net_year']}",
            help="Year when total actualized gain (including rent) is maximum"
        )
        col_peak4.metric(
            "📉 First Decline Year",
            f"Year {exit_analysis['first_decline_year']}" if exit_analysis['first_decline_year'] else "Never",
            help="First year when actualized value drops YoY"
        )

        st.divider()

        # Main inflection chart
        st.subheader("Property Value: Nominal vs Actualized Over Time")

        fig_inflection = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("Property Value Evolution", "Year-over-Year Change (Actualized)")
        )

        # Top chart: Values
        fig_inflection.add_trace(
            go.Scatter(x=df["Year"], y=df["Nominal Property Value"], mode="lines",
                       name="Nominal Value", line=dict(color="#636EFA", width=3)),
            row=1, col=1
        )
        fig_inflection.add_trace(
            go.Scatter(x=df["Year"], y=df["Actualized Property Value"], mode="lines",
                       name="Actualized Value", line=dict(color="#AB63FA", width=4)),
            row=1, col=1
        )

        # Add peak marker
        fig_inflection.add_trace(
            go.Scatter(
                x=[exit_analysis['peak_act_year']],
                y=[exit_analysis['peak_act_value']],
                mode="markers+text",
                name="Peak Value",
                marker=dict(size=15, color="#FF6B6B", symbol="star"),
                text=["PEAK"],
                textposition="top center",
                textfont=dict(size=12, color="#FF6B6B")
            ),
            row=1, col=1
        )

        # Loan end line
        fig_inflection.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
        fig_inflection.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)

        # Bottom chart: YoY changes
        colors_yoy = ["#00CC96" if v >= 0 else "#EF553B" for v in df["YoY Property Change (Act.)"]]
        fig_inflection.add_trace(
            go.Bar(x=df["Year"], y=df["YoY Property Change (Act.)"],
                   name="YoY Change (Act.)", marker_color=colors_yoy, showlegend=False),
            row=2, col=1
        )
        fig_inflection.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)

        fig_inflection.update_layout(
            height=650,
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )
        fig_inflection.update_yaxes(title_text="Value (€)", row=1, col=1)
        fig_inflection.update_yaxes(title_text="YoY Change (€)", row=2, col=1)
        fig_inflection.update_xaxes(title_text="Year", row=2, col=1)

        # Add annotations
        fig_inflection.add_annotation(
            x=loan_years, y=df[df["Year"] == loan_years]["Nominal Property Value"].values[0],
            text="Loan Ends", showarrow=True, arrowhead=2, ax=40, ay=-40,
            font=dict(color="red", size=11), row=1, col=1
        )

        st.plotly_chart(fig_inflection, use_container_width=True)

        # Explanation
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("### 📈 Why Nominal Keeps Rising")
            st.markdown(f"""
            The **nominal value** (blue line) keeps growing at **{real_estate_inflation}%/year** forever.
            This is what you'd see in real estate listings — the "sticker price."

            - Year 1: {fmt(property_value * (1 + real_estate_inflation/100))} €
            - Year {loan_years}: {fmt(final_nom_at_loan)} €
            - Year {projection_years}: {fmt(final_nom)} €

            **But this is an illusion** — it doesn't account for what money is worth.
            """)

        with col_exp2:
            st.markdown("### 📉 Why Actualized Peaks Then Falls")
            st.markdown(f"""
            The **actualized value** (purple line) shows **real purchasing power**.

            **During the loan:** Your debt is eroded by inflation, giving you an edge.

            **After the loan:** No more leverage benefit. Since RE appreciation ({real_estate_inflation}%)
            < inflation ({inflation_rate}%), **real value declines by ~{round(inflation_rate - real_estate_inflation, 1)}%/year**.

            - Peak at Year {exit_analysis['peak_act_year']}: {fmt(exit_analysis['peak_act_value'])} €
            - Year {projection_years}: {fmt(final_act)} € (**-{fmt(exit_analysis['peak_act_value'] - final_act)} €**)
            """)

        st.divider()

        # Strategic recommendations
        st.subheader("🎯 Strategic Recommendations")

        effective_real_return = real_estate_inflation - inflation_rate

        if effective_real_return < 0:
            st.error(f"""
            ⚠️ **Alert: Negative Real Return ({round(effective_real_return, 1)}%/year)**

            With RE appreciation ({real_estate_inflation}%) below inflation ({inflation_rate}%), 
            your property loses **{fmt(abs(effective_real_return))}%** of real value each year after the loan.

            **Optimal Strategy:**
            1. **Sell near Year {exit_analysis['peak_act_year']}** to capture maximum real value
            2. **Or refinance** to restart the leverage clock
            3. **Or increase rent** above inflation to compensate

            Holding past Year {exit_analysis['peak_act_year']} means losing ~{fmt(abs(effective_real_return) * final_act_at_loan / 100)} € of purchasing power per year.
            """)
        elif effective_real_return == 0:
            st.warning(f"""
            ⚡ **Neutral Real Return**

            RE appreciation matches inflation — your property maintains purchasing power but doesn't grow.
            The loan's leverage was your only real advantage.

            **Consider selling at loan end** or refinancing to continue benefiting from leverage.
            """)
        else:
            st.success(f"""
            ✅ **Positive Real Return (+{round(effective_real_return, 1)}%/year)**

            Great news! RE appreciation exceeds inflation. Your property gains real value over time.
            The loan accelerated your gains, but holding long-term is also profitable.
            """)

        # Comparison table
        st.subheader("📊 Value Comparison at Key Milestones")

        milestones = [loan_years]
        if projection_years > loan_years:
            milestones.extend([min(loan_years + 5, projection_years), min(loan_years + 10, projection_years)])
            if projection_years not in milestones:
                milestones.append(projection_years)

        milestone_data = []
        for year in sorted(set(milestones)):
            row = df[df["Year"] == year].iloc[0]
            milestone_data.append({
                "Year": year,
                "Status": "Loan Ends" if year == loan_years else f"+{year - loan_years} years",
                "Nominal Value": f"{fmt(row['Nominal Property Value'])} €",
                "Actualized Value": f"{fmt(row['Actualized Property Value'])} €",
                "Real vs Peak": f"{fmt(row['Actualized Property Value'] - exit_analysis['peak_act_value'])} €",
                "Net Net (Act.)": f"{fmt(row['Net Net Real Estate (Act.)'])} €",
            })

        st.dataframe(pd.DataFrame(milestone_data), use_container_width=True, hide_index=True)

    # NEW TAB: NOMINAL VS REAL DETAILED COMPARISON
    with tab_nominal_vs_real:
        st.subheader("⚖️ The Illusion of Nominal Gains")

        st.markdown("""
        <div class="insight-box">
        <b>The Trap:</b> Looking only at nominal values makes you think you're always getting richer.
        But <b>€100,000 in 20 years won't buy what €100,000 buys today</b>. This section reveals the gap.
        </div>
        """, unsafe_allow_html=True)

        # Gap analysis
        col_gap1, col_gap2, col_gap3 = st.columns(3)

        gap_at_loan_end = final_nom_at_loan - final_act_at_loan
        gap_at_end = final_nom - final_act
        illusion_pct = (gap_at_loan_end / final_nom_at_loan) * 100

        col_gap1.metric(
            "Nominal-Real Gap (Loan End)",
            f"{fmt(gap_at_loan_end)} €",
            delta=f"{round(illusion_pct, 1)}% is inflation illusion",
            delta_color="inverse"
        )
        col_gap2.metric(
            f"Nominal-Real Gap (Year {projection_years})",
            f"{fmt(gap_at_end)} €",
            delta=f"{round((gap_at_end / final_nom) * 100, 1)}% is inflation illusion",
            delta_color="inverse"
        )
        col_gap3.metric(
            "Discount Factor",
            f"{round(df['Discount Factor'].iloc[-1] * 100, 1)}%",
            help=f"€1 in Year {projection_years} = €{round(df['Discount Factor'].iloc[-1], 3)} today"
        )

        st.divider()

        # Side by side gains
        st.subheader("Net Gains: Nominal vs Actualized")

        fig_gains = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Nominal Gains (What You See)", "Actualized Gains (What You Get)"),
            horizontal_spacing=0.1
        )

        # Nominal
        fig_gains.add_trace(
            go.Scatter(x=df["Year"], y=df["Net Gain (Nominal)"], mode="lines",
                       name="Net Gain (Nom.)", line=dict(color="#2ecc71", width=3)),
            row=1, col=1
        )
        fig_gains.add_trace(
            go.Scatter(x=df["Year"], y=df["Net Net Real Estate (Nom.)"], mode="lines",
                       name="Net Net (Nom.)", line=dict(color="#27ae60", width=3, dash="dash")),
            row=1, col=1
        )

        # Actualized
        fig_gains.add_trace(
            go.Scatter(x=df["Year"], y=df["Net Gain (Actualized)"], mode="lines",
                       name="Net Gain (Act.)", line=dict(color="#9b59b6", width=3)),
            row=1, col=2
        )
        fig_gains.add_trace(
            go.Scatter(x=df["Year"], y=df["Net Net Real Estate (Act.)"], mode="lines",
                       name="Net Net (Act.)", line=dict(color="#8e44ad", width=3, dash="dash")),
            row=1, col=2
        )

        # Add loan end markers
        fig_gains.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig_gains.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
        fig_gains.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=1, col=1)
        fig_gains.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=1, col=2)

        fig_gains.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified"
        )
        fig_gains.update_yaxes(title_text="Gain (€)", row=1, col=1)
        fig_gains.update_yaxes(title_text="Gain (€)", row=1, col=2)

        st.plotly_chart(fig_gains, use_container_width=True)

        st.markdown("""
        **Left chart (Nominal):** Both lines keep rising — looks great!

        **Right chart (Actualized):** The truth. Notice how gains **plateau or decline after the loan ends**.
        This is your real wealth, adjusted for what money can actually buy.
        """)

        st.divider()

        # Annual return comparison
        st.subheader("Annual Return Rates: The Reality Check")

        fig_returns = go.Figure()

        fig_returns.add_trace(
            go.Scatter(x=df["Year"], y=df["Nominal Annual Return (%)"], mode="lines+markers",
                       name="Nominal Return", line=dict(color="#3498db", width=2),
                       marker=dict(size=6))
        )
        fig_returns.add_trace(
            go.Scatter(x=df["Year"], y=df["Actualized Annual Return (%)"], mode="lines+markers",
                       name="Actualized Return", line=dict(color="#e74c3c", width=2),
                       marker=dict(size=6))
        )

        fig_returns.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_returns.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7)
        fig_returns.add_annotation(x=loan_years, y=max(df["Nominal Annual Return (%)"]),
                                   text="Loan Ends", showarrow=False, yshift=10, font=dict(color="red"))

        fig_returns.update_layout(
            title="Year-over-Year Property Value Return",
            xaxis_title="Year",
            yaxis_title="Annual Return (%)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )

        st.plotly_chart(fig_returns, use_container_width=True)

        col_ret1, col_ret2 = st.columns(2)
        with col_ret1:
            st.metric(
                "Constant Nominal Return",
                f"+{real_estate_inflation}%/year",
                help="RE appreciation rate — always positive in our model"
            )
        with col_ret2:
            st.metric(
                "Constant Real Return",
                f"{'+' if effective_real_return >= 0 else ''}{round(effective_real_return, 2)}%/year",
                delta="Gain" if effective_real_return >= 0 else "Loss",
                delta_color="normal" if effective_real_return >= 0 else "inverse",
                help="RE appreciation minus inflation"
            )

        st.info(f"""
        **The math is simple:**
        - Nominal return: +{real_estate_inflation}%/year (RE appreciation)
        - Real return: {real_estate_inflation}% - {inflation_rate}% = **{'+' if effective_real_return >= 0 else ''}{round(effective_real_return, 2)}%/year**

        If this is negative, you're losing purchasing power every year you hold after the loan.
        """)

    with tab_global:
        col_pie1, col_pie2 = st.columns(2)
        col_pie1.plotly_chart(
            create_donut(["Principal", "Interest"], [loan_amount, result.total_int_nom], "Nominal Breakdown"),
            use_container_width=True,
        )
        col_pie2.plotly_chart(
            create_donut(["Principal (Act.)", "Interest (Act.)"], [result.total_cap_act, result.total_int_act], "Actualized Breakdown"),
            use_container_width=True,
        )

    with tab_annual:
        st.subheader("Property Value vs Repayments Over Time")
        fig_immo = go.Figure()

        fig_immo.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7)
        fig_immo.add_annotation(x=loan_years, y=df["Nominal Property Value"].max(), text="Loan Ends", showarrow=False, yshift=10, font=dict(color="red"))

        fig_immo.add_trace(go.Scatter(x=df["Year"], y=df["Nominal Property Value"], mode="lines", name="Market Price (Nominal)", line=dict(color="#636EFA", width=3)))
        fig_immo.add_trace(go.Scatter(x=df["Year"], y=df["Actualized Property Value"], mode="lines", name="Real Value (Actualized)", line=dict(color="#AB63FA", width=3, dash="dash")))
        fig_immo.add_trace(go.Scatter(x=df["Year"], y=df["Cumul. Repayment (Actualized)"], mode="lines", name="Cumul. Repayment (Act.)", line=dict(color="#FFA15A", width=2)))
        fig_immo.update_layout(xaxis_title="Year", yaxis_title="Amount (€)", height=500, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_immo, use_container_width=True)

        st.info("""
        **Chart Legend:**
        - **Market Price (Blue)**: Listed price, inflated by real estate appreciation.
        - **Real Value (Purple dashed)**: True value in today's purchasing power.
        - **Cumul. Repayment (Orange)**: Total paid out of pocket (actualized monthly payments).

        ⚠️ Notice how the purple line **peaks at loan end** then declines — this is the leverage effect disappearing!
        """)

    with tab_rent:
        st.subheader("🏡 Cumulative Cash Flows (Actualized)")
        fig_rent = go.Figure()
        fig_rent.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7)
        fig_rent.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Rent (Actualized)"], name="Cumul. Rent (Act.)", marker_color="#00CC96"))
        fig_rent.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Charges (Actualized)"], name="Cumul. Charges (Act.)", marker_color="#EF553B"))
        fig_rent.update_layout(barmode="group", xaxis_title="Year", yaxis_title="Actualized Amount (€)", height=450, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rent, use_container_width=True)

    with tab_waterfall:
        st.subheader("🏆 Net Net Breakdown (Actualized vs Nominal) at Loan End")
        col_w1, col_w2 = st.columns(2)
        labels = ["Property Value", "Rental Income", "Credit Cost", "Taxes/Charges", "Net Net Balance"]

        values_act = [final_act_at_loan, result.total_rent_act, -total_cost_act, -result.total_charges_act, 0]
        fig_w_act = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=labels,
            y=values_act,
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#00CC96"}},
            decreasing={"marker": {"color": "#EF553B"}},
            totals={"marker": {"color": "#636EFA"}},
        ))
        fig_w_act.update_layout(title="Real Gain (Actualized)", height=450, paper_bgcolor="rgba(0,0,0,0)")
        col_w1.plotly_chart(fig_w_act, use_container_width=True)

        values_nom = [final_nom_at_loan, result.total_rent_nom, -(loan_amount + result.total_int_nom), -result.total_charges_nom, 0]
        fig_w_nom = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=labels,
            y=values_nom,
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#34495e"}},
        ))
        fig_w_nom.update_layout(title="Gross Gain (Nominal)", height=450, paper_bgcolor="rgba(0,0,0,0)")
        col_w2.plotly_chart(fig_w_nom, use_container_width=True)

    with tab_stock:
        st.subheader("💰 Real Estate vs Stock Market")
        st.write(f"Alternative scenario: You invest the down payment (**{fmt(down_payment)} €**) and invest the monthly payment (**{fmt(result.monthly_payment)} €**) in stocks at **{stock_return}%/year**.")

        col_comp1, col_comp2 = st.columns(2)

        fig_c_act = go.Figure()
        fig_c_act.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7)
        fig_c_act.add_trace(go.Scatter(x=df["Year"], y=df["Net Net Real Estate (Act.)"], mode="lines", name="Real Estate Gain (Act.)", line=dict(color="#00CC96", width=4)))
        fig_c_act.add_trace(go.Scatter(x=df["Year"], y=df["Stock Gain (Act.)"], mode="lines", name="Stock Gain (Act.)", line=dict(color="#17becf", width=4)))
        fig_c_act.update_layout(title="Real Gain Comparison (Actualized)", xaxis_title="Year", height=450, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)")
        col_comp1.plotly_chart(fig_c_act, use_container_width=True)

        fig_c_nom = go.Figure()
        fig_c_nom.add_vline(x=loan_years, line_dash="dash", line_color="red", opacity=0.7)
        fig_c_nom.add_trace(go.Scatter(x=df["Year"], y=df["Net Net Real Estate (Nom.)"], mode="lines", name="Real Estate Gain (Nom.)", line=dict(color="#00CC96", width=4)))
        fig_c_nom.add_trace(go.Scatter(x=df["Year"], y=df["Stock Gain (Nom.)"], mode="lines", name="Stock Gain (Nom.)", line=dict(color="#17becf", width=4)))
        fig_c_nom.update_layout(title="Gross Gain Comparison (Nominal)", xaxis_title="Year", height=450, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)")
        col_comp2.plotly_chart(fig_c_nom, use_container_width=True)

        st.info("Stock gain = final portfolio value minus total invested (down payment + monthly payments). After loan ends, no new stock investments are made.")

    with tab_sensitivity:
        st.subheader("Inflection & Profitability Analysis")
        st.info("""
        **Understanding thresholds:**
        - **Net Inflection (dotted)**: Equity focus (Property Value - Debt).
        - **Net Net Inflection (solid)**: Total view (Equity + Cash Flows).
        """)
        col_s1, col_s2, col_s3 = st.columns(3)

        rate_range, net_rate, net_net_rate = compute_sensitivity_rate(
            property_value, loan_amount, int(loan_years), inflation_rate,
            real_estate_inflation, initial_rent, property_tax,
            occupancy_rate, stock_return, down_payment
        )
        col_s1.plotly_chart(create_sensitivity_chart(rate_range, net_rate, net_net_rate, "Sensitivity: Interest Rate", "#17becf"), use_container_width=True)
        col_s1.metric("Net Inflection", f"{round(rate_range[np.argmin(np.abs(net_rate))], 2)}%")
        col_s1.metric("Net Net Inflection", f"{round(rate_range[np.argmin(np.abs(net_net_rate))], 2)}%")
        col_s1.write("**Insight:** The lower your rate compared to Net Net inflection, the stronger your leverage effect.")

        market_range, net_market, net_net_market = compute_sensitivity_market(
            property_value, int(loan_years), inflation_rate,
            total_cost_act, result.total_rent_act, result.total_charges_act
        )
        col_s2.plotly_chart(create_sensitivity_chart(market_range, net_market, net_net_market, "Sensitivity: RE Market", "#ff7f0e"), use_container_width=True)
        col_s2.metric("Net Inflection", f"{round(market_range[np.argmin(np.abs(net_market))], 2)}%")
        col_s2.metric("Net Net Inflection", f"{round(market_range[np.argmin(np.abs(net_net_market))], 2)}%")
        col_s2.write("**Insight:** If Net Net inflection is negative, the property makes you richer even if the market drops slightly.")

        inf_range, net_inf, net_net_inf = compute_sensitivity_inflation(
            property_value, loan_amount, annual_rate, int(loan_years),
            real_estate_inflation, initial_rent, property_tax,
            occupancy_rate, stock_return, down_payment
        )
        col_s3.plotly_chart(create_sensitivity_chart(inf_range, net_inf, net_net_inf, "Sensitivity: Inflation", "#e377c2"), use_container_width=True)
        col_s3.metric("Net Inflection", f"{round(inf_range[np.argmin(np.abs(net_inf))], 2)}%")
        col_s3.metric("Net Net Inflection", f"{round(inf_range[np.argmin(np.abs(net_net_inf))], 2)}%")
        col_s3.write("**Insight:** Inflation is the borrower's ally — it erodes the real value of debt over time.")

    with tab_data:
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ Your down payment covers the entire property. No financing needed.")

st.caption("Note: Actualization converts all future amounts to today's purchasing power (constant euros).")