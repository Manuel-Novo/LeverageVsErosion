import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Leverage vs Inflation Erosion",
    page_icon="🏦",
    layout="wide"
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #1f1f1f !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #4b4b4b !important; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Custom boxes */
    .highlight-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
    }
    .box-positive { background-color: #d4edda; border-left: 5px solid #28a745; }
    .box-negative { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .box-warning { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .box-info { background-color: #d1ecf1; border-left: 5px solid #17a2b8; }
    
    /* Key insight boxes */
    .key-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .key-insight h2 { color: white; margin: 0; font-size: 2em; }
    .key-insight p { color: rgba(255,255,255,0.9); margin-top: 10px; }
    
    /* Bank loss box */
    .bank-loss-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .bank-loss-box h2 { color: white; margin: 0; }
    
    /* Erosion warning box */
    .erosion-warning {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .erosion-warning h2 { color: white; margin: 0; }
    
    /* Section headers */
    .section-header {
        background-color: #2c3e50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class LoanMetrics:
    """Core loan calculation results."""
    df: pd.DataFrame
    monthly_payment: float
    # Nominal values (what you see on paper)
    total_interest_nominal: float
    total_cost_nominal: float  # principal + interest
    # Actualized values (real purchasing power)
    total_interest_actualized: float
    total_principal_actualized: float
    total_cost_actualized: float  # principal + interest actualized
    # Cash flows
    total_rent_nominal: float
    total_rent_actualized: float
    total_charges_nominal: float
    total_charges_actualized: float


@dataclass
class ExitAnalysis:
    """Optimal exit point analysis."""
    peak_value_year: int
    peak_value_amount: float
    peak_net_net_year: int
    peak_net_net_amount: float
    first_decline_year: Optional[int]
    annual_erosion_rate: float  # % lost per year after loan


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_currency(value: float) -> str:
    """Format number as currency with space separator."""
    return f"{round(value, 0):,.0f}".replace(",", " ")


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage."""
    return f"{round(value, decimals):+.{decimals}f}%"


def get_color_for_value(value: float, threshold: float = 0) -> str:
    """Return green for positive, red for negative values."""
    return "#28a745" if value >= threshold else "#dc3545"


# =============================================================================
# CORE CALCULATIONS
# =============================================================================
@st.cache_data
def calculate_loan_amortization(
    property_value: float,
    loan_amount: float,
    annual_rate: float,
    loan_years: int,
    projection_years: int,
    inflation_rate: float,
    real_estate_growth: float,
    monthly_rent: float,
    annual_property_tax: float,
    occupancy_rate: float,
    stock_return: float,
    down_payment: float,
    rent_if_not_buying: float,
    monthly_stock_investment: float,
) -> LoanMetrics:
    """
    Calculate complete loan amortization with inflation-adjusted values.
    
    Key insight: We track both nominal (paper) values and actualized (real purchasing power) values.
    The difference reveals how inflation erodes debt and benefits borrowers.
    """
    # Monthly rate and payment calculation
    monthly_rate = (annual_rate / 100) / 12
    total_months = loan_years * 12
    
    if annual_rate > 0:
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** total_months
        ) / ((1 + monthly_rate) ** total_months - 1)
    else:
        monthly_payment = loan_amount / total_months
    
    # Growth factors (pre-computed for efficiency)
    inflation_factor = 1 + inflation_rate / 100
    real_estate_factor = 1 + real_estate_growth / 100
    stock_monthly_factor = 1 + (stock_return / 100) / 12
    occupancy_factor = occupancy_rate / 100
    
    # Running totals
    remaining_principal = loan_amount
    
    # Nominal cumulative values
    cumul_interest_nom = 0.0
    cumul_principal_nom = 0.0
    cumul_rent_nom = 0.0
    cumul_charges_nom = 0.0
    
    # Actualized cumulative values
    cumul_interest_act = 0.0
    cumul_principal_act = 0.0
    cumul_rent_act = 0.0
    cumul_charges_act = 0.0
    
    # Stock portfolio simulation (alternative scenario)
    stock_value = down_payment  # Start with down payment
    total_stock_invested = down_payment
    cumul_rent_paid_if_not_buying_nom = 0.0  # Rent you'd pay if renting instead
    cumul_rent_paid_if_not_buying_act = 0.0
    
    yearly_data = []
    
    for year in range(1, projection_years + 1):
        is_loan_active = year <= loan_years
        year_interest = 0.0
        year_principal = 0.0
        
        # Inflation-adjusted amounts for this year
        # (year-1 because year 1 uses base amounts)
        inflation_multiplier = inflation_factor ** (year - 1)
        current_monthly_investment = monthly_stock_investment * inflation_multiplier
        current_monthly_rent_if_not_buying = rent_if_not_buying * inflation_multiplier
        
        # Monthly calculations within year
        for month in range(12):
            if is_loan_active and remaining_principal > 0:
                interest_payment = remaining_principal * monthly_rate
                principal_payment = min(monthly_payment - interest_payment, remaining_principal)
                
                year_interest += interest_payment
                year_principal += principal_payment
                remaining_principal -= principal_payment
            
            # Stock scenario: invest monthly amount (grows with inflation)
            stock_value = stock_value * stock_monthly_factor + current_monthly_investment
            total_stock_invested += current_monthly_investment
            
            # Track rent you'd pay if not buying (grows with inflation)
            cumul_rent_paid_if_not_buying_nom += current_monthly_rent_if_not_buying
        
        # Discount factor: converts future money to today's purchasing power
        discount_factor = 1 / (inflation_factor ** year)
        
        # Actualize rent paid if not buying
        annual_rent_paid_if_not_buying = current_monthly_rent_if_not_buying * 12
        cumul_rent_paid_if_not_buying_act += annual_rent_paid_if_not_buying * discount_factor
        
        # Property values
        property_nominal = property_value * (real_estate_factor ** year)
        property_actualized = property_nominal * discount_factor
        
        # Rent and charges (grow with inflation, adjusted for occupancy)
        annual_rent_nom = (monthly_rent * 12) * (inflation_factor ** (year - 1)) * occupancy_factor
        annual_charges_nom = annual_property_tax * (inflation_factor ** (year - 1))
        
        # Update cumulative values
        cumul_interest_nom += year_interest
        cumul_principal_nom += year_principal
        cumul_interest_act += year_interest * discount_factor
        cumul_principal_act += year_principal * discount_factor
        cumul_rent_nom += annual_rent_nom
        cumul_rent_act += annual_rent_nom * discount_factor
        cumul_charges_nom += annual_charges_nom
        cumul_charges_act += annual_charges_nom * discount_factor
        
        # Total costs
        total_paid_nom = cumul_interest_nom + cumul_principal_nom
        total_paid_act = cumul_interest_act + cumul_principal_act
        
        # Net calculations
        # Net = Property Value - Total Loan Cost
        net_nominal = property_nominal - (loan_amount + cumul_interest_nom)
        net_actualized = property_actualized - total_paid_act
        
        # Net Net = Net + Rent - Charges (full picture)
        net_net_nominal = net_nominal + cumul_rent_nom - cumul_charges_nom
        net_net_actualized = net_actualized + cumul_rent_act - cumul_charges_act
        
        # Stock comparison (subtract rent you'd pay if not buying)
        stock_actualized = stock_value * discount_factor
        stock_gain_nom = stock_value - total_stock_invested - cumul_rent_paid_if_not_buying_nom
        stock_gain_act = stock_actualized - (total_stock_invested * discount_factor) - cumul_rent_paid_if_not_buying_act
        
        yearly_data.append({
            "Year": year,
            "Loan Active": "✓" if is_loan_active else "✗",
            "Remaining Principal": round(max(0, remaining_principal), 0),
            # Property
            "Property (Nominal)": round(property_nominal, 0),
            "Property (Actualized)": round(property_actualized, 0),
            # Loan costs
            "Cumul. Interest (Nominal)": round(cumul_interest_nom, 0),
            "Cumul. Interest (Actualized)": round(cumul_interest_act, 0),
            "Total Paid (Nominal)": round(total_paid_nom, 0),
            "Total Paid (Actualized)": round(total_paid_act, 0),
            # Cash flows (rental income if you rent out the property)
            "Cumul. Rent (Nominal)": round(cumul_rent_nom, 0),
            "Cumul. Rent (Actualized)": round(cumul_rent_act, 0),
            "Cumul. Charges (Nominal)": round(cumul_charges_nom, 0),
            "Cumul. Charges (Actualized)": round(cumul_charges_act, 0),
            # Net results
            "Net Gain (Nominal)": round(net_nominal, 0),
            "Net Gain (Actualized)": round(net_actualized, 0),
            "Net Net (Nominal)": round(net_net_nominal, 0),
            "Net Net (Actualized)": round(net_net_actualized, 0),
            # Stock comparison
            "Stock Value (Nominal)": round(stock_value, 0),
            "Stock Value (Actualized)": round(stock_actualized, 0),
            "Stock Invested (Nominal)": round(total_stock_invested, 0),
            "Rent Paid if Not Buying (Nominal)": round(cumul_rent_paid_if_not_buying_nom, 0),
            "Rent Paid if Not Buying (Actualized)": round(cumul_rent_paid_if_not_buying_act, 0),
            "Stock Gain (Nominal)": round(stock_gain_nom, 0),
            "Stock Gain (Actualized)": round(stock_gain_act, 0),
            # Analysis helpers
            "Discount Factor": round(discount_factor, 4),
            "YoY Property Change (Act.)": 0,  # Filled below
        })
    
    df = pd.DataFrame(yearly_data)
    
    # Calculate year-over-year changes
    df["YoY Property Change (Act.)"] = df["Property (Actualized)"].diff().fillna(0)
    df["YoY Net Net Change (Act.)"] = df["Net Net (Actualized)"].diff().fillna(0)
    
    # Calculate the values at loan end for the return object
    loan_end_data = df[df["Year"] == loan_years].iloc[0]
    
    return LoanMetrics(
        df=df,
        monthly_payment=monthly_payment,
        total_interest_nominal=loan_end_data["Cumul. Interest (Nominal)"],
        total_cost_nominal=loan_amount + loan_end_data["Cumul. Interest (Nominal)"],
        total_interest_actualized=loan_end_data["Cumul. Interest (Actualized)"],
        total_principal_actualized=cumul_principal_act if year <= loan_years else df[df["Year"] == loan_years]["Total Paid (Actualized)"].values[0] - df[df["Year"] == loan_years]["Cumul. Interest (Actualized)"].values[0],
        total_cost_actualized=loan_end_data["Total Paid (Actualized)"],
        total_rent_nominal=loan_end_data["Cumul. Rent (Nominal)"],
        total_rent_actualized=loan_end_data["Cumul. Rent (Actualized)"],
        total_charges_nominal=loan_end_data["Cumul. Charges (Nominal)"],
        total_charges_actualized=loan_end_data["Cumul. Charges (Actualized)"],
    )


@st.cache_data
def analyze_exit_strategy(df: pd.DataFrame, loan_years: int, inflation_rate: float, real_estate_growth: float) -> ExitAnalysis:
    """Determine optimal exit points and erosion metrics."""
    
    # Peak actualized property value
    peak_idx = df["Property (Actualized)"].idxmax()
    peak_year = int(df.loc[peak_idx, "Year"])
    peak_value = df.loc[peak_idx, "Property (Actualized)"]
    
    # Peak net net
    peak_net_net_idx = df["Net Net (Actualized)"].idxmax()
    peak_net_net_year = int(df.loc[peak_net_net_idx, "Year"])
    peak_net_net_value = df.loc[peak_net_net_idx, "Net Net (Actualized)"]
    
    # First decline year (after loan)
    post_loan = df[df["Year"] > loan_years]
    declining = post_loan[post_loan["YoY Property Change (Act.)"] < 0]
    first_decline = int(declining["Year"].iloc[0]) if not declining.empty else None
    
    # Annual erosion rate after loan
    annual_erosion = inflation_rate - real_estate_growth
    
    return ExitAnalysis(
        peak_value_year=peak_year,
        peak_value_amount=peak_value,
        peak_net_net_year=peak_net_net_year,
        peak_net_net_amount=peak_net_net_value,
        first_decline_year=first_decline,
        annual_erosion_rate=annual_erosion,
    )


@st.cache_data
def compute_rate_sensitivity(
    property_value: float, loan_amount: float, loan_years: int,
    inflation_rate: float, real_estate_growth: float, monthly_rent: float,
    property_tax: float, occupancy_rate: float, stock_return: float, down_payment: float,
    rent_if_not_buying: float, monthly_stock_investment: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sensitivity analysis for interest rate variations."""
    rates = np.linspace(0, 15, 100)
    net_values, net_net_values = [], []
    
    final_property_act = property_value * ((1 + real_estate_growth / 100) ** loan_years) / ((1 + inflation_rate / 100) ** loan_years)
    
    for rate in rates:
        result = calculate_loan_amortization(
            property_value, loan_amount, rate, loan_years, loan_years,
            inflation_rate, real_estate_growth, monthly_rent, property_tax,
            occupancy_rate, stock_return, down_payment,
            rent_if_not_buying, monthly_stock_investment
        )
        net_values.append(final_property_act - result.total_cost_actualized)
        net_net_values.append(
            (final_property_act + result.total_rent_actualized) - 
            (result.total_cost_actualized + result.total_charges_actualized)
        )
    
    return rates, np.array(net_values), np.array(net_net_values)


@st.cache_data
def compute_market_sensitivity(
    property_value: float, loan_years: int, inflation_rate: float,
    total_cost_act: float, total_rent_act: float, total_charges_act: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sensitivity analysis for real estate market variations."""
    market_rates = np.linspace(-5, 15, 100)
    denominator = (1 + inflation_rate / 100) ** loan_years
    
    final_values = property_value * ((1 + market_rates / 100) ** loan_years) / denominator
    net_values = final_values - total_cost_act
    net_net_values = (final_values + total_rent_act) - (total_cost_act + total_charges_act)
    
    return market_rates, net_values, net_net_values


@st.cache_data
def compute_inflation_sensitivity(
    property_value: float, loan_amount: float, annual_rate: float, loan_years: int,
    real_estate_growth: float, monthly_rent: float, property_tax: float,
    occupancy_rate: float, stock_return: float, down_payment: float,
    rent_if_not_buying: float, monthly_stock_investment: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sensitivity analysis for inflation rate variations."""
    inflation_rates = np.linspace(0, 10, 100)
    net_values, net_net_values = [], []
    
    property_growth = (1 + real_estate_growth / 100) ** loan_years
    
    for inf_rate in inflation_rates:
        result = calculate_loan_amortization(
            property_value, loan_amount, annual_rate, loan_years, loan_years,
            inf_rate, real_estate_growth, monthly_rent, property_tax,
            occupancy_rate, stock_return, down_payment,
            rent_if_not_buying, monthly_stock_investment
        )
        final_property_act = property_value * property_growth / ((1 + inf_rate / 100) ** loan_years)
        net_values.append(final_property_act - result.total_cost_actualized)
        net_net_values.append(
            (final_property_act + result.total_rent_actualized) - 
            (result.total_cost_actualized + result.total_charges_actualized)
        )
    
    return inflation_rates, np.array(net_values), np.array(net_net_values)


# =============================================================================
# CHART BUILDERS
# =============================================================================
def create_donut_chart(labels: list, values: list, title: str, colors: list = None) -> go.Figure:
    """Create a donut chart for breakdown visualization."""
    if colors is None:
        colors = ["#28a745", "#dc3545"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside"
    )])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        margin=dict(t=60, b=20, l=20, r=20),
        height=350,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_sensitivity_chart(x: np.ndarray, y_net: np.ndarray, y_net_net: np.ndarray, 
                             title: str, x_label: str, color: str) -> go.Figure:
    """Create sensitivity analysis chart with inflection points."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y_net,
        name="Net (Equity only)",
        line=dict(color=color, dash="dot", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_net_net,
        name="Net Net (Total)",
        line=dict(color=color, width=4)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # Mark inflection points
    net_inflection_idx = np.argmin(np.abs(y_net))
    net_net_inflection_idx = np.argmin(np.abs(y_net_net))
    
    fig.add_vline(x=x[net_inflection_idx], line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=x[net_net_inflection_idx], line_dash="dot", line_color=color, opacity=0.7)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_label,
        yaxis_title="Gain (€)",
        height=400,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    return fig


def create_main_evolution_chart(df: pd.DataFrame, loan_years: int) -> go.Figure:
    """Create the main property value evolution chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "📈 Property Value: Nominal vs Real (Actualized)",
            "📊 Year-over-Year Change in Real Value"
        )
    )
    
    # Top: Property values
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Property (Nominal)"],
        name="Nominal (Paper Value)",
        line=dict(color="#3498db", width=3),
        hovertemplate="Year %{x}<br>Nominal: €%{y:,.0f}<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Property (Actualized)"],
        name="Actualized (Real Value)",
        line=dict(color="#9b59b6", width=4),
        hovertemplate="Year %{x}<br>Real: €%{y:,.0f}<extra></extra>"
    ), row=1, col=1)
    
    # Mark peak
    peak_idx = df["Property (Actualized)"].idxmax()
    peak_year = df.loc[peak_idx, "Year"]
    peak_value = df.loc[peak_idx, "Property (Actualized)"]
    
    fig.add_trace(go.Scatter(
        x=[peak_year], y=[peak_value],
        mode="markers+text",
        name="Peak Real Value",
        marker=dict(size=15, color="#e74c3c", symbol="star"),
        text=["PEAK"],
        textposition="top center",
        textfont=dict(color="#e74c3c", size=12, weight="bold"),
        hovertemplate=f"Peak at Year {peak_year}<br>€{peak_value:,.0f}<extra></extra>"
    ), row=1, col=1)
    
    # Bottom: YoY changes
    colors = ["#28a745" if v >= 0 else "#dc3545" for v in df["YoY Property Change (Act.)"]]
    fig.add_trace(go.Bar(
        x=df["Year"], y=df["YoY Property Change (Act.)"],
        name="YoY Change",
        marker_color=colors,
        hovertemplate="Year %{x}<br>Change: €%{y:,.0f}<extra></extra>",
        showlegend=False
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    
    # Loan end marker
    fig.add_vline(x=loan_years, line_dash="dash", line_color="#e74c3c", line_width=2)
    fig.add_annotation(
        x=loan_years, y=df["Property (Nominal)"].max(),
        text="🏦 LOAN ENDS",
        showarrow=True, arrowhead=2,
        ax=50, ay=-30,
        font=dict(color="#e74c3c", size=12, weight="bold"),
        bgcolor="white", bordercolor="#e74c3c", borderwidth=2
    )
    
    fig.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Value (€)", row=1, col=1)
    fig.update_yaxes(title_text="Change (€)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    
    return fig


def create_loan_cost_comparison_chart(df: pd.DataFrame, loan_amount: float, loan_years: int) -> go.Figure:
    """Chart showing nominal vs actualized loan cost over time."""
    df_loan = df[df["Year"] <= loan_years].copy()
    
    fig = go.Figure()
    
    # Nominal cost line (what you pay on paper)
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Nominal)"],
        name="Nominal Cost (What You Pay)",
        line=dict(color="#e74c3c", width=3),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.1)"
    ))
    
    # Actualized cost line (real cost)
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Actualized)"],
        name="Real Cost (Purchasing Power)",
        line=dict(color="#27ae60", width=3),
        fill="tozeroy",
        fillcolor="rgba(39, 174, 96, 0.1)"
    ))
    
    # Original loan amount reference
    fig.add_hline(
        y=loan_amount, 
        line_dash="dash", 
        line_color="#3498db",
        annotation_text=f"Original Loan: €{format_currency(loan_amount)}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text="💸 Inflation Erodes Your Loan Cost Over Time",
            font=dict(size=18)
        ),
        xaxis_title="Year",
        yaxis_title="Cumulative Cost (€)",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    return fig


def create_bank_perspective_chart(df: pd.DataFrame, loan_amount: float, loan_years: int) -> go.Figure:
    """Chart showing the bank's perspective - their real return."""
    df_loan = df[df["Year"] <= loan_years].copy()
    
    # Bank's nominal receipts vs real value
    fig = go.Figure()
    
    # What bank receives (nominal)
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Nominal)"],
        name="Bank Receives (Nominal)",
        line=dict(color="#3498db", width=3)
    ))
    
    # What it's actually worth
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Actualized)"],
        name="Actually Worth (Real)",
        line=dict(color="#e74c3c", width=3, dash="dash")
    ))
    
    # Bank's loss area
    fig.add_trace(go.Scatter(
        x=df_loan["Year"].tolist() + df_loan["Year"].tolist()[::-1],
        y=df_loan["Total Paid (Nominal)"].tolist() + df_loan["Total Paid (Actualized)"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.3)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bank's Loss (Your Gain)",
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        title=dict(
            text="🏦 Bank's Perspective: What They Receive vs What It's Worth",
            font=dict(size=16)
        ),
        xaxis_title="Year",
        yaxis_title="Amount (€)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    return fig


def create_post_loan_erosion_chart(df: pd.DataFrame, loan_years: int) -> go.Figure:
    """Chart focusing on value erosion after loan ends."""
    
    fig = go.Figure()
    
    # Full timeline
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Property (Actualized)"],
        name="Real Property Value",
        line=dict(color="#9b59b6", width=3)
    ))
    
    # Highlight post-loan period
    df_post = df[df["Year"] >= loan_years]
    fig.add_trace(go.Scatter(
        x=df_post["Year"], y=df_post["Property (Actualized)"],
        name="Post-Loan Period",
        line=dict(color="#e74c3c", width=4),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.1)"
    ))
    
    # Loan end marker
    loan_end_value = df[df["Year"] == loan_years]["Property (Actualized)"].values[0]
    fig.add_vline(x=loan_years, line_dash="dash", line_color="#2c3e50", line_width=2)
    
    fig.add_annotation(
        x=loan_years, y=loan_end_value,
        text=f"Loan Ends<br>Value: €{format_currency(loan_end_value)}",
        showarrow=True, arrowhead=2,
        ax=-80, ay=-40,
        bgcolor="white", bordercolor="#2c3e50"
    )
    
    # End value annotation
    end_value = df["Property (Actualized)"].iloc[-1]
    end_year = df["Year"].iloc[-1]
    
    fig.add_annotation(
        x=end_year, y=end_value,
        text=f"Year {end_year}<br>Value: €{format_currency(end_value)}",
        showarrow=True, arrowhead=2,
        ax=-60, ay=30,
        bgcolor="white", bordercolor="#e74c3c"
    )
    
    fig.update_layout(
        title=dict(
            text="⚠️ Without Loan Leverage, Real Value Erodes",
            font=dict(size=16)
        ),
        xaxis_title="Year",
        yaxis_title="Real Property Value (€)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("⚙️ Parameters")
    
    st.header("🏠 Property")
    property_value = st.number_input("Property Value (€)", value=250_000, step=5_000, format="%d")
    down_payment = st.number_input("Down Payment (€)", value=50_000, step=1_000, format="%d")
    loan_amount = property_value - down_payment
    
    if loan_amount > 0:
        st.success(f"**Loan Amount: €{format_currency(loan_amount)}**")
    else:
        st.error("Down payment exceeds property value!")
    
    st.divider()
    
    st.header("💳 Loan Terms")
    annual_rate = st.slider("Interest Rate (%/year)", 0.0, 10.0, 3.5, 0.1)
    loan_years = st.number_input("Loan Duration (years)", value=20, min_value=5, max_value=30)
    
    st.divider()
    
    st.header("🔮 Projection")
    projection_years = st.slider(
        "Analysis Horizon (years)",
        min_value=int(loan_years),
        max_value=40,
        value=35,
        help="Extend beyond loan to see post-loan value erosion"
    )
    
    st.divider()
    
    st.header("💰 Rental Income")
    monthly_rent = st.number_input("Monthly Rent (€)", value=1_000, step=50, format="%d")
    occupancy_rate = st.slider("Occupancy Rate (%)", 0, 100, 95)
    property_tax = st.number_input("Annual Property Tax (€)", value=5_400, step=100, format="%d")
    
    st.divider()
    
    st.header("📊 Economic Assumptions")
    inflation_rate = st.slider("General Inflation (%/year)", 0.0, 10.0, 2.5, 0.1)
    real_estate_growth = st.slider("Real Estate Growth (%/year)", -5.0, 10.0, 2.6, 0.1)
    stock_return = st.slider("Stock Market Return (%/year)", 0.0, 15.0, 10.4, 0.1)
    
    st.divider()
    
    st.header("📈 Stock Alternative Scenario")
    st.caption("If you don't buy, you rent and invest in stocks instead")
    
    rent_if_not_buying = st.number_input(
        "Rent You'd Pay (€/month)", 
        value=900, 
        step=50, 
        format="%d",
        help="If you don't buy, you still need to live somewhere. This rent is subtracted from your stock gains."
    )
    monthly_stock_investment = st.number_input(
        "Monthly Stock Investment (€)", 
        value=500, 
        step=50, 
        format="%d",
        help="How much you'd invest monthly in stocks if not buying property. Both rent and investment grow with inflation."
    )


# =============================================================================
# MAIN PAGE
# =============================================================================
st.title("🏦 Leverage vs Inflation Erosion")
st.markdown("**Understand how loans protect you from inflation — and what happens when they end.**")

if loan_amount <= 0:
    st.error("Please adjust your inputs: down payment exceeds property value.")
    st.stop()

# Run calculations
metrics = calculate_loan_amortization(
    property_value, loan_amount, annual_rate, int(loan_years), projection_years,
    inflation_rate, real_estate_growth, monthly_rent, property_tax,
    occupancy_rate, stock_return, down_payment,
    rent_if_not_buying, monthly_stock_investment
)
df = metrics.df
exit_info = analyze_exit_strategy(df, int(loan_years), inflation_rate, real_estate_growth)

# Get key values
loan_end_row = df[df["Year"] == loan_years].iloc[0]
final_row = df.iloc[-1]

# =============================================================================
# KEY INSIGHT BOXES
# =============================================================================
st.markdown("---")

# Calculate bank's loss/gain
bank_nominal_receipts = metrics.total_cost_nominal
bank_real_value = metrics.total_cost_actualized
bank_loss = bank_nominal_receipts - bank_real_value
bank_loss_pct = (bank_loss / bank_nominal_receipts) * 100

# Calculate best year to sell (when stocks overtake real estate, or peak value)
df_comparison = df.copy()
stock_overtake_years = df_comparison[df_comparison["Stock Gain (Actualized)"] > df_comparison["Net Net (Actualized)"]]
if not stock_overtake_years.empty:
    best_sell_year = int(stock_overtake_years["Year"].iloc[0])
    sell_reason = "stocks_overtake"
else:
    best_sell_year = exit_info.peak_value_year
    sell_reason = "peak_value"

# Get values at best sell year
best_sell_row = df[df["Year"] == best_sell_year].iloc[0]
value_at_best_sell = best_sell_row["Net Net (Actualized)"]
stock_at_best_sell = best_sell_row["Stock Gain (Actualized)"]

col1, col2, col3 = st.columns(3)

with col1:
    # Does the bank lose money?
    if bank_real_value < loan_amount:
        actual_loss = loan_amount - bank_real_value
        st.markdown(f"""
        <div class="bank-loss-box">
            <h3 style="margin:0; color: rgba(255,255,255,0.9);">🎉 THE BANK LOSES MONEY</h3>
            <h2>€{format_currency(actual_loss)}</h2>
            <p>The real value of what they receive (€{format_currency(bank_real_value)}) is <b>LESS</b> than what they lent you (€{format_currency(loan_amount)})!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="key-insight">
            <h3 style="margin:0; color: rgba(255,255,255,0.9);">💸 INFLATION SAVINGS</h3>
            <h2>€{format_currency(bank_loss)}</h2>
            <p>You save {bank_loss_pct:.1f}% thanks to inflation eroding your debt!</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Best year to sell
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; text-align: center;">
        <h3 style="margin:0; color: rgba(255,255,255,0.9);">🎯 BEST YEAR TO SELL</h3>
        <h2 style="color: white; margin: 10px 0;">Year {best_sell_year}</h2>
        <p style="margin:0; color: rgba(255,255,255,0.9);">Real Estate Value: €{format_currency(value_at_best_sell)}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Post-loan erosion warning
    if exit_info.annual_erosion_rate > 0:
        value_at_loan_end = loan_end_row["Property (Actualized)"]
        value_at_projection_end = final_row["Property (Actualized)"]
        erosion_amount = value_at_loan_end - value_at_projection_end
        
        st.markdown(f"""
        <div class="erosion-warning">
            <h3 style="margin:0; color: rgba(255,255,255,0.9);">⚠️ POST-LOAN VALUE EROSION</h3>
            <h2>-{exit_info.annual_erosion_rate:.1f}%/year</h2>
            <p>After the loan ends, you lose €{format_currency(erosion_amount)} in real value over {projection_years - loan_years} years!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="key-insight">
            <h3 style="margin:0; color: rgba(255,255,255,0.9);">✅ POSITIVE REAL RETURN</h3>
            <h2>+{abs(exit_info.annual_erosion_rate):.1f}%/year</h2>
            <p>Real estate growth exceeds inflation — your property gains real value over time!</p>
        </div>
        """, unsafe_allow_html=True)

# Explanation for best sell year
st.markdown("---")
if sell_reason == "stocks_overtake":
    st.info(f"""
    **🎯 Why Year {best_sell_year}?** At this point, investing in stocks ({stock_return}%/year) would give you better returns than holding the property.
    
    **Your options at Year {best_sell_year}:**
    1. **Sell & invest in stocks** — Your €{format_currency(value_at_best_sell)} could grow faster in the market
    2. **Refinance** — Take a new loan to restart your inflation shield and extract equity
    3. **Hold if you expect higher RE growth** — Only if you believe real estate will outperform {real_estate_growth}%/year
    """)
else:
    st.info(f"""
    **🎯 Why Year {best_sell_year}?** This is when your property reaches **peak real value** (actualized). After this point, inflation erodes your gains faster than real estate appreciates.
    
    **Your options at Year {best_sell_year}:**
    1. **Sell & capture gains** — Lock in €{format_currency(value_at_best_sell)} of real wealth
    2. **Refinance** — Take a new loan to restart your inflation shield (debt gets eroded again!)
    3. **Increase rent aggressively** — If you can raise rent above inflation, you offset the value erosion
    """)

# =============================================================================
# SUMMARY METRICS
# =============================================================================
st.markdown("---")
st.subheader("📊 Financial Summary at Loan End")

# Calculate additional metrics
real_interest_rate = ((1 + annual_rate/100) / (1 + inflation_rate/100) - 1) * 100
leverage_ratio = property_value / down_payment
total_nominal_paid = metrics.total_cost_nominal
property_value_at_loan_end = loan_end_row["Property (Nominal)"]
nominal_equity_at_end = property_value_at_loan_end - loan_amount  # What you "own" nominally
real_equity_at_end = loan_end_row["Property (Actualized)"] - metrics.total_cost_actualized
net_rental_income = metrics.total_rent_actualized - metrics.total_charges_actualized
gross_yield = (monthly_rent * 12 / property_value) * 100
net_yield = ((monthly_rent * 12 * occupancy_rate/100 - property_tax) / property_value) * 100

# Row 1: Loan basics
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

col_m1.metric(
    "Monthly Payment",
    f"€{format_currency(metrics.monthly_payment)}",
    help="Fixed monthly payment over loan duration"
)
col_m2.metric(
    "Total Interest (Nominal)",
    f"€{format_currency(metrics.total_interest_nominal)}",
    help="Total interest paid in nominal euros"
)
col_m3.metric(
    "Total Cost (Nominal)",
    f"€{format_currency(metrics.total_cost_nominal)}",
    help="Principal + Interest (what you pay on paper)"
)
col_m4.metric(
    "Total Cost (Real)",
    f"€{format_currency(metrics.total_cost_actualized)}",
    delta=f"-€{format_currency(bank_loss)} saved",
    delta_color="normal",
    help="What it actually costs in today's purchasing power"
)
col_m5.metric(
    "Real Interest Rate",
    f"{real_interest_rate:.2f}%",
    delta="Negative = Bank loses" if real_interest_rate < 0 else None,
    delta_color="normal" if real_interest_rate < 0 else "off",
    help="Nominal rate minus inflation. If negative, you're paid to borrow!"
)

# Row 2: Value and returns
st.markdown("**Property Value & Returns:**")
col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)

col_v1.metric(
    "Property Value (Nominal)",
    f"€{format_currency(property_value_at_loan_end)}",
    delta=f"+€{format_currency(property_value_at_loan_end - property_value)}",
    help="Market value at loan end"
)
col_v2.metric(
    "Property Value (Real)",
    f"€{format_currency(loan_end_row['Property (Actualized)'])}",
    delta=f"{'+' if loan_end_row['Property (Actualized)'] > property_value else ''}{format_currency(loan_end_row['Property (Actualized)'] - property_value)}",
    help="Real purchasing power at loan end"
)
col_v3.metric(
    "Leverage Ratio",
    f"{leverage_ratio:.1f}x",
    help=f"You control €{format_currency(property_value)} with only €{format_currency(down_payment)}"
)
col_v4.metric(
    "Gross Rental Yield",
    f"{gross_yield:.2f}%",
    help="Annual rent / Property value"
)
col_v5.metric(
    "Net Rental Yield",
    f"{net_yield:.2f}%",
    help="(Annual rent × occupancy - taxes) / Property value"
)

# Row 3: Net position
st.markdown("**Your Net Position at Loan End:**")
col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns(5)

col_n1.metric(
    "Net Equity (Nominal)",
    f"€{format_currency(loan_end_row['Net Gain (Nominal)'])}",
    help="Property value - Total loan cost (nominal)"
)
col_n2.metric(
    "Net Equity (Real)",
    f"€{format_currency(loan_end_row['Net Gain (Actualized)'])}",
    help="Property value - Total loan cost (actualized)"
)
col_n3.metric(
    "Net Rental Income (Real)",
    f"€{format_currency(net_rental_income)}",
    help="Total rent - Total charges (actualized)"
)
col_n4.metric(
    "Net Net Total (Real)",
    f"€{format_currency(loan_end_row['Net Net (Actualized)'])}",
    help="Everything combined: property + rent - costs - charges"
)
col_n5.metric(
    "Return on Down Payment",
    f"{(loan_end_row['Net Net (Actualized)'] / down_payment * 100):.0f}%",
    help=f"Net Net gain relative to your €{format_currency(down_payment)} down payment"
)

# =============================================================================
# TABS
# =============================================================================
st.markdown("---")

tabs = st.tabs([
    "🎯 Key Insight: Loan as Shield",
    "📉 Post-Loan Erosion",
    "🏦 Bank's Perspective",
    "⚖️ Nominal vs Real",
    "📈 Full Evolution",
    "💰 Rent & Charges",
    "📊 Stock Comparison",
    "🔍 Sensitivity",
    "📑 Data"
])

# TAB 1: LOAN AS INFLATION SHIELD
with tabs[0]:
    st.subheader("🎯 The Loan is Your Inflation Shield")
    
    st.markdown("""
    <div style="text-align: left; padding: 20px; background-color: #e8f4fd; border-left: 5px solid #3498db; border-radius: 8px;">
    <h4 style="color: #2c3e50; margin-top: 0;">🔑 Key Concept: Why Borrowers Win Against Inflation</h4>
    
    <p style="color: #1f1f1f;">When you take a loan, your monthly payment is <b>fixed in nominal terms</b>. But inflation makes each euro worth less over time.</p>
    
    <p style="color: #1f1f1f;"><b>Result:</b> You repay the bank with money that has less purchasing power than when you borrowed it.</p>
    
    <ul style="color: #1f1f1f;">
        <li>Year 1: Your €1,000 payment = €1,000 in today's money</li>
        <li>Year 10: Your €1,000 payment = ~€820 in today's money (at 2% inflation)</li>
        <li>Year 20: Your €1,000 payment = ~€670 in today's money</li>
    </ul>
    
    <p style="color: #1f1f1f;"><b>The bank gets the same number of euros, but those euros buy less.</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chart: Loan cost erosion
    st.plotly_chart(
        create_loan_cost_comparison_chart(df, loan_amount, int(loan_years)),
        use_container_width=True
    )
    
    # Comparison metrics
    col_a, col_b, col_c = st.columns(3)
    
    col_a.metric(
        "You Pay (Nominal)",
        f"€{format_currency(metrics.total_cost_nominal)}",
        help="Total euros you hand over"
    )
    col_b.metric(
        "Actually Costs You (Real)",
        f"€{format_currency(metrics.total_cost_actualized)}",
        help="Real purchasing power transferred"
    )
    col_c.metric(
        "Your Inflation Gain",
        f"€{format_currency(bank_loss)}",
        delta=f"{bank_loss_pct:.1f}% saved",
        help="Difference between nominal and real cost"
    )
    
    # Does bank lose money?
    st.markdown("---")
    if bank_real_value < loan_amount:
        st.success(f"""
        ### 🎉 Yes, The Bank Loses Money!
        
        - **Bank lent you:** €{format_currency(loan_amount)}
        - **Bank receives (nominal):** €{format_currency(bank_nominal_receipts)}
        - **Real value of receipts:** €{format_currency(bank_real_value)}
        
        The real value of what the bank gets back is **€{format_currency(loan_amount - bank_real_value)} LESS** than what they lent you.
        
        In real terms, the bank made a loss on this loan. **Inflation worked entirely in your favor.**
        """)
    else:
        st.info(f"""
        ### 📊 Bank Still Profits, But Less Than It Seems
        
        - **Bank lent you:** €{format_currency(loan_amount)}
        - **Bank receives (nominal):** €{format_currency(bank_nominal_receipts)}
        - **Real value of receipts:** €{format_currency(bank_real_value)}
        
        The bank's **real profit** is only €{format_currency(bank_real_value - loan_amount)}, not €{format_currency(metrics.total_interest_nominal)}.
        
        Inflation eroded **{bank_loss_pct:.1f}%** of their nominal return.
        """)

# TAB 2: POST-LOAN EROSION
with tabs[1]:
    st.subheader("📉 The Hidden Risk: What Happens After the Loan?")
    
    if exit_info.annual_erosion_rate > 0:
        st.error(f"""
        ### ⚠️ Warning: Your Real Wealth Decreases After the Loan Ends
        
        **During the loan:** Inflation erodes your debt = **you win**
        
        **After the loan:** Inflation erodes your property's real value = **you lose**
        
        With real estate growth ({real_estate_growth}%) < inflation ({inflation_rate}%), you lose **{exit_info.annual_erosion_rate:.1f}% of real value per year** after the loan ends.
        """)
    else:
        st.success(f"""
        ### ✅ Good News: Your Property Gains Real Value
        
        With real estate growth ({real_estate_growth}%) > inflation ({inflation_rate}%), your property gains **{abs(exit_info.annual_erosion_rate):.1f}% real value per year** even after the loan ends.
        """)
    
    # Post-loan erosion chart
    st.plotly_chart(
        create_post_loan_erosion_chart(df, int(loan_years)),
        use_container_width=True
    )
    
    # Value comparison table
    st.subheader("📊 Value at Key Milestones")
    
    milestones = [int(loan_years)]
    for extra in [5, 10, 15]:
        year = int(loan_years) + extra
        if year <= projection_years:
            milestones.append(year)
    
    milestone_data = []
    peak_value = exit_info.peak_value_amount
    
    for year in milestones:
        row = df[df["Year"] == year].iloc[0]
        vs_peak = row["Property (Actualized)"] - peak_value
        milestone_data.append({
            "Year": year,
            "Status": "🏦 Loan Ends" if year == loan_years else f"+{year - loan_years} years",
            "Nominal Value": f"€{format_currency(row['Property (Nominal)'])}",
            "Real Value": f"€{format_currency(row['Property (Actualized)'])}",
            "vs Peak": f"€{format_currency(vs_peak)}" if vs_peak != 0 else "PEAK",
            "Net Net (Real)": f"€{format_currency(row['Net Net (Actualized)'])}",
        })
    
    st.dataframe(pd.DataFrame(milestone_data), use_container_width=True, hide_index=True)
    
    # Strategic advice
    st.markdown("---")
    st.subheader("🎯 Strategic Recommendations")
    
    if exit_info.annual_erosion_rate > 0:
        st.markdown(f"""
        Based on your parameters, consider these strategies:
        
        1. **🏷️ Sell near Year {exit_info.peak_value_year}** — This is when your property has maximum real value.
        
        2. **🔄 Refinance at loan end** — Taking a new loan restarts your inflation shield. You get cash out AND protection against inflation.
        
        3. **📈 Increase rent above inflation** — If you can raise rent by more than {inflation_rate}%/year, you can offset the value erosion.
        
        4. **⏰ Don't wait too long** — Every year past the loan end, you lose ~€{format_currency(exit_info.peak_value_amount * exit_info.annual_erosion_rate / 100)} in real purchasing power.
        """)
    else:
        st.markdown(f"""
        Your property is in a strong position:
        
        - ✅ Real estate growth exceeds inflation
        - ✅ Property gains real value over time
        - ✅ Holding long-term is profitable
        
        The loan accelerated your gains (leverage), but you can profitably hold even after it ends.
        """)

# TAB 3: BANK'S PERSPECTIVE
with tabs[2]:
    st.subheader("🏦 From the Bank's Point of View")
    
    st.markdown("""
    Banks lend at nominal rates, but inflation erodes the real value of what they receive back.
    The green area shows how much purchasing power the bank loses.
    """)
    
    st.plotly_chart(
        create_bank_perspective_chart(df, loan_amount, int(loan_years)),
        use_container_width=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Bank Lent",
        f"€{format_currency(loan_amount)}"
    )
    col2.metric(
        "Bank Receives (Nominal)",
        f"€{format_currency(bank_nominal_receipts)}",
        delta=f"+€{format_currency(metrics.total_interest_nominal)} interest"
    )
    col3.metric(
        "Real Value of Receipts",
        f"€{format_currency(bank_real_value)}",
        delta=f"-€{format_currency(bank_loss)} vs nominal",
        delta_color="inverse"
    )
    
    st.markdown("---")
    
    # Real interest rate calculation
    nominal_rate = annual_rate
    real_rate = ((1 + nominal_rate/100) / (1 + inflation_rate/100) - 1) * 100
    
    st.subheader("📊 Interest Rate Reality Check")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Nominal Interest Rate", f"{nominal_rate:.1f}%")
    col_r2.metric("Inflation Rate", f"{inflation_rate:.1f}%")
    col_r3.metric(
        "Real Interest Rate",
        f"{real_rate:.2f}%",
        help="What the bank actually earns after inflation"
    )
    
    if real_rate < 0:
        st.error(f"""
        **The real interest rate is NEGATIVE!**
        
        The bank charges {nominal_rate:.1f}% but inflation is {inflation_rate:.1f}%. 
        In real terms, the bank is paying YOU {abs(real_rate):.2f}%/year to borrow their money!
        """)
    elif real_rate < 1:
        st.warning(f"""
        **The bank barely makes money in real terms.**
        
        Their real return is only {real_rate:.2f}%/year — barely keeping up with inflation.
        """)

# TAB 4: NOMINAL VS REAL
with tabs[3]:
    st.subheader("⚖️ The Illusion of Nominal Values")
    
    st.markdown("""
    **Nominal values** (what you see on paper) always look better than **real values** (actual purchasing power).
    This gap is the "inflation illusion" — don't let it fool you!
    """)
    
    # Gap metrics
    gap_at_end = final_row["Property (Nominal)"] - final_row["Property (Actualized)"]
    illusion_pct = (gap_at_end / final_row["Property (Nominal)"]) * 100
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        f"Nominal Value (Year {projection_years})",
        f"€{format_currency(final_row['Property (Nominal)'])}"
    )
    col2.metric(
        f"Real Value (Year {projection_years})",
        f"€{format_currency(final_row['Property (Actualized)'])}"
    )
    col3.metric(
        "Inflation Illusion",
        f"€{format_currency(gap_at_end)}",
        delta=f"{illusion_pct:.0f}% of nominal is illusion",
        delta_color="inverse"
    )
    
    # Side by side comparison
    st.markdown("---")
    st.subheader("Net Gains: Paper vs Reality")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_nom = go.Figure()
        fig_nom.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Gain (Nominal)"],
            name="Net Gain", line=dict(color="#27ae60", width=2)
        ))
        fig_nom.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Nominal)"],
            name="Net Net", line=dict(color="#2ecc71", width=3)
        ))
        fig_nom.add_hline(y=0, line_dash="dash")
        fig_nom.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig_nom.update_layout(
            title="Nominal (Paper) Gains",
            xaxis_title="Year", yaxis_title="Gain (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_nom, use_container_width=True)
        st.caption("📈 Looks great! Always going up...")
    
    with col_chart2:
        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Gain (Actualized)"],
            name="Net Gain", line=dict(color="#8e44ad", width=2)
        ))
        fig_act.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Actualized)"],
            name="Net Net", line=dict(color="#9b59b6", width=3)
        ))
        fig_act.add_hline(y=0, line_dash="dash")
        fig_act.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig_act.update_layout(
            title="Actualized (Real) Gains",
            xaxis_title="Year", yaxis_title="Gain (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_act, use_container_width=True)
        st.caption("📉 Reality: peaks then declines after loan ends")

# TAB 5: FULL EVOLUTION
with tabs[4]:
    st.subheader("📈 Complete Property Value Evolution")
    
    st.plotly_chart(
        create_main_evolution_chart(df, int(loan_years)),
        use_container_width=True
    )
    
    st.info("""
    **How to read this chart:**
    - **Blue line (Nominal):** The "sticker price" — what you'd see in real estate listings. Always rising.
    - **Purple line (Actualized):** Real purchasing power. Notice how it peaks around when the loan ends!
    - **Red star:** The optimal moment to capture maximum real value.
    - **Bottom bars:** Green = real value increased that year. Red = real value decreased.
    """)

# TAB 6: RENT & CHARGES
with tabs[5]:
    st.subheader("💰 Rental Income & Charges")
    
    # Two charts side by side: Nominal vs Actualized
    col_rent1, col_rent2 = st.columns(2)
    
    with col_rent1:
        fig_rent_nom = go.Figure()
        fig_rent_nom.add_trace(go.Bar(
            x=df["Year"], y=df["Cumul. Rent (Nominal)"],
            name="Cumulative Rent", marker_color="#27ae60"
        ))
        fig_rent_nom.add_trace(go.Bar(
            x=df["Year"], y=df["Cumul. Charges (Nominal)"],
            name="Cumulative Charges", marker_color="#e74c3c"
        ))
        fig_rent_nom.add_vline(x=loan_years, line_dash="dash", line_color="gray")
        fig_rent_nom.update_layout(
            title="Nominal Values (What You See)",
            barmode="group",
            xaxis_title="Year", yaxis_title="Amount (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_rent_nom, use_container_width=True)
    
    with col_rent2:
        fig_rent_act = go.Figure()
        fig_rent_act.add_trace(go.Bar(
            x=df["Year"], y=df["Cumul. Rent (Actualized)"],
            name="Cumulative Rent", marker_color="#2ecc71"
        ))
        fig_rent_act.add_trace(go.Bar(
            x=df["Year"], y=df["Cumul. Charges (Actualized)"],
            name="Cumulative Charges", marker_color="#c0392b"
        ))
        fig_rent_act.add_vline(x=loan_years, line_dash="dash", line_color="gray")
        fig_rent_act.update_layout(
            title="Actualized Values (Real Purchasing Power)",
            barmode="group",
            xaxis_title="Year", yaxis_title="Amount (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_rent_act, use_container_width=True)
    
    # Summary metrics - both nominal and actualized
    st.markdown("---")
    st.markdown("**At Loan End:**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric(
        "Total Rent (Nominal)", 
        f"€{format_currency(metrics.total_rent_nominal)}",
        help="Sum of all rent received in nominal euros"
    )
    col2.metric(
        "Total Rent (Real)", 
        f"€{format_currency(metrics.total_rent_actualized)}",
        delta=f"-€{format_currency(metrics.total_rent_nominal - metrics.total_rent_actualized)}",
        delta_color="inverse",
        help="Real purchasing power of rent received"
    )
    col3.metric(
        "Total Charges (Nominal)", 
        f"€{format_currency(metrics.total_charges_nominal)}",
        help="Sum of all property taxes in nominal euros"
    )
    col4.metric(
        "Total Charges (Real)", 
        f"€{format_currency(metrics.total_charges_actualized)}",
        delta=f"-€{format_currency(metrics.total_charges_nominal - metrics.total_charges_actualized)}",
        delta_color="normal",
        help="Real purchasing power of charges paid"
    )
    
    net_rent_nom = metrics.total_rent_nominal - metrics.total_charges_nominal
    net_rent_act = metrics.total_rent_actualized - metrics.total_charges_actualized
    
    col5.metric(
        "Net Rental (Nominal)", 
        f"€{format_currency(net_rent_nom)}",
        help="Rent - Charges (nominal)"
    )
    col6.metric(
        "Net Rental (Real)", 
        f"€{format_currency(net_rent_act)}",
        delta=f"-€{format_currency(net_rent_nom - net_rent_act)}",
        delta_color="inverse",
        help="Rent - Charges (actualized)"
    )
    
    # Explanation
    st.info(f"""
    **Why the difference?**
    
    - **Nominal rent** grows with inflation ({inflation_rate}%/year), so it looks like you earn more each year.
    - **Real rent** stays relatively flat — your rent just keeps up with inflation, not outpacing it.
    - **Charges** work the same way — nominal grows, real stays flat.
    
    The gap between nominal and real shows the **inflation illusion** in your rental income.
    """)

# TAB 7: STOCK COMPARISON
with tabs[6]:
    st.subheader("📊 Real Estate vs Stock Market — Fair Comparison")
    
    st.markdown(f"""
    **The scenario:** Instead of buying, you:
    - Invest your down payment (€{format_currency(down_payment)}) in stocks immediately
    - Pay rent of €{format_currency(rent_if_not_buying)}/month (growing with inflation at {inflation_rate}%/year)
    - Invest €{format_currency(monthly_stock_investment)}/month in stocks (also growing with inflation)
    - Earn {stock_return}%/year on your stock portfolio
    
    **Stock Net Gain = Portfolio Value - Total Invested - Total Rent Paid**
    """)
    
    # Key comparison metrics at loan end
    loan_end_stock = df[df["Year"] == loan_years].iloc[0]
    
    col_cmp1, col_cmp2, col_cmp3, col_cmp4 = st.columns(4)
    col_cmp1.metric(
        "Stock Portfolio (Nominal)",
        f"€{format_currency(loan_end_stock['Stock Value (Nominal)'])}",
        help="Total value of stock investments"
    )
    col_cmp2.metric(
        "Total Invested",
        f"€{format_currency(loan_end_stock['Stock Invested (Nominal)'])}",
        help="Down payment + monthly investments"
    )
    col_cmp3.metric(
        "Rent Paid (Nominal)",
        f"€{format_currency(loan_end_stock['Rent Paid if Not Buying (Nominal)'])}",
        help="Total rent paid over the period (grows with inflation)"
    )
    col_cmp4.metric(
        "Stock Net Gain (Nominal)",
        f"€{format_currency(loan_end_stock['Stock Gain (Nominal)'])}",
        help="Portfolio - Invested - Rent Paid"
    )
    
    st.markdown("---")
    
    # Comparison chart
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_stock_act = go.Figure()
        fig_stock_act.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Actualized)"],
            name="Real Estate (Net Net)", line=dict(color="#27ae60", width=3)
        ))
        fig_stock_act.add_trace(go.Scatter(
            x=df["Year"], y=df["Stock Gain (Actualized)"],
            name="Stocks (Net of Rent)", line=dict(color="#3498db", width=3)
        ))
        fig_stock_act.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig_stock_act.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_stock_act.update_layout(
            title="Actualized Gains (Real Purchasing Power)",
            xaxis_title="Year", yaxis_title="Gain (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_stock_act, use_container_width=True)
    
    with col_chart2:
        fig_stock_nom = go.Figure()
        fig_stock_nom.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Nominal)"],
            name="Real Estate (Net Net)", line=dict(color="#27ae60", width=3)
        ))
        fig_stock_nom.add_trace(go.Scatter(
            x=df["Year"], y=df["Stock Gain (Nominal)"],
            name="Stocks (Net of Rent)", line=dict(color="#3498db", width=3)
        ))
        fig_stock_nom.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig_stock_nom.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_stock_nom.update_layout(
            title="Nominal Gains",
            xaxis_title="Year", yaxis_title="Gain (€)",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_stock_nom, use_container_width=True)
    
    # Final comparison
    st.markdown("---")
    st.subheader(f"📊 Final Comparison at Year {projection_years}")
    
    final_re = final_row["Net Net (Actualized)"]
    final_stock = final_row["Stock Gain (Actualized)"]
    final_re_nom = final_row["Net Net (Nominal)"]
    final_stock_nom = final_row["Stock Gain (Nominal)"]
    
    winner_act = "🏠 Real Estate" if final_re > final_stock else "📈 Stocks"
    winner_nom = "🏠 Real Estate" if final_re_nom > final_stock_nom else "📈 Stocks"
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    col_f1.metric(
        "Real Estate (Actualized)", 
        f"€{format_currency(final_re)}",
        delta=f"€{format_currency(final_re - final_stock)} vs Stocks" if final_re > final_stock else None,
    )
    col_f2.metric(
        "Stocks (Actualized)", 
        f"€{format_currency(final_stock)}",
        delta=f"€{format_currency(final_stock - final_re)} vs RE" if final_stock > final_re else None,
    )
    col_f3.metric("Winner (Real Terms)", winner_act)
    
    # Breakdown of stock scenario
    st.markdown("---")
    st.subheader("💰 Stock Scenario Breakdown")
    
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric(
        f"Total Invested over {projection_years} years",
        f"€{format_currency(final_row['Stock Invested (Nominal)'])}"
    )
    col_b2.metric(
        f"Total Rent Paid over {projection_years} years",
        f"€{format_currency(final_row['Rent Paid if Not Buying (Nominal)'])}"
    )
    col_b3.metric(
        "Final Portfolio Value",
        f"€{format_currency(final_row['Stock Value (Nominal)'])}"
    )
    col_b4.metric(
        "Net Gain After Rent",
        f"€{format_currency(final_row['Stock Gain (Nominal)'])}"
    )
    
    st.info(f"""
    **Key insight:** The rent you'd pay if not buying (€{format_currency(rent_if_not_buying)}/month, growing with inflation) 
    significantly impacts the stock scenario. This makes the comparison fair — you can't live for free!
    
    **Real Estate advantage:** You stop paying "rent" (mortgage) after {loan_years} years, while renters pay forever.
    
    **Stock advantage:** More liquidity, no property management, potentially higher returns if market outperforms RE.
    """)

# TAB 8: SENSITIVITY
with tabs[7]:
    st.subheader("🔍 Sensitivity Analysis")
    
    st.markdown("""
    These charts show how your results change with different assumptions.
    The **inflection point** is where gains turn to losses (or vice versa).
    """)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    # Interest rate sensitivity
    rate_range, net_rate, net_net_rate = compute_rate_sensitivity(
        property_value, loan_amount, int(loan_years), inflation_rate,
        real_estate_growth, monthly_rent, property_tax,
        occupancy_rate, stock_return, down_payment,
        rent_if_not_buying, monthly_stock_investment
    )
    
    with col_s1:
        st.plotly_chart(
            create_sensitivity_chart(rate_range, net_rate, net_net_rate, 
                                     "Interest Rate Impact", "Rate (%)", "#17becf"),
            use_container_width=True
        )
        inflection_net = rate_range[np.argmin(np.abs(net_rate))]
        inflection_net_net = rate_range[np.argmin(np.abs(net_net_rate))]
        st.metric("Break-even Rate (Net)", f"{inflection_net:.1f}%")
        st.metric("Break-even Rate (Net Net)", f"{inflection_net_net:.1f}%")
    
    # Market sensitivity
    market_range, net_market, net_net_market = compute_market_sensitivity(
        property_value, int(loan_years), inflation_rate,
        metrics.total_cost_actualized, metrics.total_rent_actualized, metrics.total_charges_actualized
    )
    
    with col_s2:
        st.plotly_chart(
            create_sensitivity_chart(market_range, net_market, net_net_market,
                                     "RE Market Impact", "Growth (%)", "#ff7f0e"),
            use_container_width=True
        )
        inflection_net = market_range[np.argmin(np.abs(net_market))]
        inflection_net_net = market_range[np.argmin(np.abs(net_net_market))]
        st.metric("Break-even Growth (Net)", f"{inflection_net:.1f}%")
        st.metric("Break-even Growth (Net Net)", f"{inflection_net_net:.1f}%")
    
    # Inflation sensitivity
    inf_range, net_inf, net_net_inf = compute_inflation_sensitivity(
        property_value, loan_amount, annual_rate, int(loan_years),
        real_estate_growth, monthly_rent, property_tax,
        occupancy_rate, stock_return, down_payment,
        rent_if_not_buying, monthly_stock_investment
    )
    
    with col_s3:
        st.plotly_chart(
            create_sensitivity_chart(inf_range, net_inf, net_net_inf,
                                     "Inflation Impact", "Inflation (%)", "#e377c2"),
            use_container_width=True
        )
        inflection_net = inf_range[np.argmin(np.abs(net_inf))]
        inflection_net_net = inf_range[np.argmin(np.abs(net_net_inf))]
        st.metric("Break-even Inflation (Net)", f"{inflection_net:.1f}%")
        st.metric("Break-even Inflation (Net Net)", f"{inflection_net_net:.1f}%")

# TAB 9: DATA
with tabs[8]:
    st.subheader("📑 Complete Data Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="leverage_vs_erosion_analysis.csv",
        mime="text/csv"
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("""
**Glossary:**
- **Nominal:** Face value in euros (what you see on paper)
- **Actualized:** Real purchasing power (adjusted for inflation)
- **Net:** Property value minus loan cost
- **Net Net:** Net plus rental income minus charges
- **Discount Factor:** How much €1 future is worth today
""")