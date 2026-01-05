"""
LEVERAGE VS INFLATION EROSION
============================
A comprehensive real estate investment analyzer that reveals:
1. How loans protect you from inflation (the bank can actually LOSE money)
2. What happens when the loan ends (you lose your inflation shield)

Author: Financial Education Tool
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Leverage vs Inflation Erosion",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLES
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }
    
    /* Metric styling */
    [data-testid="stMetricValue"] { 
        color: #1f1f1f !important; 
        font-weight: 600; 
        font-size: 1.3rem;
    }
    [data-testid="stMetricLabel"] { 
        color: #4b4b4b !important; 
        font-weight: 500;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    /* Key insight boxes with gradients */
    .bank-loss-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    .bank-loss-box h2, .bank-loss-box h3, .bank-loss-box p { color: white; margin: 5px 0; }
    
    .best-sell-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .best-sell-box h2, .best-sell-box h3, .best-sell-box p { color: white; margin: 5px 0; }
    
    .erosion-warning-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    .erosion-warning-box h2, .erosion-warning-box h3, .erosion-warning-box p { color: white; margin: 5px 0; }
    
    .positive-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
    }
    .positive-box h2, .positive-box h3, .positive-box p { color: white; margin: 5px 0; }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #3498db;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1f1f1f;
    }
    .info-box h4 { color: #2c3e50; margin-top: 0; }
    .info-box p, .info-box li { color: #1f1f1f; }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1f1f1f;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1f1f1f;
    }
    
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1f1f1f;
    }
    
    /* Formula boxes */
    .formula-box {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #1f1f1f;
    }
    
    /* Section headers */
    .section-divider {
        border-top: 3px solid #3498db;
        margin: 30px 0 20px 0;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class LoanMetrics:
    """
    Complete loan calculation results.
    
    Contains both NOMINAL values (what you see on paper) and 
    ACTUALIZED values (real purchasing power adjusted for inflation).
    """
    df: pd.DataFrame                    # Full yearly data
    monthly_payment: float              # Fixed monthly payment
    
    # Nominal values (face value in euros)
    total_interest_nominal: float       # Sum of all interest payments
    total_cost_nominal: float           # Principal + Interest
    
    # Actualized values (purchasing power in today's euros)
    total_interest_actualized: float    # Interest adjusted for inflation
    total_principal_actualized: float   # Principal adjusted for inflation
    total_cost_actualized: float        # Total real cost
    
    # Cash flows from rental
    total_rent_nominal: float
    total_rent_actualized: float
    total_charges_nominal: float
    total_charges_actualized: float


@dataclass
class ExitAnalysis:
    """Analysis of optimal exit timing and value erosion."""
    peak_value_year: int                # Year of maximum actualized property value
    peak_value_amount: float            # Maximum actualized property value
    peak_net_net_year: int              # Year of maximum total gain
    peak_net_net_amount: float          # Maximum total gain
    first_decline_year: Optional[int]   # First year property loses real value
    annual_erosion_rate: float          # RE growth - Inflation (can be negative)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def format_currency(value: float, show_decimals: bool = False) -> str:
    """Format number as currency with space as thousands separator."""
    if show_decimals:
        return f"{value:,.2f}".replace(",", " ")
    return f"{value:,.0f}".replace(",", " ")


def format_percent(value: float, decimals: int = 2) -> str:
    """Format number as percentage with sign."""
    return f"{value:+.{decimals}f}%" if value != 0 else f"{value:.{decimals}f}%"


def calculate_monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate fixed monthly payment using the annuity formula.
    
    Formula: PMT = P × [r(1+r)^n] / [(1+r)^n - 1]
    Where:
        P = Principal (loan amount)
        r = Monthly interest rate (annual rate / 12)
        n = Total number of payments (years × 12)
    """
    if annual_rate <= 0:
        return principal / (years * 12)
    
    monthly_rate = (annual_rate / 100) / 12
    num_payments = years * 12
    
    numerator = monthly_rate * (1 + monthly_rate) ** num_payments
    denominator = (1 + monthly_rate) ** num_payments - 1
    
    return principal * (numerator / denominator)


def calculate_discount_factor(inflation_rate: float, year: int) -> float:
    """
    Calculate the discount factor to convert future money to today's value.
    
    Formula: DF = 1 / (1 + inflation)^year
    
    Example at 2.5% inflation:
        Year 1:  DF = 1/1.025^1  = 0.976 (€1 future = €0.976 today)
        Year 10: DF = 1/1.025^10 = 0.781 (€1 future = €0.781 today)
        Year 20: DF = 1/1.025^20 = 0.610 (€1 future = €0.610 today)
    """
    return 1 / ((1 + inflation_rate / 100) ** year)


# =============================================================================
# CORE CALCULATION ENGINE
# =============================================================================
@st.cache_data
def calculate_full_analysis(
    property_value: float,
    loan_amount: float,
    annual_rate: float,
    loan_years: int,
    projection_years: int,
    inflation_rate: float,
    real_estate_growth: float,
    monthly_rent_income: float,
    annual_property_tax: float,
    occupancy_rate: float,
    stock_return: float,
    down_payment: float,
    rent_if_not_buying: float,
    monthly_stock_investment: float,
) -> LoanMetrics:
    """
    Complete loan amortization and investment analysis.
    
    This function calculates:
    1. Loan amortization (monthly payments split into principal/interest)
    2. Property value evolution (nominal and actualized)
    3. Rental income and charges
    4. Stock market alternative scenario
    5. Wealth creation sources (leverage, rent, appreciation)
    
    KEY CONCEPTS:
    -------------
    NOMINAL: Face value in euros (what you see on bank statements)
    ACTUALIZED: Real purchasing power (adjusted for inflation)
    
    The difference reveals how inflation benefits borrowers:
    - You repay with money that's worth less over time
    - The bank receives euros that buy less
    """
    
    # ===================
    # SETUP & CONSTANTS
    # ===================
    monthly_rate = (annual_rate / 100) / 12
    total_loan_months = loan_years * 12
    monthly_payment = calculate_monthly_payment(loan_amount, annual_rate, loan_years)
    
    # Growth/discount factors
    inflation_factor = 1 + inflation_rate / 100      # e.g., 1.025 for 2.5%
    re_growth_factor = 1 + real_estate_growth / 100  # e.g., 1.026 for 2.6%
    stock_monthly_growth = 1 + (stock_return / 100) / 12
    occupancy_factor = occupancy_rate / 100
    
    # ===================
    # RUNNING TOTALS
    # ===================
    remaining_principal = loan_amount
    
    # Nominal (face value)
    cumul_interest_nom = 0.0
    cumul_principal_paid_nom = 0.0
    cumul_rent_income_nom = 0.0
    cumul_charges_nom = 0.0
    
    # Actualized (real value)
    cumul_interest_act = 0.0
    cumul_principal_paid_act = 0.0
    cumul_rent_income_act = 0.0
    cumul_charges_act = 0.0
    
    # Stock alternative scenario
    stock_portfolio_value = down_payment
    total_stock_invested = down_payment
    cumul_rent_paid_if_renting_nom = 0.0
    cumul_rent_paid_if_renting_act = 0.0
    
    # ===================
    # YEARLY CALCULATIONS
    # ===================
    yearly_data = []
    
    for year in range(1, projection_years + 1):
        is_loan_active = year <= loan_years
        year_interest = 0.0
        year_principal = 0.0
        
        # Inflation adjustment for this year's cash flows
        # Year 1 uses base amounts, Year 2 uses base × inflation, etc.
        inflation_multiplier = inflation_factor ** (year - 1)
        
        # Stock scenario: monthly investment and rent grow with inflation
        current_monthly_stock_invest = monthly_stock_investment * inflation_multiplier
        current_monthly_rent_expense = rent_if_not_buying * inflation_multiplier
        
        # ---------------------
        # MONTHLY CALCULATIONS
        # ---------------------
        for month in range(12):
            # LOAN PAYMENTS (only during loan period)
            if is_loan_active and remaining_principal > 0:
                # Interest portion: remaining balance × monthly rate
                interest_payment = remaining_principal * monthly_rate
                
                # Principal portion: monthly payment - interest
                principal_payment = min(monthly_payment - interest_payment, remaining_principal)
                
                year_interest += interest_payment
                year_principal += principal_payment
                remaining_principal -= principal_payment
            
            # STOCK SCENARIO (runs entire projection)
            # Portfolio grows, then we add new investment
            stock_portfolio_value = stock_portfolio_value * stock_monthly_growth + current_monthly_stock_invest
            total_stock_invested += current_monthly_stock_invest
            
            # Track rent you'd pay if not buying
            cumul_rent_paid_if_renting_nom += current_monthly_rent_expense
        
        # ---------------------
        # YEARLY AGGREGATIONS
        # ---------------------
        
        # Discount factor for this year
        discount_factor = calculate_discount_factor(inflation_rate, year)
        
        # PROPERTY VALUES
        property_nominal = property_value * (re_growth_factor ** year)
        property_actualized = property_nominal * discount_factor
        
        # RENTAL INCOME (if you own and rent out)
        annual_rent_income_nom = (monthly_rent_income * 12) * inflation_multiplier * occupancy_factor
        annual_charges_nom = annual_property_tax * inflation_multiplier
        
        # CUMULATIVE UPDATES
        cumul_interest_nom += year_interest
        cumul_principal_paid_nom += year_principal
        cumul_interest_act += year_interest * discount_factor
        cumul_principal_paid_act += year_principal * discount_factor
        
        cumul_rent_income_nom += annual_rent_income_nom
        cumul_rent_income_act += annual_rent_income_nom * discount_factor
        cumul_charges_nom += annual_charges_nom
        cumul_charges_act += annual_charges_nom * discount_factor
        
        cumul_rent_paid_if_renting_act += (current_monthly_rent_expense * 12) * discount_factor
        
        # ---------------------
        # DERIVED METRICS
        # ---------------------
        
        # Total loan cost
        total_paid_nom = cumul_interest_nom + cumul_principal_paid_nom
        total_paid_act = cumul_interest_act + cumul_principal_paid_act
        
        # NET GAIN = Property Value - Loan Cost
        net_gain_nom = property_nominal - (loan_amount + cumul_interest_nom)
        net_gain_act = property_actualized - total_paid_act
        
        # NET NET = Net + Rent - Charges (complete picture)
        net_net_nom = net_gain_nom + cumul_rent_income_nom - cumul_charges_nom
        net_net_act = net_gain_act + cumul_rent_income_act - cumul_charges_act
        
        # STOCK SCENARIO RESULTS
        stock_portfolio_act = stock_portfolio_value * discount_factor
        # Stock gain = Portfolio - Invested - Rent Paid (fair comparison)
        stock_gain_nom = stock_portfolio_value - total_stock_invested - cumul_rent_paid_if_renting_nom
        stock_gain_act = stock_portfolio_act - (total_stock_invested * discount_factor) - cumul_rent_paid_if_renting_act
        
        # WEALTH SOURCES (for decomposition chart)
        # 1. Leverage effect = Nominal payments - Actualized payments
        leverage_effect = total_paid_nom - total_paid_act
        
        # 2. RE Real Appreciation = Actualized property - Initial value
        re_real_gain = property_actualized - property_value
        
        # 3. Net Rental = Rent received - Charges paid (actualized)
        net_rental_gain = cumul_rent_income_act - cumul_charges_act
        
        # ---------------------
        # STORE YEARLY DATA
        # ---------------------
        yearly_data.append({
            "Year": year,
            "Loan Active": "✓" if is_loan_active else "✗",
            "Remaining Principal": round(max(0, remaining_principal), 0),
            
            # Property values
            "Property (Nominal)": round(property_nominal, 0),
            "Property (Actualized)": round(property_actualized, 0),
            
            # Loan costs
            "Year Interest": round(year_interest, 0),
            "Year Principal": round(year_principal, 0),
            "Cumul. Interest (Nominal)": round(cumul_interest_nom, 0),
            "Cumul. Interest (Actualized)": round(cumul_interest_act, 0),
            "Total Paid (Nominal)": round(total_paid_nom, 0),
            "Total Paid (Actualized)": round(total_paid_act, 0),
            
            # Rental cash flows
            "Annual Rent Income": round(annual_rent_income_nom, 0),
            "Annual Charges": round(annual_charges_nom, 0),
            "Cumul. Rent (Nominal)": round(cumul_rent_income_nom, 0),
            "Cumul. Rent (Actualized)": round(cumul_rent_income_act, 0),
            "Cumul. Charges (Nominal)": round(cumul_charges_nom, 0),
            "Cumul. Charges (Actualized)": round(cumul_charges_act, 0),
            
            # Net results
            "Net Gain (Nominal)": round(net_gain_nom, 0),
            "Net Gain (Actualized)": round(net_gain_act, 0),
            "Net Net (Nominal)": round(net_net_nom, 0),
            "Net Net (Actualized)": round(net_net_act, 0),
            
            # Stock comparison
            "Stock Value (Nominal)": round(stock_portfolio_value, 0),
            "Stock Value (Actualized)": round(stock_portfolio_act, 0),
            "Stock Invested (Nominal)": round(total_stock_invested, 0),
            "Rent Paid if Renting (Nominal)": round(cumul_rent_paid_if_renting_nom, 0),
            "Rent Paid if Renting (Actualized)": round(cumul_rent_paid_if_renting_act, 0),
            "Stock Gain (Nominal)": round(stock_gain_nom, 0),
            "Stock Gain (Actualized)": round(stock_gain_act, 0),
            
            # Wealth decomposition
            "Leverage Effect": round(leverage_effect, 0),
            "RE Real Gain": round(re_real_gain, 0),
            "Net Rental Gain": round(net_rental_gain, 0),
            
            # Helpers
            "Discount Factor": round(discount_factor, 4),
        })
    
    # ===================
    # BUILD DATAFRAME
    # ===================
    df = pd.DataFrame(yearly_data)
    
    # Add year-over-year changes
    df["YoY Property (Nominal)"] = df["Property (Nominal)"].diff().fillna(0)
    df["YoY Property (Actualized)"] = df["Property (Actualized)"].diff().fillna(0)
    df["YoY Leverage Effect"] = df["Leverage Effect"].diff().fillna(0)
    df["YoY Net Rental"] = df["Net Rental Gain"].diff().fillna(0)
    df["YoY RE Real Gain"] = df["RE Real Gain"].diff().fillna(0)
    df["YoY Net Net (Actualized)"] = df["Net Net (Actualized)"].diff().fillna(0)
    
    # ===================
    # EXTRACT LOAN-END VALUES
    # ===================
    loan_end_row = df[df["Year"] == loan_years].iloc[0]
    
    return LoanMetrics(
        df=df,
        monthly_payment=monthly_payment,
        total_interest_nominal=loan_end_row["Cumul. Interest (Nominal)"],
        total_cost_nominal=loan_amount + loan_end_row["Cumul. Interest (Nominal)"],
        total_interest_actualized=loan_end_row["Cumul. Interest (Actualized)"],
        total_principal_actualized=loan_end_row["Total Paid (Actualized)"] - loan_end_row["Cumul. Interest (Actualized)"],
        total_cost_actualized=loan_end_row["Total Paid (Actualized)"],
        total_rent_nominal=loan_end_row["Cumul. Rent (Nominal)"],
        total_rent_actualized=loan_end_row["Cumul. Rent (Actualized)"],
        total_charges_nominal=loan_end_row["Cumul. Charges (Nominal)"],
        total_charges_actualized=loan_end_row["Cumul. Charges (Actualized)"],
    )


@st.cache_data
def analyze_exit_strategy(df: pd.DataFrame, loan_years: int, inflation_rate: float, real_estate_growth: float) -> ExitAnalysis:
    """
    Determine optimal exit timing based on actualized values.
    
    Key insight: The best time to sell is often around when the loan ends,
    because that's when you stop benefiting from leverage (debt erosion).
    """
    
    # Find peak actualized property value
    peak_idx = df["Property (Actualized)"].idxmax()
    peak_year = int(df.loc[peak_idx, "Year"])
    peak_value = df.loc[peak_idx, "Property (Actualized)"]
    
    # Find peak total gain (Net Net)
    peak_nn_idx = df["Net Net (Actualized)"].idxmax()
    peak_nn_year = int(df.loc[peak_nn_idx, "Year"])
    peak_nn_value = df.loc[peak_nn_idx, "Net Net (Actualized)"]
    
    # Find first year of decline (after loan)
    post_loan = df[df["Year"] > loan_years]
    declining = post_loan[post_loan["YoY Property (Actualized)"] < 0]
    first_decline = int(declining["Year"].iloc[0]) if not declining.empty else None
    
    # Annual erosion rate = RE growth - Inflation
    # Positive = property gains real value
    # Negative = property loses real value
    erosion_rate = real_estate_growth - inflation_rate
    
    return ExitAnalysis(
        peak_value_year=peak_year,
        peak_value_amount=peak_value,
        peak_net_net_year=peak_nn_year,
        peak_net_net_amount=peak_nn_value,
        first_decline_year=first_decline,
        annual_erosion_rate=erosion_rate,
    )


# =============================================================================
# SENSITIVITY ANALYSIS FUNCTIONS
# =============================================================================
@st.cache_data
def sensitivity_interest_rate(
    property_value: float, loan_amount: float, loan_years: int,
    inflation_rate: float, real_estate_growth: float, monthly_rent: float,
    property_tax: float, occupancy_rate: float, stock_return: float, 
    down_payment: float, rent_if_not_buying: float, monthly_stock_investment: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How do results change with different interest rates?"""
    rates = np.linspace(0, 15, 80)
    net_values, net_net_values = [], []
    
    final_property_act = (
        property_value * 
        ((1 + real_estate_growth / 100) ** loan_years) / 
        ((1 + inflation_rate / 100) ** loan_years)
    )
    
    for rate in rates:
        result = calculate_full_analysis(
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
def sensitivity_re_market(
    property_value: float, loan_years: int, inflation_rate: float,
    total_cost_act: float, total_rent_act: float, total_charges_act: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How do results change with different RE growth rates?"""
    rates = np.linspace(-5, 15, 80)
    
    final_values = (
        property_value * 
        ((1 + rates / 100) ** loan_years) / 
        ((1 + inflation_rate / 100) ** loan_years)
    )
    
    net_values = final_values - total_cost_act
    net_net_values = (final_values + total_rent_act) - (total_cost_act + total_charges_act)
    
    return rates, net_values, net_net_values


@st.cache_data
def sensitivity_inflation(
    property_value: float, loan_amount: float, annual_rate: float, loan_years: int,
    real_estate_growth: float, monthly_rent: float, property_tax: float,
    occupancy_rate: float, stock_return: float, down_payment: float,
    rent_if_not_buying: float, monthly_stock_investment: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How do results change with different inflation rates?"""
    rates = np.linspace(0, 10, 80)
    net_values, net_net_values = [], []
    
    property_growth_factor = (1 + real_estate_growth / 100) ** loan_years
    
    for inf_rate in rates:
        result = calculate_full_analysis(
            property_value, loan_amount, annual_rate, loan_years, loan_years,
            inf_rate, real_estate_growth, monthly_rent, property_tax,
            occupancy_rate, stock_return, down_payment,
            rent_if_not_buying, monthly_stock_investment
        )
        final_act = property_value * property_growth_factor / ((1 + inf_rate / 100) ** loan_years)
        net_values.append(final_act - result.total_cost_actualized)
        net_net_values.append(
            (final_act + result.total_rent_actualized) - 
            (result.total_cost_actualized + result.total_charges_actualized)
        )
    
    return rates, np.array(net_values), np.array(net_net_values)


# =============================================================================
# CHART BUILDERS
# =============================================================================
def build_wealth_sources_chart(df: pd.DataFrame, loan_years: int) -> go.Figure:
    """
    Stacked area chart showing the THREE sources of wealth creation.
    
    Key insight: Leverage effect STOPS after loan ends!
    """
    fig = go.Figure()
    
    # Ensure leverage effect stays constant after loan ends
    df_chart = df.copy()
    max_leverage = df_chart[df_chart["Year"] == loan_years]["Leverage Effect"].values[0]
    df_chart.loc[df_chart["Year"] > loan_years, "Leverage Effect"] = max_leverage
    
    # Layer 1: Leverage Effect (bottom)
    fig.add_trace(go.Scatter(
        x=df_chart["Year"], 
        y=df_chart["Leverage Effect"],
        name="🏦 Leverage Effect (Debt Erosion)",
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.7)",
        line=dict(color="#27ae60", width=2),
        hovertemplate="Year %{x}<br>Leverage: €%{y:,.0f}<extra></extra>"
    ))
    
    # Layer 2: + Net Rental
    layer2 = df_chart["Leverage Effect"] + df_chart["Net Rental Gain"]
    fig.add_trace(go.Scatter(
        x=df_chart["Year"], 
        y=layer2,
        name="💰 + Net Rental Income",
        fill="tonexty",
        fillcolor="rgba(52, 152, 219, 0.7)",
        line=dict(color="#3498db", width=2),
        hovertemplate="Year %{x}<br>Leverage + Rent: €%{y:,.0f}<extra></extra>"
    ))
    
    # Layer 3: + RE Appreciation
    layer3 = layer2 + df_chart["RE Real Gain"]
    fig.add_trace(go.Scatter(
        x=df_chart["Year"], 
        y=layer3,
        name="🏠 + Property Appreciation",
        fill="tonexty",
        fillcolor="rgba(155, 89, 182, 0.7)",
        line=dict(color="#9b59b6", width=2),
        hovertemplate="Year %{x}<br>Total: €%{y:,.0f}<extra></extra>"
    ))
    
    # Loan end marker
    fig.add_vline(x=loan_years, line_dash="dash", line_color="#e74c3c", line_width=3)
    fig.add_annotation(
        x=loan_years, y=layer3.max(),
        text="🏦 LOAN ENDS<br>Leverage stops!",
        showarrow=True, arrowhead=2,
        ax=70, ay=-40,
        font=dict(color="#e74c3c", size=12, weight="bold"),
        bgcolor="white", bordercolor="#e74c3c", borderwidth=2, borderpad=4
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title=dict(text="📊 Cumulative Wealth by Source (in Today's €)", font=dict(size=18)),
        xaxis_title="Year",
        yaxis_title="Cumulative Gain (€)",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    return fig


def build_annual_contribution_chart(df: pd.DataFrame, loan_years: int) -> go.Figure:
    """Bar chart showing year-over-year contributions from each source."""
    
    # Cap leverage YoY at 0 after loan
    df_chart = df.copy()
    df_chart.loc[df_chart["Year"] > loan_years, "YoY Leverage Effect"] = 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_chart["Year"], 
        y=df_chart["YoY Leverage Effect"],
        name="🏦 Leverage",
        marker_color="#27ae60"
    ))
    
    fig.add_trace(go.Bar(
        x=df_chart["Year"], 
        y=df_chart["YoY Net Rental"],
        name="💰 Net Rental",
        marker_color="#3498db"
    ))
    
    fig.add_trace(go.Bar(
        x=df_chart["Year"], 
        y=df_chart["YoY RE Real Gain"],
        name="🏠 RE Appreciation",
        marker_color="#9b59b6"
    ))
    
    fig.add_vline(x=loan_years + 0.5, line_dash="dash", line_color="#e74c3c", line_width=2)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title=dict(text="📊 Annual Wealth Contribution by Source", font=dict(size=16)),
        barmode="relative",
        xaxis_title="Year",
        yaxis_title="Annual Contribution (€)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    return fig


def build_property_evolution_chart(df: pd.DataFrame, loan_years: int) -> go.Figure:
    """Main chart showing nominal vs actualized property values."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Property Value: Nominal vs Real", "Year-over-Year Real Change")
    )
    
    # Top: Property values
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Property (Nominal)"],
        name="Nominal (Paper Value)",
        line=dict(color="#3498db", width=3),
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Property (Actualized)"],
        name="Actualized (Real Value)",
        line=dict(color="#9b59b6", width=4),
    ), row=1, col=1)
    
    # Mark peak
    peak_idx = df["Property (Actualized)"].idxmax()
    peak_year = df.loc[peak_idx, "Year"]
    peak_value = df.loc[peak_idx, "Property (Actualized)"]
    
    fig.add_trace(go.Scatter(
        x=[peak_year], y=[peak_value],
        mode="markers+text",
        name="Peak",
        marker=dict(size=15, color="#e74c3c", symbol="star"),
        text=["PEAK"],
        textposition="top center",
        textfont=dict(color="#e74c3c", size=11, weight="bold"),
    ), row=1, col=1)
    
    # Bottom: YoY changes
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in df["YoY Property (Actualized)"]]
    fig.add_trace(go.Bar(
        x=df["Year"], y=df["YoY Property (Actualized)"],
        name="YoY Change",
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    
    # Loan end marker
    fig.add_vline(x=loan_years, line_dash="dash", line_color="#e74c3c", line_width=2)
    
    fig.update_layout(
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Value (€)", row=1, col=1)
    fig.update_yaxes(title_text="YoY (€)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    
    return fig


def build_loan_erosion_chart(df: pd.DataFrame, loan_amount: float, loan_years: int) -> go.Figure:
    """Chart showing how inflation erodes the real cost of the loan."""
    
    df_loan = df[df["Year"] <= loan_years].copy()
    
    fig = go.Figure()
    
    # Area between nominal and actualized = your savings
    fig.add_trace(go.Scatter(
        x=df_loan["Year"].tolist() + df_loan["Year"].tolist()[::-1],
        y=df_loan["Total Paid (Nominal)"].tolist() + df_loan["Total Paid (Actualized)"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.3)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Your Savings (Inflation Gain)",
        hoverinfo="skip"
    ))
    
    # Nominal line
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Nominal)"],
        name="Nominal Cost (What You Pay)",
        line=dict(color="#e74c3c", width=3),
    ))
    
    # Actualized line
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Actualized)"],
        name="Real Cost (Purchasing Power)",
        line=dict(color="#27ae60", width=3),
    ))
    
    # Reference line for loan amount
    fig.add_hline(
        y=loan_amount, 
        line_dash="dot", 
        line_color="#3498db",
        annotation_text=f"Original Loan: €{format_currency(loan_amount)}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text="💸 Inflation Erodes Your Loan Cost", font=dict(size=18)),
        xaxis_title="Year",
        yaxis_title="Cumulative Payment (€)",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    return fig


def build_bank_perspective_chart(df: pd.DataFrame, loan_amount: float, loan_years: int) -> go.Figure:
    """Show what the bank receives vs what it's actually worth."""
    
    df_loan = df[df["Year"] <= loan_years].copy()
    
    fig = go.Figure()
    
    # Bank's loss area
    fig.add_trace(go.Scatter(
        x=df_loan["Year"].tolist() + df_loan["Year"].tolist()[::-1],
        y=df_loan["Total Paid (Nominal)"].tolist() + df_loan["Total Paid (Actualized)"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bank's Loss",
        hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Nominal)"],
        name="Bank Receives (Nominal)",
        line=dict(color="#3498db", width=3),
    ))
    
    fig.add_trace(go.Scatter(
        x=df_loan["Year"], y=df_loan["Total Paid (Actualized)"],
        name="Actually Worth (Real)",
        line=dict(color="#e74c3c", width=3, dash="dash"),
    ))
    
    fig.add_hline(y=loan_amount, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=dict(text="🏦 Bank's Perspective: Nominal vs Real Value Received", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Amount (€)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified"
    )
    
    return fig


def build_sensitivity_chart(x: np.ndarray, y_net: np.ndarray, y_net_net: np.ndarray, 
                           title: str, x_label: str, color: str) -> go.Figure:
    """Sensitivity analysis chart with break-even markers."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y_net,
        name="Net (Property only)",
        line=dict(color=color, dash="dot", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=y_net_net,
        name="Net Net (Total)",
        line=dict(color=color, width=4)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # Mark break-even points
    net_be_idx = np.argmin(np.abs(y_net))
    nn_be_idx = np.argmin(np.abs(y_net_net))
    
    fig.add_vline(x=x[net_be_idx], line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=x[nn_be_idx], line_dash="dot", line_color=color, opacity=0.7)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_label,
        yaxis_title="Gain (€)",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# SIDEBAR - USER INPUTS
# =============================================================================
with st.sidebar:
    st.title("⚙️ Parameters")
    
    # PROPERTY
    st.header("🏠 Property")
    property_value = st.number_input(
        "Property Value (€)", 
        value=250_000, 
        step=5_000, 
        format="%d",
        help="Purchase price of the property"
    )
    down_payment = st.number_input(
        "Down Payment (€)", 
        value=50_000, 
        step=1_000, 
        format="%d",
        help="Your initial cash investment"
    )
    loan_amount = property_value - down_payment
    
    if loan_amount > 0:
        st.success(f"**Loan Amount: €{format_currency(loan_amount)}**")
    else:
        st.error("❌ Down payment exceeds property value!")
    
    st.divider()
    
    # LOAN TERMS
    st.header("💳 Loan Terms")
    annual_rate = st.slider(
        "Interest Rate (%/year)", 
        0.0, 10.0, 3.5, 0.1,
        help="Annual nominal interest rate"
    )
    loan_years = st.number_input(
        "Loan Duration (years)", 
        value=20, 
        min_value=5, 
        max_value=30,
        help="Number of years to repay the loan"
    )
    
    st.divider()
    
    # PROJECTION
    st.header("🔮 Projection Horizon")
    projection_years = st.slider(
        "Analysis Period (years)",
        min_value=int(loan_years),
        max_value=40,
        value=35,
        help="Extend beyond loan term to see what happens AFTER"
    )
    
    st.divider()
    
    # RENTAL INCOME
    st.header("💰 Rental Income (If You Rent Out)")
    monthly_rent = st.number_input(
        "Monthly Rent (€)", 
        value=1_000, 
        step=50, 
        format="%d",
        help="Monthly rental income if you rent out the property"
    )
    occupancy_rate = st.slider(
        "Occupancy Rate (%)", 
        0, 100, 95,
        help="% of time property is rented (accounts for vacancies)"
    )
    property_tax = st.number_input(
        "Annual Property Tax (€)", 
        value=2_500, 
        step=100, 
        format="%d",
        help="Annual taxes and fixed charges"
    )
    
    st.divider()
    
    # ECONOMIC ASSUMPTIONS
    st.header("📊 Economic Assumptions")
    inflation_rate = st.slider(
        "General Inflation (%/year)", 
        0.0, 10.0, 2.5, 0.1,
        help="Expected annual inflation rate"
    )
    real_estate_growth = st.slider(
        "Real Estate Growth (%/year)", 
        -5.0, 10.0, 2.6, 0.1,
        help="Expected annual property value growth"
    )
    stock_return = st.slider(
        "Stock Market Return (%/year)", 
        0.0, 15.0, 10.4, 0.1,
        help="Expected annual stock market return"
    )
    
    st.divider()
    
    # STOCK ALTERNATIVE
    st.header("📈 Alternative: Rent & Invest")
    st.caption("If you DON'T buy, you need to rent somewhere")
    
    rent_if_not_buying = st.number_input(
        "Rent You'd Pay (€/month)", 
        value=900, 
        step=50, 
        format="%d",
        help="Monthly rent if you choose not to buy (subtracted from stock gains)"
    )
    monthly_stock_investment = st.number_input(
        "Monthly Stock Investment (€)", 
        value=500, 
        step=50, 
        format="%d",
        help="How much you'd invest monthly in stocks"
    )


# =============================================================================
# MAIN PAGE
# =============================================================================
st.title("🏦 Leverage vs Inflation Erosion")

st.markdown("""
<div class="info-box">
<h4 style="color: #2c3e50;">📚 What This Tool Reveals</h4>
<p style="color: #1f1f1f;">This analyzer shows TWO key insights most people miss:</p>
<ol style="color: #1f1f1f;">
<li><b>Loans PROTECT you from inflation</b> — You repay with money worth less each year. The bank can actually LOSE money in real terms!</li>
<li><b>After the loan ends, you lose this protection</b> — Your property may start LOSING real value if RE growth &lt; inflation.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Validate inputs
if loan_amount <= 0:
    st.error("❌ Please adjust inputs: down payment exceeds property value.")
    st.stop()

# =============================================================================
# RUN CALCULATIONS
# =============================================================================
metrics = calculate_full_analysis(
    property_value, loan_amount, annual_rate, int(loan_years), projection_years,
    inflation_rate, real_estate_growth, monthly_rent, property_tax,
    occupancy_rate, stock_return, down_payment,
    rent_if_not_buying, monthly_stock_investment
)

df = metrics.df
exit_info = analyze_exit_strategy(df, int(loan_years), inflation_rate, real_estate_growth)

# Key derived values
loan_end_row = df[df["Year"] == loan_years].iloc[0]
final_row = df.iloc[-1]

# Bank analysis
bank_receives_nominal = metrics.total_cost_nominal
bank_receives_real = metrics.total_cost_actualized
bank_loss = bank_receives_nominal - bank_receives_real
bank_loss_pct = (bank_loss / bank_receives_nominal) * 100
bank_actually_loses = bank_receives_real < loan_amount
bank_real_loss = loan_amount - bank_receives_real if bank_actually_loses else 0

# Real interest rate
real_interest_rate = ((1 + annual_rate/100) / (1 + inflation_rate/100) - 1) * 100

# Best year to sell (when stocks overtake OR peak value)
stock_overtake = df[df["Stock Gain (Actualized)"] > df["Net Net (Actualized)"]]
if not stock_overtake.empty:
    best_sell_year = int(stock_overtake["Year"].iloc[0])
    sell_reason = "stocks_overtake"
else:
    best_sell_year = exit_info.peak_net_net_year
    sell_reason = "peak_value"

best_sell_row = df[df["Year"] == best_sell_year].iloc[0]


# =============================================================================
# KEY INSIGHT BOXES (TOP OF PAGE)
# =============================================================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

# BOX 1: Bank Loss / Inflation Savings
with col1:
    if bank_actually_loses:
        st.markdown(f"""
        <div class="bank-loss-box">
            <h3>🎉 THE BANK LOSES MONEY!</h3>
            <h2 style="font-size: 2.2em;">€{format_currency(bank_real_loss)}</h2>
            <p><b>They lent:</b> €{format_currency(loan_amount)}</p>
            <p><b>They receive (real):</b> €{format_currency(bank_receives_real)}</p>
            <p style="margin-top:10px; font-size: 0.9em;">The bank gets back LESS purchasing power than they lent you!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bank-loss-box">
            <h3>💸 YOUR INFLATION SAVINGS</h3>
            <h2 style="font-size: 2.2em;">€{format_currency(bank_loss)}</h2>
            <p><b>You pay (nominal):</b> €{format_currency(bank_receives_nominal)}</p>
            <p><b>Real cost:</b> €{format_currency(bank_receives_real)}</p>
            <p style="margin-top:10px; font-size: 0.9em;">You save <b>{bank_loss_pct:.1f}%</b> thanks to inflation!</p>
        </div>
        """, unsafe_allow_html=True)

# BOX 2: Best Year to Sell
with col2:
    st.markdown(f"""
    <div class="best-sell-box">
        <h3>🎯 OPTIMAL EXIT YEAR</h3>
        <h2 style="font-size: 2.5em;">Year {best_sell_year}</h2>
        <p><b>Net Net Value:</b> €{format_currency(best_sell_row['Net Net (Actualized)'])}</p>
        <p style="margin-top:10px; font-size: 0.9em;">{'Stocks outperform after this' if sell_reason == 'stocks_overtake' else 'Peak total real value'}</p>
    </div>
    """, unsafe_allow_html=True)

# BOX 3: Post-Loan Reality
with col3:
    if exit_info.annual_erosion_rate > 0:
        st.markdown(f"""
        <div class="positive-box">
            <h3>✅ POSITIVE REAL RETURN</h3>
            <h2 style="font-size: 2.2em;">+{exit_info.annual_erosion_rate:.1f}%/year</h2>
            <p>RE growth ({real_estate_growth}%) &gt; Inflation ({inflation_rate}%)</p>
            <p style="margin-top:10px; font-size: 0.9em;">Property gains real value even after loan!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        erosion_pct = abs(exit_info.annual_erosion_rate)
        st.markdown(f"""
        <div class="erosion-warning-box">
            <h3>⚠️ VALUE EROSION AFTER LOAN</h3>
            <h2 style="font-size: 2.2em;">-{erosion_pct:.1f}%/year</h2>
            <p>RE growth ({real_estate_growth}%) &lt; Inflation ({inflation_rate}%)</p>
            <p style="margin-top:10px; font-size: 0.9em;">Property loses real value after loan ends!</p>
        </div>
        """, unsafe_allow_html=True)

# Explanation for best sell year
st.markdown("---")
if sell_reason == "stocks_overtake":
    st.info(f"""
    **🎯 Why Year {best_sell_year}?** At this point, the stock market ({stock_return}%/year) would give you better returns than holding the property.
    
    **Your options:**
    1. **Sell & invest in stocks** — Capture €{format_currency(best_sell_row['Net Net (Actualized)'])} and grow faster
    2. **Refinance** — New loan = new inflation shield, extract equity tax-efficiently  
    3. **Hold** — Only if you expect RE to outperform ({real_estate_growth}% currently)
    """)
else:
    st.info(f"""
    **🎯 Why Year {best_sell_year}?** This is when your **total real wealth** (property + rent - costs) peaks.
    
    **Your options:**
    1. **Sell** — Lock in €{format_currency(best_sell_row['Net Net (Actualized)'])} of real wealth
    2. **Refinance** — Take new loan to restart inflation protection
    3. **Increase rent** — Offset value erosion with higher rental income
    """)


# =============================================================================
# FINANCIAL SUMMARY
# =============================================================================
st.markdown("---")
st.subheader(f"📊 Financial Summary at Loan End (Year {loan_years})")

# Calculated values
leverage_ratio = property_value / down_payment
property_value_at_end = loan_end_row["Property (Nominal)"]
property_real_at_end = loan_end_row["Property (Actualized)"]
net_rental_real = metrics.total_rent_actualized - metrics.total_charges_actualized
gross_yield = (monthly_rent * 12 / property_value) * 100
net_yield = ((monthly_rent * 12 * occupancy_rate/100 - property_tax) / property_value) * 100
roi_on_dp = (loan_end_row['Net Net (Actualized)'] / down_payment) * 100

# ROW 1: Loan Costs
st.markdown("**💳 Loan Costs:**")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Monthly Payment",
    f"€{format_currency(metrics.monthly_payment)}",
    help=f"""
**FORMULA:** PMT = P × [r(1+r)ⁿ] / [(1+r)ⁿ-1]

**YOUR CALCULATION:**
• Principal (P) = €{format_currency(loan_amount)}
• Monthly rate (r) = {annual_rate}% ÷ 12 = {annual_rate/12:.4f}%
• Payments (n) = {loan_years} × 12 = {loan_years*12}

**RESULT:** €{format_currency(metrics.monthly_payment)}/month
    """
)

c2.metric(
    "Total Interest (Nominal)",
    f"€{format_currency(metrics.total_interest_nominal)}",
    help=f"""
**FORMULA:** Total Interest = (PMT × n) - Principal

**YOUR CALCULATION:**
• Monthly Payment = €{format_currency(metrics.monthly_payment)}
• Payments = {loan_years*12}
• Total Paid = €{format_currency(metrics.monthly_payment * loan_years * 12)}
• Principal = €{format_currency(loan_amount)}

**RESULT:** €{format_currency(metrics.total_interest_nominal)}
    """
)

c3.metric(
    "Total Cost (Nominal)",
    f"€{format_currency(metrics.total_cost_nominal)}",
    help=f"""
**FORMULA:** Total Cost = Principal + Interest

**YOUR CALCULATION:**
• Principal = €{format_currency(loan_amount)}
• Interest = €{format_currency(metrics.total_interest_nominal)}

**RESULT:** €{format_currency(metrics.total_cost_nominal)}
    """
)

c4.metric(
    "Total Cost (Real)",
    f"€{format_currency(metrics.total_cost_actualized)}",
    delta=f"-€{format_currency(bank_loss)} saved",
    help=f"""
**CONCEPT:** Each payment worth LESS due to inflation.

**FORMULA:** Real Cost = Σ (Payment / (1+inflation)^year)

**DISCOUNT FACTORS at {inflation_rate}%:**
• Year 1: €1 = €{1/(1+inflation_rate/100)**1:.3f} today
• Year 10: €1 = €{1/(1+inflation_rate/100)**10:.3f} today
• Year {loan_years}: €1 = €{1/(1+inflation_rate/100)**loan_years:.3f} today

**YOUR SAVINGS:** €{format_currency(bank_loss)} ({bank_loss_pct:.1f}%)
    """
)

c5.metric(
    "Real Interest Rate",
    f"{real_interest_rate:.2f}%",
    delta="Bank loses!" if real_interest_rate < 0 else None,
    delta_color="normal" if real_interest_rate < 0 else "off",
    help=f"""
**FORMULA:** Real Rate = [(1+Nominal)/(1+Inflation)] - 1

**YOUR CALCULATION:**
• Nominal = {annual_rate}%
• Inflation = {inflation_rate}%
• Real = [{1+annual_rate/100:.4f}/{1+inflation_rate/100:.4f}] - 1

**RESULT:** {real_interest_rate:.2f}%

{'⚠️ NEGATIVE = Bank LOSES money!' if real_interest_rate < 0 else ''}
    """
)

# ROW 2: Property Values
st.markdown("**🏠 Property Value & Returns:**")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Property (Nominal)",
    f"€{format_currency(property_value_at_end)}",
    delta=f"+€{format_currency(property_value_at_end - property_value)}",
    help=f"""
**FORMULA:** Value = Initial × (1+RE_growth)^years

**YOUR CALCULATION:**
• Initial = €{format_currency(property_value)}
• RE Growth = {real_estate_growth}%
• Years = {loan_years}

**RESULT:** €{format_currency(property_value_at_end)}
    """
)

c2.metric(
    "Property (Real)",
    f"€{format_currency(property_real_at_end)}",
    delta=f"{'+' if property_real_at_end > property_value else ''}{format_currency(property_real_at_end - property_value)}",
    help=f"""
**FORMULA:** Real = Nominal / (1+Inflation)^years

**YOUR CALCULATION:**
• Nominal = €{format_currency(property_value_at_end)}
• Inflation = {inflation_rate}%
• Years = {loan_years}

**RESULT:** €{format_currency(property_real_at_end)}
    """
)

c3.metric(
    "Leverage Ratio",
    f"{leverage_ratio:.1f}x",
    help=f"""
**FORMULA:** Leverage = Property / Down Payment

**YOUR CALCULATION:**
• Property = €{format_currency(property_value)}
• Down Payment = €{format_currency(down_payment)}

**RESULT:** {leverage_ratio:.1f}x

You control €{leverage_ratio:.0f} per €1 invested!
    """
)

c4.metric(
    "Gross Yield",
    f"{gross_yield:.2f}%",
    help=f"""
**FORMULA:** Yield = (Rent × 12) / Property

**YOUR CALCULATION:**
• Annual Rent = €{format_currency(monthly_rent * 12)}
• Property = €{format_currency(property_value)}

**RESULT:** {gross_yield:.2f}%
    """
)

c5.metric(
    "Net Yield",
    f"{net_yield:.2f}%",
    help=f"""
**FORMULA:** Net = (Rent×Occupancy - Tax) / Property

**YOUR CALCULATION:**
• Gross Rent = €{format_currency(monthly_rent * 12)}
• After {occupancy_rate}% occupancy = €{format_currency(monthly_rent * 12 * occupancy_rate/100)}
• Minus Tax = €{format_currency(property_tax)}
• Net Income = €{format_currency(monthly_rent * 12 * occupancy_rate/100 - property_tax)}

**RESULT:** {net_yield:.2f}%
    """
)

# ROW 3: Net Position
st.markdown("**💰 Your Net Position:**")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Net Equity (Nominal)",
    f"€{format_currency(loan_end_row['Net Gain (Nominal)'])}",
    help=f"""
**FORMULA:** Net = Property - Loan Cost

**YOUR CALCULATION:**
• Property = €{format_currency(property_value_at_end)}
• Loan Cost = €{format_currency(metrics.total_cost_nominal)}

**RESULT:** €{format_currency(loan_end_row['Net Gain (Nominal)'])}
    """
)

c2.metric(
    "Net Equity (Real)",
    f"€{format_currency(loan_end_row['Net Gain (Actualized)'])}",
    help=f"""
**FORMULA:** Net Real = Property (Real) - Cost (Real)

**YOUR CALCULATION:**
• Property Real = €{format_currency(property_real_at_end)}
• Cost Real = €{format_currency(metrics.total_cost_actualized)}

**RESULT:** €{format_currency(loan_end_row['Net Gain (Actualized)'])}
    """
)

c3.metric(
    "Net Rental (Real)",
    f"€{format_currency(net_rental_real)}",
    help=f"""
**FORMULA:** Net Rental = Rent (Real) - Charges (Real)

**YOUR CALCULATION:**
• Rent Real = €{format_currency(metrics.total_rent_actualized)}
• Charges Real = €{format_currency(metrics.total_charges_actualized)}

**RESULT:** €{format_currency(net_rental_real)}
    """
)

c4.metric(
    "Net Net Total (Real)",
    f"€{format_currency(loan_end_row['Net Net (Actualized)'])}",
    help=f"""
**FORMULA:** Net Net = Net Equity + Net Rental

**YOUR CALCULATION:**
• Net Equity Real = €{format_currency(loan_end_row['Net Gain (Actualized)'])}
• Net Rental Real = €{format_currency(net_rental_real)}

**RESULT:** €{format_currency(loan_end_row['Net Net (Actualized)'])}

This is your TOTAL real wealth gain!
    """
)

c5.metric(
    "Return on Down Payment",
    f"{roi_on_dp:.0f}%",
    help=f"""
**FORMULA:** ROI = Net Net / Down Payment × 100

**YOUR CALCULATION:**
• Net Net = €{format_currency(loan_end_row['Net Net (Actualized)'])}
• Down Payment = €{format_currency(down_payment)}

**RESULT:** {roi_on_dp:.0f}%

That's {roi_on_dp/100:.1f}x your money!
    """
)


# =============================================================================
# DETAILED TABS
# =============================================================================
st.markdown("---")

tabs = st.tabs([
    "🎯 Loan as Shield",
    "📉 After the Loan",
    "🏦 Bank's View",
    "⚖️ Nominal vs Real",
    "📈 Property Evolution",
    "💰 Rent & Charges",
    "📊 Stock Comparison",
    "🔍 Sensitivity",
    "📑 Data"
])

# TAB 1: LOAN AS INFLATION SHIELD
with tabs[0]:
    st.subheader("🎯 The Loan is Your Inflation Shield")
    
    st.markdown(f"""
    <div class="info-box">
    <h4 style="color: #2c3e50;">🔑 Key Concept: Why Borrowers Win</h4>
    <p style="color: #1f1f1f;">Your monthly payment is <b>fixed</b>, but inflation makes each euro worth less.</p>
    <p style="color: #1f1f1f;"><b>Result:</b> You repay with money that has LESS purchasing power.</p>
    <div class="formula-box">
    Example at {inflation_rate}% inflation:<br>
    • Year 1: €1,000 = €1,000 today<br>
    • Year 10: €1,000 = €{1000/(1+inflation_rate/100)**10:,.0f} today<br>
    • Year {loan_years}: €1,000 = €{1000/(1+inflation_rate/100)**loan_years:,.0f} today
    </div>
    <p style="color: #1f1f1f;"><b>The bank gets the same euros, but they buy less.</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(build_loan_erosion_chart(df, loan_amount, int(loan_years)), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("You Pay (Nominal)", f"€{format_currency(bank_receives_nominal)}")
    col2.metric("Real Cost", f"€{format_currency(bank_receives_real)}")
    col3.metric("Your Savings", f"€{format_currency(bank_loss)}", delta=f"{bank_loss_pct:.1f}%")
    
    if bank_actually_loses:
        st.success(f"🎉 **THE BANK LOSES €{format_currency(bank_real_loss)}!** They get back less purchasing power than they lent.")
    else:
        st.info(f"📊 Bank's real profit: €{format_currency(bank_receives_real - loan_amount)} (not €{format_currency(metrics.total_interest_nominal)})")

# TAB 2: AFTER THE LOAN
with tabs[1]:
    st.subheader("📉 What Happens After the Loan?")
    
    if exit_info.annual_erosion_rate > 0:
        st.success(f"✅ RE growth ({real_estate_growth}%) > inflation ({inflation_rate}%) = property gains real value")
    else:
        st.error(f"⚠️ RE growth ({real_estate_growth}%) < inflation ({inflation_rate}%) = property LOSES real value")
    
    st.plotly_chart(build_wealth_sources_chart(df, int(loan_years)), use_container_width=True)
    
    st.markdown(f"""
    <div class="info-box">
    <h4 style="color: #2c3e50;">📊 Reading the Chart</h4>
    <ul style="color: #1f1f1f;">
        <li>🟢 <b>Green:</b> Leverage (debt erosion) — STOPS after loan!</li>
        <li>🔵 <b>Blue:</b> Net rental income</li>
        <li>🟣 <b>Purple:</b> Property appreciation (can be negative)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(build_annual_contribution_chart(df, int(loan_years)), use_container_width=True)

# TAB 3: BANK'S VIEW
with tabs[2]:
    st.subheader("🏦 Bank's Perspective")
    st.plotly_chart(build_bank_perspective_chart(df, loan_amount, int(loan_years)), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Bank Lent", f"€{format_currency(loan_amount)}")
    col2.metric("Receives (Nominal)", f"€{format_currency(bank_receives_nominal)}")
    col3.metric("Worth (Real)", f"€{format_currency(bank_receives_real)}", delta=f"-€{format_currency(bank_loss)}", delta_color="inverse")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nominal Rate", f"{annual_rate:.1f}%")
    col2.metric("Inflation", f"{inflation_rate:.1f}%")
    col3.metric("Real Rate", f"{real_interest_rate:.2f}%")

# TAB 4: NOMINAL VS REAL
with tabs[3]:
    st.subheader("⚖️ The Illusion of Nominal Values")
    
    st.markdown("""
    <div class="info-box">
    <p style="color: #1f1f1f;"><b>Nominal values</b> (what you see on paper) always look better than <b>real values</b> (actual purchasing power). 
    This gap is the "inflation illusion" — don't let it fool you!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    gap_at_end = final_row["Property (Nominal)"] - final_row["Property (Actualized)"]
    illusion_pct = (gap_at_end / final_row["Property (Nominal)"]) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Nominal Value (Year {projection_years})",
        f"€{format_currency(final_row['Property (Nominal)'])}",
        help="What real estate agents would quote"
    )
    col2.metric(
        f"Real Value (Year {projection_years})",
        f"€{format_currency(final_row['Property (Actualized)'])}",
        help="What it's worth in today's purchasing power"
    )
    col3.metric(
        "Inflation Illusion",
        f"€{format_currency(gap_at_end)}",
        delta=f"{illusion_pct:.0f}% is just inflation",
        delta_color="inverse",
        help="The difference is NOT real wealth — it's just inflation!"
    )
    
    st.markdown("---")
    st.subheader("📊 Net Net Gains: Paper vs Reality")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Gain (Nominal)"],
            name="Net (Property only)",
            line=dict(color="#27ae60", width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Nominal)"],
            name="Net Net (Total)",
            line=dict(color="#27ae60", width=3)
        ))
        fig.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="📈 Nominal Gains (Paper Value)",
            xaxis_title="Year",
            yaxis_title="Gain (€)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📈 Looks great — always rising!")
        
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Gain (Actualized)"],
            name="Net (Property only)",
            line=dict(color="#9b59b6", width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Actualized)"],
            name="Net Net (Total)",
            line=dict(color="#9b59b6", width=3)
        ))
        fig.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="📉 Real Gains (Purchasing Power)",
            xaxis_title="Year",
            yaxis_title="Gain (€)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📉 Reality: peaks then can decline after loan ends")
    
    # Comparison at key points
    st.markdown("---")
    st.subheader("📊 The Gap at Key Points")
    
    key_years = [int(loan_years), projection_years]
    if int(loan_years) + 10 <= projection_years:
        key_years.insert(1, int(loan_years) + 10)
    
    gap_data = []
    for year in key_years:
        row = df[df["Year"] == year].iloc[0]
        nominal = row["Net Net (Nominal)"]
        real = row["Net Net (Actualized)"]
        gap = nominal - real
        gap_data.append({
            "Year": year,
            "Net Net (Nominal)": f"€{format_currency(nominal)}",
            "Net Net (Real)": f"€{format_currency(real)}",
            "Illusion Gap": f"€{format_currency(gap)}",
            "% Illusion": f"{(gap/nominal*100) if nominal > 0 else 0:.0f}%"
        })
    
    st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)

# TAB 5: PROPERTY EVOLUTION
with tabs[4]:
    st.subheader("📈 Property Evolution")
    st.plotly_chart(build_property_evolution_chart(df, int(loan_years)), use_container_width=True)

# TAB 6: RENT & CHARGES
with tabs[5]:
    st.subheader("💰 Rent & Charges")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Rent (Nominal)"], name="Rent", marker_color="#27ae60"))
        fig.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Charges (Nominal)"], name="Charges", marker_color="#e74c3c"))
        fig.update_layout(title="Nominal", barmode="group", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Rent (Actualized)"], name="Rent", marker_color="#2ecc71"))
        fig.add_trace(go.Bar(x=df["Year"], y=df["Cumul. Charges (Actualized)"], name="Charges", marker_color="#c0392b"))
        fig.update_layout(title="Real", barmode="group", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rent (Nom)", f"€{format_currency(metrics.total_rent_nominal)}")
    col2.metric("Rent (Real)", f"€{format_currency(metrics.total_rent_actualized)}")
    col3.metric("Charges (Nom)", f"€{format_currency(metrics.total_charges_nominal)}")
    col4.metric("Charges (Real)", f"€{format_currency(metrics.total_charges_actualized)}")

# TAB 7: STOCK COMPARISON
with tabs[6]:
    st.subheader("📊 Real Estate vs Stock Market — Fair Comparison")
    
    st.markdown(f"""
    <div class="info-box">
    <h4 style="color: #2c3e50;">The Scenario: Instead of buying, you:</h4>
    <ul style="color: #1f1f1f;">
        <li>Invest your down payment (<b>€{format_currency(down_payment)}</b>) in stocks immediately</li>
        <li>Pay rent of <b>€{format_currency(rent_if_not_buying)}/month</b> (growing with inflation at {inflation_rate}%/year)</li>
        <li>Invest <b>€{format_currency(monthly_stock_investment)}/month</b> in stocks (also growing with inflation)</li>
        <li>Earn <b>{stock_return}%/year</b> on your stock portfolio</li>
    </ul>
    <div class="formula-box">
    <b>Stock Net Gain = Portfolio Value - Total Invested - Total Rent Paid</b>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics at loan end
    loan_end_stock = df[df["Year"] == loan_years].iloc[0]
    
    st.markdown("---")
    st.subheader(f"📊 Stock Scenario at Loan End (Year {loan_years})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Stock Portfolio (Nominal)",
        f"€{format_currency(loan_end_stock['Stock Value (Nominal)'])}",
        help="Total value of your stock investments"
    )
    col2.metric(
        "Total Invested",
        f"€{format_currency(loan_end_stock['Stock Invested (Nominal)'])}",
        help="Down payment + all monthly investments"
    )
    col3.metric(
        "Rent Paid (Nominal)",
        f"€{format_currency(loan_end_stock['Rent Paid if Renting (Nominal)'])}",
        help="Total rent paid over the period (grows with inflation)"
    )
    col4.metric(
        "Stock Net Gain (Nominal)",
        f"€{format_currency(loan_end_stock['Stock Gain (Nominal)'])}",
        help="Portfolio - Invested - Rent Paid"
    )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Actualized)"],
            name="Real Estate (Net Net)",
            line=dict(color="#27ae60", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Stock Gain (Actualized)"],
            name="Stocks (Net of Rent)",
            line=dict(color="#3498db", width=3)
        ))
        fig.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="📉 Actualized Gains (Real Purchasing Power)",
            xaxis_title="Year",
            yaxis_title="Gain (€)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Net Net (Nominal)"],
            name="Real Estate (Net Net)",
            line=dict(color="#27ae60", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Stock Gain (Nominal)"],
            name="Stocks (Net of Rent)",
            line=dict(color="#3498db", width=3)
        ))
        fig.add_vline(x=loan_years, line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="📈 Nominal Gains",
            xaxis_title="Year",
            yaxis_title="Gain (€)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Final comparison
    st.markdown("---")
    st.subheader(f"📊 Final Comparison at Year {projection_years}")
    
    final_re_act = final_row["Net Net (Actualized)"]
    final_stock_act = final_row["Stock Gain (Actualized)"]
    final_re_nom = final_row["Net Net (Nominal)"]
    final_stock_nom = final_row["Stock Gain (Nominal)"]
    
    winner = "🏠 Real Estate" if final_re_act > final_stock_act else "📈 Stocks"
    diff = abs(final_re_act - final_stock_act)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Real Estate (Actualized)",
        f"€{format_currency(final_re_act)}",
        help="Net Net total in today's purchasing power"
    )
    col2.metric(
        "Stocks (Actualized)",
        f"€{format_currency(final_stock_act)}",
        delta=f"€{format_currency(diff)} vs RE" if final_stock_act > final_re_act else None,
        help="Stock gains minus rent paid, in today's purchasing power"
    )
    col3.metric(
        "Winner (Real Terms)",
        winner,
        help="Which strategy gives more real wealth"
    )
    
    # Stock scenario breakdown
    st.markdown("---")
    st.subheader("💰 Stock Scenario Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Total Invested over {projection_years} years",
        f"€{format_currency(final_row['Stock Invested (Nominal)'])}",
        help="Down payment + all monthly contributions"
    )
    col2.metric(
        f"Total Rent Paid over {projection_years} years",
        f"€{format_currency(final_row['Rent Paid if Renting (Nominal)'])}",
        help="Cumulative rent (growing with inflation)"
    )
    col3.metric(
        "Final Portfolio Value",
        f"€{format_currency(final_row['Stock Value (Nominal)'])}",
        help="What your stock portfolio is worth"
    )
    col4.metric(
        "Net Gain After Rent",
        f"€{format_currency(final_row['Stock Gain (Nominal)'])}",
        help="Portfolio - Invested - Rent = Your real gain"
    )
    
    # Key insight
    st.markdown(f"""
    <div class="info-box">
    <h4 style="color: #2c3e50;">💡 Key Insights</h4>
    <p style="color: #1f1f1f;"><b>The rent you'd pay if not buying</b> (€{format_currency(rent_if_not_buying)}/month, growing with inflation) 
    significantly impacts the stock scenario. This makes the comparison fair — you can't live for free!</p>
    <ul style="color: #1f1f1f;">
        <li><b>🏠 Real Estate advantage:</b> You stop paying "rent" (mortgage) after {loan_years} years, while renters pay forever.</li>
        <li><b>📈 Stock advantage:</b> More liquidity, no property management, potentially higher returns if market outperforms RE.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 8: SENSITIVITY
with tabs[7]:
    st.subheader("🔍 Sensitivity Analysis")
    
    st.markdown("""
    <div class="info-box">
    <p style="color: #1f1f1f;">These charts show how your results change with different assumptions. 
    The <b>break-even point</b> is where gains turn to losses (crossing the €0 line).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Interest Rate Sensitivity
    rates_ir, net_ir, nn_ir = sensitivity_interest_rate(
        property_value, loan_amount, int(loan_years), inflation_rate, real_estate_growth,
        monthly_rent, property_tax, occupancy_rate, stock_return, down_payment,
        rent_if_not_buying, monthly_stock_investment
    )
    with col1:
        st.plotly_chart(
            build_sensitivity_chart(rates_ir, net_ir, nn_ir, "📊 Interest Rate Impact", "Rate (%)", "#17becf"),
            use_container_width=True
        )
        be_net_ir = rates_ir[np.argmin(np.abs(net_ir))]
        be_nn_ir = rates_ir[np.argmin(np.abs(nn_ir))]
        st.metric(
            "Break-even (Net)", 
            f"{be_net_ir:.1f}%",
            help="Interest rate where Net Gain = €0"
        )
        st.metric(
            "Break-even (Net Net)", 
            f"{be_nn_ir:.1f}%",
            help="Interest rate where Net Net = €0"
        )
        st.caption(f"Current rate: {annual_rate}%")
    
    # RE Growth Sensitivity
    rates_re, net_re, nn_re = sensitivity_re_market(
        property_value, int(loan_years), inflation_rate,
        metrics.total_cost_actualized, metrics.total_rent_actualized, metrics.total_charges_actualized
    )
    with col2:
        st.plotly_chart(
            build_sensitivity_chart(rates_re, net_re, nn_re, "📊 RE Growth Impact", "Growth (%)", "#ff7f0e"),
            use_container_width=True
        )
        be_net_re = rates_re[np.argmin(np.abs(net_re))]
        be_nn_re = rates_re[np.argmin(np.abs(nn_re))]
        st.metric(
            "Break-even (Net)", 
            f"{be_net_re:.1f}%",
            help="RE growth rate where Net Gain = €0"
        )
        st.metric(
            "Break-even (Net Net)", 
            f"{be_nn_re:.1f}%",
            help="RE growth rate where Net Net = €0"
        )
        st.caption(f"Current growth: {real_estate_growth}%")
    
    # Inflation Sensitivity
    rates_inf, net_inf, nn_inf = sensitivity_inflation(
        property_value, loan_amount, annual_rate, int(loan_years), real_estate_growth,
        monthly_rent, property_tax, occupancy_rate, stock_return, down_payment,
        rent_if_not_buying, monthly_stock_investment
    )
    with col3:
        st.plotly_chart(
            build_sensitivity_chart(rates_inf, net_inf, nn_inf, "📊 Inflation Impact", "Inflation (%)", "#e377c2"),
            use_container_width=True
        )
        be_net_inf = rates_inf[np.argmin(np.abs(net_inf))]
        be_nn_inf = rates_inf[np.argmin(np.abs(nn_inf))]
        st.metric(
            "Break-even (Net)", 
            f"{be_net_inf:.1f}%",
            help="Inflation rate where Net Gain = €0"
        )
        st.metric(
            "Break-even (Net Net)", 
            f"{be_nn_inf:.1f}%",
            help="Inflation rate where Net Net = €0"
        )
        st.caption(f"Current inflation: {inflation_rate}%")
    
    # Summary table
    st.markdown("---")
    st.subheader("📊 Break-even Summary")
    
    be_summary = pd.DataFrame({
        "Variable": ["Interest Rate", "RE Growth", "Inflation"],
        "Current Value": [f"{annual_rate}%", f"{real_estate_growth}%", f"{inflation_rate}%"],
        "Break-even (Net)": [f"{be_net_ir:.1f}%", f"{be_net_re:.1f}%", f"{be_net_inf:.1f}%"],
        "Break-even (Net Net)": [f"{be_nn_ir:.1f}%", f"{be_nn_re:.1f}%", f"{be_nn_inf:.1f}%"],
        "Margin (Net Net)": [
            f"{be_nn_ir - annual_rate:+.1f}pp",
            f"{real_estate_growth - be_nn_re:+.1f}pp",
            f"{be_nn_inf - inflation_rate:+.1f}pp"
        ]
    })
    st.dataframe(be_summary, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
    <h4 style="color: #2c3e50;">📖 How to Read</h4>
    <ul style="color: #1f1f1f;">
        <li><b>Net:</b> Property value - Loan cost (property only)</li>
        <li><b>Net Net:</b> Net + Rent - Charges (complete picture)</li>
        <li><b>Margin:</b> How far you are from break-even (positive = safe)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 9: DATA
with tabs[8]:
    st.subheader("📑 Data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False), "analysis.csv", "text/csv")


# =============================================================================
# GLOSSARY
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="info-box">
<h4 style="color: #2c3e50;">📖 Glossary</h4>
<ul style="color: #1f1f1f;">
    <li><b>Nominal:</b> Face value (bank statements)</li>
    <li><b>Actualized/Real:</b> Purchasing power (today's €)</li>
    <li><b>Net:</b> Property - Loan Cost</li>
    <li><b>Net Net:</b> Net + Rent - Charges</li>
    <li><b>Leverage Effect:</b> Nominal payments - Real payments</li>
</ul>
</div>
""", unsafe_allow_html=True)