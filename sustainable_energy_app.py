import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('global-data-on-sustainable-energy.csv')
    
    # Convert columns to numeric where possible
    numeric_cols = ['Renewable energy share in the total final energy consumption (%)', 
                    'gdp_per_capita', 'Value_co2_emissions_kt_by_country']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter for latest year with sufficient data
    latest_year = df['Year'].max()
    latest_df = df[df['Year'] == latest_year].copy()
    
    # Create composite sustainability score
    latest_df['Sustainability Score'] = (
        latest_df['Renewable energy share in the total final energy consumption (%)'].fillna(0) +
        latest_df['Access to electricity (% of population)'].fillna(0) +
        (100 - latest_df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].fillna(0)/10)
    )
    
    return df, latest_df, latest_year

df, latest_df, latest_year = load_data()

# Streamlit app
st.title("🌍 Global Sustainable Energy Analysis")
st.markdown("### Identifying Investment Opportunities in Renewable Energy")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries in Dataset", df['Entity'].nunique())
col2.metric("Years Covered", f"{df['Year'].min()}-{df['Year'].max()}")
col3.metric("Avg. Renewable Share", f"{df['Renewable energy share in the total final energy consumption (%)'].mean():.1f}%")
col4.metric("Avg. Electricity Access", f"{df['Access to electricity (% of population)'].mean():.1f}%")

st.divider()

# Visualization 1: Sustainability Score Map
st.subheader("Investment Readiness by Country")
st.markdown("**Sustainability Score** (Higher = Better Investment Opportunity)")

# Create investment readiness score
scaler = MinMaxScaler()
investment_factors = latest_df[[
    'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'gdp_per_capita',
    'Renewable-electricity-generating-capacity-per-capita'
]].fillna(0)

investment_factors_scaled = pd.DataFrame(
    scaler.fit_transform(investment_factors),
    columns=investment_factors.columns,
    index=investment_factors.index
)

latest_df['Investment Score'] = (
    investment_factors_scaled['Access to electricity (% of population)'] * 0.3 +
    investment_factors_scaled['Renewable energy share in the total final energy consumption (%)'] * 0.4 +
    investment_factors_scaled['gdp_per_capita'] * 0.2 +
    investment_factors_scaled['Renewable-electricity-generating-capacity-per-capita'] * 0.1
)

fig1 = px.choropleth(
    latest_df,
    locations="Entity",
    locationmode="country names",
    color="Investment Score",
    hover_name="Entity",
    hover_data=["Renewable energy share in the total final energy consumption (%)", 
                "Access to electricity (% of population)"],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"Renewable Energy Investment Readiness ({latest_year})"
)
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Renewable Energy vs GDP Growth
st.subheader("Renewable Energy Adoption vs Economic Development")
st.markdown("**Size represents CO2 emissions intensity**")

fig2 = px.scatter(
    latest_df.dropna(subset=['gdp_per_capita', 
                            'Renewable energy share in the total final energy consumption (%)']),
    x="gdp_per_capita",
    y="Renewable energy share in the total final energy consumption (%)",
    size="Value_co2_emissions_kt_by_country",
    color="Entity",
    hover_name="Entity",
    log_x=True,
    size_max=60,
    labels={
        "gdp_per_capita": "GDP per Capita (log scale)",
        "Value_co2_emissions_kt_by_country": "CO2 Emissions"
    }
)
st.plotly_chart(fig2, use_container_width=True)

# Visualization 3: Top Countries Analysis
st.subheader("Top Renewable Energy Performers")
st.markdown("**Countries with Highest Renewable Energy Adoption**")

top_countries = latest_df.nlargest(10, 'Renewable energy share in the total final energy consumption (%)')
fig3 = go.Figure(data=[
    go.Bar(
        name='Renewable Energy',
        x=top_countries['Entity'],
        y=top_countries['Renewable energy share in the total final energy consumption (%)'],
        marker_color='#2ca02c'
    ),
    go.Bar(
        name='Electricity Access',
        x=top_countries['Entity'],
        y=top_countries['Access to electricity (% of population)'],
        marker_color='#1f77b4'
    )
])
fig3.update_layout(barmode='group', title="Renewable Energy Leaders")
st.plotly_chart(fig3, use_container_width=True)

# Visualization 4: Trend Analysis
st.subheader("Renewable Energy Adoption Trend (2000-2020)")
selected_countries = st.multiselect(
    "Select countries to compare:",
    df['Entity'].unique(),
    default=["Germany", "Brazil", "India", "Australia"]
)

trend_df = df[df['Entity'].isin(selected_countries)]
fig4 = px.line(
    trend_df,
    x="Year",
    y="Renewable energy share in the total final energy consumption (%)",
    color="Entity",
    markers=True,
    title="Renewable Energy Growth Over Time"
)
st.plotly_chart(fig4, use_container_width=True)

# Investment Recommendation Section
st.divider()
st.subheader("Investment Recommendations")

# Identify promising countries
promising_countries = latest_df[
    (latest_df['Investment Score'] > 0.7) &
    (latest_df['gdp_per_capita'] > 3000) &
    (latest_df['Renewable energy share in the total final energy consumption (%)'] > 20)
].sort_values('Investment Score', ascending=False)

if not promising_countries.empty:
    st.success("**Top Recommended Investment Destinations:**")
    cols = st.columns(3)
    for i, (_, row) in enumerate(promising_countries.head(6).iterrows()):
        with cols[i % 3]:
            st.metric(
                label=row['Entity'],
                value=f"Score: {row['Investment Score']:.2f}",
                help=f"""
                Renewable Share: {row['Renewable energy share in the total final energy consumption (%)']:.1f}%
                Electricity Access: {row['Access to electricity (% of population)']:.1f}%
                GDP per capita: ${row['gdp_per_capita']:,.0f}
                """
            )
else:
    st.warning("No countries meet the investment criteria")

st.divider()
st.caption("Data Source: Global Data on Sustainable Energy (2000-2020)")