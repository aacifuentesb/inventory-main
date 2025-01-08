import plotly.graph_objects as go
import streamlit as st


def plot_historical_demand(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Historical Demand'))
    fig.update_layout(title="Historical Demand", xaxis_title="Date", yaxis_title="Quantity")
    return fig

def display_historical_analysis_tab(df,selected_sku):

    st.header("Historical Analysis")
    weekly_data = df[df['SKU'] == selected_sku].set_index('Date')['QTY']
    weekly_data = weekly_data.groupby(weekly_data.index.to_period('W')).sum()
    weekly_data.index = weekly_data.index.to_timestamp()

    st.subheader("Weekly Metrics (units)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"{weekly_data.sum():,.2f}")
    col2.metric("Average Weekly Sales", f"{weekly_data.mean():,.2f}")
    col3.metric("Standard Deviation", f"{weekly_data.std():,.2f}")

    st.plotly_chart(plot_historical_demand(weekly_data), use_container_width=True, key="plot_historical_demand")


    st.subheader("Sales Distribution (units)")
    fig = go.Figure(data=[go.Histogram(x=weekly_data)])
    fig.update_layout(title='Sales Distribution', xaxis_title='Sales', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True, key="plot_sales_distribution")