import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd

def plot_costs_over_time(sku):
    holding_costs = sku.inventory_evolution * sku.params['holding_cost']
    ordering_costs = (sku.order_evolution > 0) * sku.params['order_cost']
    stockout_costs = sku.unfufilled_demand * sku.params['stockout_cost']
    inventory_costs = sku.order_evolution * sku.params['cost']
    sales_revenue = np.minimum(sku.inventory_evolution, sku.demand_evolution) * sku.params['price']
    profits = sales_revenue - holding_costs - ordering_costs - inventory_costs

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=("Costs Over Time", "Revenue and Profit Over Time"))

    fig.add_trace(go.Scatter(y=holding_costs, mode='lines', name='Holding Cost'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ordering_costs, mode='lines', name='Ordering Cost'), row=1, col=1)
    fig.add_trace(go.Scatter(y=stockout_costs, mode='lines', name='Stockout Cost'), row=1, col=1)
    fig.add_trace(go.Scatter(y=inventory_costs, mode='lines', name='Inventory Cost'), row=1, col=1)

    fig.add_trace(go.Scatter(y=sales_revenue, mode='lines', name='Sales Revenue'), row=2, col=1)
    fig.add_trace(go.Scatter(y=profits, mode='lines', name='Profit'), row=2, col=1)

    fig.update_layout(height=800, title_text="Cost, Revenue, and Profit Evolution")
    fig.update_xaxes(title_text="Time Period", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=2, col=1)

    return fig

def plot_profit(sku):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(sku.profit_evolution))), y=sku.profit_evolution, mode='lines', name='Profit'))
    fig.add_trace(go.Scatter(x=list(range(len(sku.profit_evolution))), y=np.cumsum(sku.profit_evolution), mode='lines', name='Cumulative Profit'))
    fig.update_layout(title="Profit Evolution", xaxis_title="Time Period", yaxis_title="Profit")
    return fig

def display_management():
    if st.session_state.sku is not None:
        sku = st.session_state.sku
        st.header("Management Dashboard")
        
        # Cost, Revenue, and Profit Evolution
        st.plotly_chart(plot_costs_over_time(sku), use_container_width=True, key="plot_costs_over_time")
        
        # Key Performance Indicators
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Service Level", f"{sku.metrics['overall']['service_level']:,.2%}")
        col2.metric("Total Profit", f"${sku.metrics['overall']['total_profit']:,.2f}")
        col3.metric("Average weekly Profit", f"${sku.metrics['overall']['average_profit']:,.2f}")
        col4.metric("Total Orders", f"{sku.metrics['overall']['total_orders']}")
        
        # Detailed Cost and Revenue Breakdown
        st.subheader("Cost and Revenue Breakdown")
        
        # Calculate detailed metrics
        total_holding_cost = np.sum(sku.inventory_evolution * sku.params['holding_cost'])
        total_ordering_cost = np.sum(sku.order_evolution > 0) * sku.params['order_cost']
        total_inventory_cost = np.sum(sku.order_evolution * sku.params['cost'])
        total_sales_revenue = np.sum(np.minimum(sku.inventory_evolution, sku.demand_evolution) * sku.params['price'])
        total_profit = total_sales_revenue - total_holding_cost - total_ordering_cost - total_inventory_cost
        
        # Create a DataFrame for the breakdown
        breakdown_df = pd.DataFrame({
            'Category': ['Holding Cost', 'Ordering Cost', 'Inventory Cost', 'Sales Revenue', 'Total Profit'],
            'Amount': [total_holding_cost, total_ordering_cost, total_inventory_cost, total_sales_revenue, total_profit]
        })
        
        # Display the breakdown as a table
        st.table(breakdown_df.style.format({'Amount': '${:,.2f}'}))
        
        # Display the breakdown as a bar chart
        fig = go.Figure(data=[go.Bar(x=breakdown_df['Category'], y=breakdown_df['Amount'])])
        fig.update_layout(title="Cost and Revenue Breakdown", xaxis_title="Category", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True, key="plot_cost_revenue_breakdown_management")
        
        # Inventory Policy Results
        st.subheader("Inventory Policy Results")
        if isinstance(sku.inventory_policy, dict):
            cols = st.columns(len(sku.inventory_policy))
            for i, (key, value) in enumerate(sku.inventory_policy.items()):
                with cols[i]:
                    st.metric(key.replace('_', ' ').title(), f"{value:,.2f}")

            if 'eoq' in sku.inventory_policy:
                st.write(f"The Economic Order Quantity (EOQ) is {sku.inventory_policy['eoq']:,.2f}. This is the optimal order quantity that minimizes total inventory holding costs and ordering costs.")
            if 'reorder_point' in sku.inventory_policy:
                st.write(f"The Reorder Point is {sku.inventory_policy['reorder_point']:,.2f}. When the inventory level reaches this point, a new order should be placed.")
        else:
            st.metric("Base Stock Level", f"{sku.inventory_policy:,.2f}")
            st.write(f"The Base Stock Level of {sku.inventory_policy:,.2f} represents the target inventory level. The system will always try to bring the inventory position up to this level after each demand occurrence.")
        
        # Additional Metrics
        st.subheader("Additional Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Inventory", f"{sku.metrics['overall']['average_inventory']:,.2f}")
        col2.metric("Inventory Turnover", f"{sku.metrics['overall']['inventory_turnover']:,.2f}")
        col3.metric("Average Order Size", f"{sku.metrics['overall']['average_order_size']:,.2f}")
        
        # Profit Analysis
        st.subheader("Profit Analysis")
        st.plotly_chart(plot_profit(sku), use_container_width=True, key="plot_profit_analysis")
        
    else:
        st.warning("Please run the simulation to view management dashboard results.")