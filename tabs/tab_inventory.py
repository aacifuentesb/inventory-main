import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app_utils import plot_inventory_demand

def display_inventory():
    if 'sku' in st.session_state and st.session_state.sku is not None:
        sku = st.session_state.sku
        
        fig = plot_inventory_demand(sku)
        st.plotly_chart(fig, use_container_width=True, key="plot_inventory_demand_inventory_tab")
        
        weeks = len(sku.demand_evolution)
        
        df_inventory = pd.DataFrame({
            'Week': range(1, weeks + 1),
            'Beginning Inventory': np.zeros(weeks),
            'Demand': sku.demand_evolution,
            'Orders Arriving': sku.orders_arriving,
            'Orders Placed': sku.order_evolution,
            'Ending Inventory': np.zeros(weeks),
            'Stockouts': np.zeros(weeks, dtype=bool)
        })
        
        # Calculate beginning and ending inventory correctly, and determine stockouts
        df_inventory.iloc[0, df_inventory.columns.get_loc('Beginning Inventory')] = sku.params['initial_inventory']
        for i in range(weeks):
            if i == 0:
                available_inventory = df_inventory.iloc[i, df_inventory.columns.get_loc('Beginning Inventory')] + df_inventory.iloc[i, df_inventory.columns.get_loc('Orders Arriving')]
            else:
                df_inventory.iloc[i, df_inventory.columns.get_loc('Beginning Inventory')] = df_inventory.iloc[i-1, df_inventory.columns.get_loc('Ending Inventory')]
                available_inventory = df_inventory.iloc[i, df_inventory.columns.get_loc('Beginning Inventory')] + df_inventory.iloc[i, df_inventory.columns.get_loc('Orders Arriving')]
            
            demand = df_inventory.iloc[i, df_inventory.columns.get_loc('Demand')]
            
            if available_inventory < demand:
                df_inventory.iloc[i, df_inventory.columns.get_loc('Stockouts')] = True
                df_inventory.iloc[i, df_inventory.columns.get_loc('Ending Inventory')] = 0
            else:
                df_inventory.iloc[i, df_inventory.columns.get_loc('Ending Inventory')] = available_inventory - demand
        
        # Update metrics
        stockout_rate = df_inventory['Stockouts'].mean()
        average_inventory = df_inventory['Ending Inventory'].mean()
        total_demand = df_inventory['Demand'].sum()
        inventory_turnover = total_demand / average_inventory if average_inventory > 0 else 0
        
        st.subheader("Inventory Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Inventory", f"{average_inventory:.2f}")
        col2.metric("Stockout Rate", f"{stockout_rate:.2%}")
        col3.metric("Inventory Turnover", f"{inventory_turnover:.2f}")
        
        # Inventory Policy Results
        st.subheader("Inventory Policy Results")
        if isinstance(sku.inventory_policy, dict):
            cols = st.columns(len(sku.inventory_policy))
            for i, (key, value) in enumerate(sku.inventory_policy.items()):
                with cols[i]:
                    st.metric(key.replace('_', ' ').title(), f"{value:,.2f}")
        else:
            st.metric("Base Stock Level", f"{sku.inventory_policy:,.2f}")



        st.subheader("Weekly Inventory Analysis")
        selected_week = st.slider("Select Week", 1, weeks, 1)
        
        st.markdown("### Step-by-Step Explanation")
        st.text(generate_weekly_explanation(df_inventory, selected_week - 1))
        
        # Format the dataframe
        for col in ['Beginning Inventory', 'Demand', 'Orders Arriving', 'Orders Placed', 'Ending Inventory']:
            df_inventory[col] = df_inventory[col].round(2)
        df_inventory['Stockouts'] = df_inventory['Stockouts'].astype(int)
        
        st.dataframe(df_inventory.style.format({
            'Beginning Inventory': '{:.2f}',
            'Demand': '{:.2f}',
            'Orders Arriving': '{:.2f}',
            'Orders Placed': '{:.2f}',
            'Ending Inventory': '{:.2f}',
            'Stockouts': '{}'
        }), use_container_width=True)
    else:
        st.warning("Please run the simulation to view inventory analysis results.")


def generate_weekly_explanation(df, week):
    explanation = f"Week {week + 1}:\n"
    explanation += f"- Beginning inventory: {df.iloc[week]['Beginning Inventory']:.2f}\n"
    explanation += f"- Demand for this week: {df.iloc[week]['Demand']:.2f}\n"
    explanation += f"- Orders arriving: {df.iloc[week]['Orders Arriving']:.2f}\n"
    explanation += f"- Orders placed: {df.iloc[week]['Orders Placed']:.2f}\n"
    
    if df.iloc[week]['Stockouts']:
        explanation += f"- Stockout occurred. Unfulfilled demand: {max(df.iloc[week]['Demand'] - df.iloc[week]['Beginning Inventory'] - df.iloc[week]['Orders Arriving'], 0):.2f}\n"
    
    explanation += f"- Ending inventory: {df.iloc[week]['Ending Inventory']:.2f}\n"
    
    return explanation