import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
import numpy as np
import pandas as pd
import base64
import streamlit as st

warnings.filterwarnings('ignore')


#LOGO 

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def plot_kpis(df):
    kpi_df = pd.DataFrame({
        'Week': df['Week'],
        'Service Level': 1 - df['Stockout'].rolling(window=4).mean(),
        'Inventory Turnover': df['Demand Fulfilled'].rolling(window=4).sum() / df['Ending Inventory'].rolling(window=4).mean(),
        'Profit': df['Fulfilled Sales Price'] - df['Fulfilled Sales Cost'] - df['Holding Cost'] - df['Inventory Cost'] - df['Stockout Cost']
    })
    
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Service Level', 'Inventory Turnover', 'Profit'))
    fig.add_trace(go.Scatter(x=kpi_df['Week'], y=kpi_df['Service Level'], mode='lines', name='Service Level'), row=1, col=1)
    fig.add_trace(go.Scatter(x=kpi_df['Week'], y=kpi_df['Inventory Turnover'], mode='lines', name='Inventory Turnover'), row=2, col=1)
    fig.add_trace(go.Scatter(x=kpi_df['Week'], y=kpi_df['Profit'], mode='lines', name='Profit'), row=3, col=1)
    fig.update_layout(height=800, title_text='Key Performance Indicators Over Time')
    return fig


def plot_inventory_demand(sku):
    fig = go.Figure()
    
    # Add Inventory Level trace
    fig.add_trace(go.Scatter(x=list(range(len(sku.inventory_evolution))), 
                             y=sku.inventory_evolution, 
                             mode='lines', 
                             name='Inventory Level'))
    
    # Add Demand trace
    fig.add_trace(go.Scatter(x=list(range(len(sku.demand_evolution))), 
                             y=sku.demand_evolution, 
                             mode='lines', 
                             name='Demand', 
                             line=dict(dash='dot')))
    
    # Add Reorder Point trace if applicable
    if 'reorder_point' in sku.inventory_policy:
        reorder_point = sku.inventory_policy['reorder_point']
        fig.add_trace(go.Scatter(x=list(range(len(sku.inventory_evolution))), 
                                 y=[reorder_point]*len(sku.inventory_evolution), 
                                 mode='lines', 
                                 name='Reorder Point', 
                                 line=dict(dash='dash')))
    
    # Add Order Points trace
    order_indices = np.where(sku.order_evolution > 0)[0]
    fig.add_trace(go.Scatter(x=order_indices, 
                             y=sku.inventory_evolution[order_indices], 
                             mode='markers', 
                             name='Order Points', 
                             marker=dict(color='green', symbol='triangle-up', size=10)))
    
    fig.update_layout(title="Inventory Evolution and Demand",
                      xaxis_title="Time Period",
                      yaxis_title="Quantity")
    
    return fig