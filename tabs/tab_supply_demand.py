import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def display_supply_demand():
    if st.session_state.sku is not None:
        sku = st.session_state.sku
        
        st.header("Supply and Demand Plans")
        
        # Create supply plan DataFrame with proper handling of non-finite values
        try:
            supply_plan = pd.DataFrame({
                'Date': sku.forecast['mean'].index,
                'Forecast Demand': np.nan_to_num(np.round(sku.forecast['mean']), 0).astype(int),  # Replace NaN with 0
                'Order Quantity': np.nan_to_num(np.round(sku.order_evolution), 0).astype(int),
                'Orders Arriving': np.nan_to_num(np.round(sku.orders_arriving), 0).astype(int),
                'Orders in Transit': np.nan_to_num(np.round(sku.orders_in_transit), 0).astype(int),
                'Inventory Level': np.nan_to_num(np.round(sku.inventory_evolution), 0).astype(int),
                'Stockouts': sku.stockouts.astype(int)
            })
            
            # Store the supply plan in session state
            st.session_state.updated_supply_plan = supply_plan
            
            # Display the supply plan
            st.dataframe(supply_plan, use_container_width=True)
            
            # Create download button for supply plan
            csv = supply_plan.to_csv(index=False)
            st.download_button(
                label="Download Supply Plan",
                data=csv,
                file_name="supply_plan.csv",
                mime="text/csv"
            )
            
            # Plot supply and demand with order points and orders in transit
            fig = go.Figure()
            
            # Add Forecast Demand
            fig.add_trace(go.Scatter(
                x=supply_plan['Date'],
                y=supply_plan['Forecast Demand'],
                mode='lines',
                name='Forecast Demand',
                line=dict(dash='dash')
            ))
            
            # Add Inventory Level
            fig.add_trace(go.Scatter(
                x=supply_plan['Date'],
                y=supply_plan['Inventory Level'],
                mode='lines',
                name='Inventory Level'
            ))
            
            # Add Orders in Transit as area
            fig.add_trace(go.Scatter(
                x=supply_plan['Date'],
                y=supply_plan['Orders in Transit'],
                mode='lines',
                name='Orders in Transit',
                fill='tozeroy',
                line=dict(color='rgba(255, 165, 0, 0.3)'),  # Orange with transparency
                fillcolor='rgba(255, 165, 0, 0.1)'
            ))
            
            # Add Order Points as triangles
            order_indices = np.where(supply_plan['Order Quantity'] > 0)[0]
            if len(order_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=supply_plan['Date'].iloc[order_indices],
                    y=supply_plan['Inventory Level'].iloc[order_indices],
                    mode='markers',
                    name='Order Points',
                    marker=dict(
                        color='green',
                        symbol='triangle-up',
                        size=12
                    ),
                    hovertemplate="Order Quantity: %{text}<br>Date: %{x}<br>Inventory Level: %{y}<extra></extra>",
                    text=supply_plan['Order Quantity'].iloc[order_indices]
                ))
            
            # Add Orders Arriving as diamonds
            arriving_indices = np.where(supply_plan['Orders Arriving'] > 0)[0]
            if len(arriving_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=supply_plan['Date'].iloc[arriving_indices],
                    y=supply_plan['Orders Arriving'].iloc[arriving_indices],
                    mode='markers',
                    name='Orders Arriving',
                    marker=dict(
                        color='blue',
                        symbol='diamond',
                        size=8
                    ),
                    hovertemplate="Arriving Quantity: %{y}<br>Date: %{x}<extra></extra>"
                ))
            
            # Add Reorder Point if available
            if isinstance(sku.inventory_policy, dict) and 'reorder_point' in sku.inventory_policy:
                reorder_point = sku.inventory_policy['reorder_point']
                fig.add_trace(go.Scatter(
                    x=supply_plan['Date'],
                    y=[reorder_point] * len(supply_plan),
                    mode='lines',
                    name='Reorder Point',
                    line=dict(dash='dot', color='red')
                ))
            
            fig.update_layout(
                title="Supply and Demand Plan",
                xaxis_title="Date",
                yaxis_title="Quantity",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating supply plan: {str(e)}")
            st.write("Debug information:")
            st.write(f"Forecast shape: {sku.forecast['mean'].shape}")
            st.write(f"Order evolution shape: {sku.order_evolution.shape}")
            st.write(f"Inventory evolution shape: {sku.inventory_evolution.shape}")
            
    else:
        st.warning("Please run the simulation to view supply and demand plans.")