import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from app_utils import plot_inventory_demand
import io

def generate_excel_file(sku, selected_sku, selected_forecast_model, selected_inventory_model, updated_policy, updated_forecast):
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # Sheet 1: Model Information and Parameters
        model_info = pd.DataFrame({
            'SKU': [selected_sku],
            'Forecast Model': [selected_forecast_model],
            'Inventory Model': [selected_inventory_model],
        })
        all_params = {**sku.params, **updated_policy}
        model_info = pd.concat([model_info, pd.DataFrame(all_params, index=[0])], axis=1)
        model_info.to_excel(writer, sheet_name='Model Information', index=False)
        
        # Sheet 2: Updated Supply Plan
        if st.session_state.updated_supply_plan is not None:
            st.session_state.updated_supply_plan.to_excel(writer, sheet_name='Supply Plan', index=False)
        else:
            pd.DataFrame({
                'Date': sku.forecast['mean'].index,
                'Order Quantity': sku.order_evolution,
                'Forecast': updated_forecast,
                'Inventory Level': sku.inventory_evolution,
            }).to_excel(writer, sheet_name='Supply Plan', index=False)
        
        # Sheet 3: Demand Plan
        pd.DataFrame({
            'Date': sku.forecast['mean'].index.strftime('%Y-%m-%d'),
            'Forecasted Demand': updated_forecast
        }).to_excel(writer, sheet_name='Demand Plan', index=False)
    
    excel_buffer.seek(0)
    return excel_buffer

def display_download_results(selected_sku, selected_forecast_model, selected_inventory_model):
    st.header("Download Results")
    if st.session_state.sku is not None:
        sku = st.session_state.sku
        st.subheader("Model Policy Results")
        dates = sku.forecast['mean'].index
        
        if isinstance(sku.inventory_policy, dict):
            policy_df = pd.DataFrame(sku.inventory_policy, index=[0])
        else:
            policy_df = pd.DataFrame({'base_stock_level': [sku.inventory_policy]})
        
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
        st.subheader("Consensus Section")
        st.write("Management can review and adjust the model policy results and forecast here:")
        
        # Initialize session state for updated policy if it doesn't exist
        if 'updated_policy' not in st.session_state:
            st.session_state.updated_policy = sku.inventory_policy.copy() if isinstance(sku.inventory_policy, dict) else {'base_stock_level': sku.inventory_policy}

        updated_policy = {}
        num_columns = len(st.session_state.updated_policy)
        columns = st.columns(num_columns)
        for i, (key, value) in enumerate(st.session_state.updated_policy.items()):
            with columns[i]:
                updated_policy[key] = st.number_input(f"Update {key}", value=float(value), key=f"consensus_{key}")
        
        st.subheader("Update Forecast")
        
        # Initialize the forecast in session state if it doesn't exist
        if 'current_forecast' not in st.session_state:
            st.session_state.current_forecast = sku.forecast['mean'].copy()
        
        # Create transposed forecast dataframe
        forecast_df = pd.DataFrame({
            st.session_state.current_forecast.index[i].strftime('%Y-%m-%d'): [st.session_state.current_forecast.values[i]]
            for i in range(len(st.session_state.current_forecast))
        }, index=['Forecast'])
        
        # Use st.data_editor for the transposed forecast
        edited_forecast = st.data_editor(
            forecast_df,
            use_container_width=True,
            hide_index=False,
            key="forecast_editor"
        )
        
        if st.button("Update Results"):
            with st.spinner("Updating results..."):
                try:
                    # Convert the edited forecast back to a series
                    new_forecast = pd.Series(edited_forecast.iloc[0].values, index=st.session_state.current_forecast.index)
                    
                    # Check if the forecast has actually changed
                    forecast_changed = not new_forecast.equals(st.session_state.current_forecast)
                    policy_changed = updated_policy != sku.inventory_policy
                    
                    if forecast_changed or policy_changed:
                        if forecast_changed:
                            st.session_state.current_forecast = new_forecast
                            sku.forecast['mean'] = new_forecast
                        
                        if policy_changed:
                            sku.update_inventory_policy(updated_policy)
                        
                        # Re-run the simulation
                        sku.simulate_inventory(st.session_state.current_forecast, len(st.session_state.current_forecast))
                        
                        # Update the session state
                        st.session_state.sku = sku

                        st.success("Results updated successfully!")
                    else:
                        st.info("No changes detected. Results remain the same.")

                    # Update the supply plan regardless of changes
                    st.session_state.updated_supply_plan = pd.DataFrame({
                        'Date': dates.strftime('%Y-%m-%d'),
                        'Forecast Demand': sku.forecast['mean'],
                        'Order Quantity': sku.order_evolution,
                        'Orders Arriving': sku.orders_arriving,
                        'Orders in Transit': sku.orders_in_transit,
                        'Inventory Level': sku.inventory_evolution,
                        'Stockouts': sku.stockouts,
                    })
                    
                    # Generate new Excel file
                    st.session_state.excel_file = generate_excel_file(sku, selected_sku, selected_forecast_model, selected_inventory_model, updated_policy, st.session_state.current_forecast)
                
                except Exception as e:
                    st.error(f"An error occurred while updating results: {str(e)}")

        
        # Show the updated metrics:
        if st.session_state.updated_supply_plan is not None:
            df_inventory = st.session_state.updated_supply_plan
            st.subheader("Updated Inventory Metrics")
            stockout_rate = df_inventory['Stockouts'].mean()
            average_inventory = df_inventory['Inventory Level'].mean()
            total_demand = df_inventory['Forecast Demand'].sum()
            inventory_turnover = total_demand / average_inventory if average_inventory > 0 else 0
            
            col1, col2, col3 = st.columns(3)

            col1.metric("Average Inventory", f"{average_inventory:.2f}")
            col2.metric("Stockout Rate", f"{stockout_rate:.2%}")
            col3.metric("Inventory Turnover", f"{inventory_turnover:.2f}")

        
        # Always display the updated inventory level and supply plan
        fig = plot_inventory_demand(sku)
        fig.add_trace(go.Scatter(x=list(range(len(st.session_state.current_forecast))), 
                     y=st.session_state.current_forecast, 
                     mode='lines', 
                     name='Updated Forecast', 
                     line=dict(dash='dot')))
        st.plotly_chart(fig, use_container_width=True,key="plot_inventory_demand_download")
        
        st.subheader("Updated Supply and Demand Plan")
        if st.session_state.updated_supply_plan is not None:
            df_inventory = st.session_state.updated_supply_plan
            df_inventory = df_inventory.set_index('Date').T
            df_inventory = df_inventory.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
            st.dataframe(df_inventory, use_container_width=True)

        st.subheader("Download Updated Results")
        if st.session_state.excel_file is not None:
            st.download_button(
                label="Download Excel file",
                data=st.session_state.excel_file,
                file_name="updated_inventory_management_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Please update the results to generate the Excel file for download.")
    else:
        st.warning("Please run the simulation to generate downloadable results.")