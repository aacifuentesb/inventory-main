from sku import run_inventory_system
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from inventory import RQContinuousReview, SSPeriodicReview, BaseStockModel, NewsvendorModel, ModifiedContinuousReview


def generate_excel_download(summary_results, detailed_results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_results.to_excel(writer, sheet_name='Summary Results', index=False)
        for i, df in enumerate(detailed_results):
            df.to_excel(writer, sheet_name=f'Simulation {i+1}', index=False)
    processed_data = output.getvalue()
    return processed_data

def plot_sensitivity_results(results):
    fig = make_subplots(rows=3, cols=2, subplot_titles=(
        "Service Level Distribution", "Margin Distribution", 
        "Average Inventory vs Service Level", "Margin vs Service Level",
        "Sales Cost vs Sales Price", "Inventory Cost vs Holding Cost"
    ))
    
    fig.add_trace(go.Histogram(x=results['Service Level'], name="Service Level"), row=1, col=1)
    fig.add_trace(go.Histogram(x=results['Margin'], name="Margin"), row=1, col=2)
    fig.add_trace(go.Scatter(x=results['Service Level'], y=results['Weeks of Inventory'], 
                             mode='markers', name="Avg Inventory vs Service Level"), row=2, col=1)
    fig.add_trace(go.Scatter(x=results['Service Level'], y=results['Margin'], 
                             mode='markers', name="Margin vs Service Level"), row=2, col=2)
    fig.add_trace(go.Scatter(x=results['Fulfilled Sales Cost'], y=results['Fulfilled Sales Price'], 
                             mode='markers', name="Sales Cost vs Sales Price"), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['Inventory Cost'], y=results['Holding Cost'], 
                             mode='markers', name="Inventory Cost vs Holding Cost"), row=3, col=2)
    
    fig.update_layout(height=1200, title_text="Sensitivity Analysis Results")
    return fig

def run_sensitivity_analysis(df, selected_sku, params, forecast_model, inventory_model, periods, start_time, num_simulations, initial_inventory):
    detailed_results = []
    summary_results = []

    for run in range(num_simulations):
        # Get base SKU setup
        sku = run_inventory_system(df, selected_sku, params, forecast_model, inventory_model, periods, start_time)
        
        # Generate a new demand scenario for each run
        demand_scenario = forecast_model.forecast(sku.data, periods)['mean'].values
        
        # Use the model's own simulation logic
        inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand = \
            inventory_model.simulate(demand_scenario, periods, params)
        
        # Calculate metrics for this run
        holding_cost = inventory * params['holding_cost']
        stockout_cost = unfufilled_demand * params['stockout_cost']
        inventory_cost = orders * params['cost']
        fulfilled_sales_cost = np.minimum(inventory, demand_scenario) * params['cost']
        fulfilled_sales_price = np.minimum(inventory, demand_scenario) * params['price']
        
        # Generate detailed results for each week of this run
        detailed_df = pd.DataFrame({
            'Week': range(1, periods + 1),
            'Beginning Inventory': np.round(inventory).astype(int),
            'Received Inventory': np.round(orders_arriving).astype(int),
            'Available Inventory': np.round(inventory + orders_arriving).astype(int),
            'Demand': np.round(demand_scenario).astype(int),
            'Fulfilled Demand': np.round(np.minimum(inventory, demand_scenario)).astype(int),
            'Ending Inventory': np.round(np.roll(inventory, -1)).astype(int),
            'Stockout': stockouts,
            'Order': np.round(orders).astype(int),
            'Orders in Transit': np.round(orders_in_transit).astype(int),
            'Reorder (1/0)': (orders > 0).astype(int),
            'Lead Time': [params['lead_time_weeks']] * periods,
            'Arriving on Day': np.where(orders > 0, np.arange(periods) + params['lead_time_weeks'], 0),
            'Stockout Cost': np.round(stockout_cost, 2),
            'Holding Cost': np.round(holding_cost, 2),
            'Inventory Cost': np.round(inventory_cost, 2),
            'Fulfilled Sales Cost': np.round(fulfilled_sales_cost, 2),
            'Fulfilled Sales Price': np.round(fulfilled_sales_price, 2),
            'Weeks of Inventory': np.round(inventory / np.mean(demand_scenario), 2)
        })
        detailed_results.append(detailed_df)

        # Calculate summary results for this run
        summary_results.append({
            'Run': run + 1,
            'Holding Cost': np.sum(holding_cost),
            'Inventory Cost': np.sum(inventory_cost),
            'Fulfilled Sales Cost': np.sum(fulfilled_sales_cost),
            'Fulfilled Sales Price': np.sum(fulfilled_sales_price),
            'Weeks of Inventory': np.mean(inventory / np.mean(demand_scenario)),
            'Service Level': 1 - np.mean(stockouts),
            'Margin': (np.sum(fulfilled_sales_price) - np.sum(fulfilled_sales_cost)) / np.sum(fulfilled_sales_price)
        })

    return pd.DataFrame(summary_results), detailed_results

def display_sensitivity_analysis(sku):
    st.header("Sensitivity Analysis")
    
    num_simulations = st.number_input("Number of Simulations", min_value=1, max_value=100, value=20)
    periods = st.number_input("Number of Periods", min_value=1, max_value=52, value=13)

    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            results_df, detailed_results = run_sensitivity_analysis(sku, num_simulations, periods)

        st.subheader("Sensitivity Analysis Results")
        st.dataframe(results_df)

        avg_results = results_df.mean()
        st.subheader("Average Results")
        st.write(f"Average Holding Cost: ${avg_results['Holding Cost']:,.2f}")
        st.write(f"Average Inventory Cost: ${avg_results['Inv Cost']:,.2f}")
        st.write(f"Average Sales Cost: ${avg_results['$ Ventas Llenadas Costo']:,.2f}")
        st.write(f"Average Sales Price: ${avg_results['$ Ventas Llenadas Precio']:,.2f}")
        st.write(f"Average Weeks of Inventory: {avg_results['Semanas de inv']:,.2f}")
        st.write(f"Average Service Level: {avg_results['Service Level']:,.2%}")
        st.write(f"Average Margin: {avg_results['Margen']:,.2%}")

        st.subheader("Detailed Simulation Results")
        selected_run = st.selectbox("Select a run to view detailed results:", range(1, num_simulations + 1))
        st.dataframe(detailed_results[selected_run - 1])

        # Visualizations
        st.subheader("Visualizations")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['Run'], y=results_df['Service Level'], mode='lines+markers', name='Service Level'))
        fig.add_trace(go.Scatter(x=results_df['Run'], y=results_df['Margen'], mode='lines+markers', name='Margin'))
        fig.update_layout(title='Service Level and Margin by Run', xaxis_title='Run', yaxis_title='Value')
        st.plotly_chart(fig,key = "plot_sensitivity_results")

def display_montecarlo_sim(selected_sku,selected_forecast_model, params, forecast_models, inventory_models, df, periods, start_time,selected_inventory_model, initial_inventory):
    st.header("Sensitivity Analysis")
    if selected_forecast_model == "Normal Distribution":
        num_simulations = st.number_input("Number of Simulations", min_value=1, max_value=10000, value=10)
        
        if st.button("Run Sensitivity Analysis"):
            # Get session state values before creating thread
            current_sku = st.session_state.sku
            current_forecast_model = forecast_models[selected_forecast_model]
            current_inventory_model = inventory_models[selected_inventory_model]
            
            with st.spinner("Running sensitivity analysis..."):
                summary_results, detailed_results = run_sensitivity_analysis(
                    df, selected_sku, params, 
                    current_forecast_model,
                    current_inventory_model,
                    periods, start_time, 
                    num_simulations, initial_inventory
                )
                st.session_state.summary_results = summary_results
                st.session_state.detailed_results = detailed_results

        if 'summary_results' in st.session_state:
            st.subheader("Summary Results")
            avg_results = st.session_state.summary_results.mean()
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            col1.metric("Holding Cost", f"${avg_results['Holding Cost']:,.2f}")
            col2.metric("Inventory Cost", f"${avg_results['Inventory Cost']:,.2f}")
            col3.metric("Fulfilled Sales Cost", f"${avg_results['Fulfilled Sales Cost']:,.2f}")
            col4.metric("Fulfilled Sales Price", f"${avg_results['Fulfilled Sales Price']:,.2f}")
            col5.metric("Weeks of Inventory", f"{avg_results['Weeks of Inventory']:.2f}")
            col6.metric("Service Level", f"{avg_results['Service Level']:,.2%}")
            col7.metric("Margin", f"{avg_results['Margin']:,.2%}")

            # Display summary results below metrics
            st.subheader("Summary Results")
            st.dataframe(st.session_state.summary_results, use_container_width=True)

            st.subheader("Detailed Simulation Results")
            selected_run = st.selectbox("Select a run to view detailed results:", range(1, num_simulations + 1))
            st.dataframe(st.session_state.detailed_results[selected_run - 1], use_container_width=True)

            # Option to download all results
            excel_data = generate_excel_download(st.session_state.summary_results, st.session_state.detailed_results)
            st.download_button(
                label="Download All Results",
                data=excel_data,
                file_name="sensitivity_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if num_simulations <= 100:
                # Existing visualizations and scenario comparisons
                st.plotly_chart(plot_sensitivity_results(st.session_state.summary_results), use_container_width=True,key= "plot_sensitivity_results_sim_output")

    else:
        st.warning("Sensitivity analysis is only available for the Normal Distribution forecast model.")
