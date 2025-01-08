import streamlit as st
from optimization import monte_carlo_optimization, calculate_metrics, plot_inventory_and_demand_opt, calculate_total_cost
import copy
import pandas as pd
import numpy as np
from plotly import graph_objects as go

def display_optimization(sku,selected_forecast_model, opt_runned):
    st.header("Parameter Optimization")
    if selected_forecast_model == "Normal Distribution":
        if st.session_state.sku is not None:
            st.write("This tab uses Monte Carlo simulation to optimize the inventory model parameters.")
            
            col1, col2 = st.columns(2)
            with col1:
                num_param_combinations = st.number_input("Number of Parameter Combinations", 
                                                        min_value=1, max_value=10000, value=20, step=5)
            with col2:
                num_demand_scenarios = st.number_input("Number of Demand Scenarios", 
                                                    min_value=10, max_value=100000, value=10, step=5)
            
            col1, col2 = st.columns(2)
            with col1:
                target_service_level = st.slider("Target Service Level", min_value=0.5, max_value=0.99, value=0.95, step=0.01)
            with col2:
                optimization_model = st.radio("Optimization Model", 
                                              [ "Soft Constraint","Hard Constraint"],
                                              help="Hard Constraint: Must meet target service level. Soft Constraint: Penalizes not meeting target.")
            
            objective_function = st.selectbox("Objective Function", 
                                              ["Minimize Total Cost", "Maximize Service Level", "Maximize Profit"],
                                              help="Choose the objective to optimize")
            
            if st.button("Run Parameter Optimization") or opt_runned:
                with st.spinner("Running Monte Carlo optimization..."):
                    original_sku = sku           
                    try:     
                        best_params, best_metric_value, top_10 = monte_carlo_optimization(
                            original_sku, 
                            num_param_combinations, 
                            num_demand_scenarios, 
                            target_service_level, 
                            optimization_model,
                            objective_function
                        )
                    except Exception as e:
                        st.error(f"Error: You may have not run the simulation yet. {e}")
                        raise e
                        return

                    if best_params is None:
                        st.error("Optimization failed. Please try again with different parameters.")
                        return
                    
                    # Create optimized SKU
                    optimized_sku = copy.deepcopy(original_sku)
                    optimized_sku.params.update(best_params)

                    # Update inventory policy with only the keys that exist in the model's policy
                    optimized_sku.inventory_policy = optimized_sku.inventory_model.calculate(optimized_sku.data.values, optimized_sku.params)
                    for key in optimized_sku.inventory_model.policy.keys():
                        if key in best_params:
                            optimized_sku.inventory_model.policy[key] = best_params[key]
                            optimized_sku.inventory_policy[key] = best_params[key]

                    optimized_sku.simulate_inventory(optimized_sku.forecast['mean'], len(optimized_sku.forecast['mean']))
                    
                    st.session_state.optimized_sku = optimized_sku
                
                st.success("Optimization complete!")
                
                # Display metrics
                st.subheader("Performance Comparison")
                metrics = calculate_metrics(original_sku, optimized_sku)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Model")
                    st.write(original_sku.inventory_policy)
                    subcol1, subcol2 = st.columns(2)
                    even = 0
                    for metric, value in metrics['original'].items():
                        if even == 0:
                            with subcol1:
                                st.metric(label=metric, 
                                          value=f"{value:.2f}" if 'Service Level' in metric else f"{value:,.2f}")
                            even = 1
                        else:
                            with subcol2:
                                st.metric(label=metric, 
                                          value=f"{value:.2f}" if 'Service Level' in metric else f"{value:,.2f}")
                            even = 0
                        

                with col2:
                    st.subheader("Optimized Model")
                    st.write(optimized_sku.inventory_policy)

                    even = 0
                    subcol1, subcol2 = st.columns(2)
                    for metric, value in metrics['optimized'].items():
                        if even == 0:
                            with subcol1:
                                st.metric(label=metric, 
                                          value=f"{value:.2f}" if 'Service Level' in metric else f"{value:,.2f}")
                            even = 1
                        else:
                            with subcol2:
                                st.metric(label=metric, 
                                          value=f"{value:.2f}" if 'Service Level' in metric else f"{value:,.2f}")
                            even = 0
                
                # Plot original inventory and demand
                st.subheader("Original Inventory and Demand")
                fig_original = plot_inventory_and_demand_opt(original_sku)
                st.plotly_chart(fig_original, use_container_width=True,key="fig_original")
                
                # Plot optimized inventory and demand
                st.subheader("Optimized Inventory and Demand")
                fig_optimized = plot_inventory_and_demand_opt(optimized_sku)
                st.plotly_chart(fig_optimized, use_container_width=True,key="fig_optimized")

                # Give the top 10 parameter combinations
                if top_10 is not None:
                    # Remove from top 10 every key that has as value a None
                    top_10 = {k: v for k, v in top_10.items() if v is not None}
                    # Show only if there are values
                    if len(top_10) > 0:
                        st.subheader("Top 10 Parameter Combinations")
                        min_value = list(top_10.keys())[0]
                        columns = top_10[min_value].keys()
                        top_10 = pd.DataFrame(top_10).T
                        top_10.columns = columns
                        # Make the index a new column called "Value"
                        top_10 = top_10.reset_index()
                        if objective_function == "Minimize Total Cost":
                            value = "Average total Cost ($)"
                        elif objective_function == "Maximize Service Level":
                            value = "Average Service Level"
                        elif objective_function == "Maximize Profit":
                            value = "Average Profit ($)"
                        top_10 = top_10.rename(columns={"index": value})
                        # Sort the values by the "Value" column
                        top_10 = top_10.sort_values(by=value)
                        st.dataframe(top_10,use_container_width=True,hide_index=True)

    else:
        st.warning("Parameter optimization is only available for the Normal Distribution forecast model.")
       
def compare_performance(original_sku, optimized_sku, metric):
    st.write("Original SKU parameters:", original_sku.params)
    st.write("Optimized SKU parameters:", optimized_sku.params)
    st.write("Original inventory policy:", original_sku.inventory_policy)
    st.write("Optimized inventory policy:", optimized_sku.inventory_policy)
    
    if metric == "Minimize Total Cost":
        original_cost = calculate_total_cost(original_sku)
        optimized_cost = calculate_total_cost(optimized_sku)
        st.write(f"Original total cost: ${original_cost:,.2f}")
        st.write(f"Optimized total cost: ${optimized_cost:,.2f}")
        st.write(f"Cost reduction: {(original_cost - optimized_cost) / original_cost:,.2%}")
    elif metric == "Maximize Service Level":
        original_sl = 1 - np.mean(original_sku.stockouts)
        optimized_sl = 1 - np.mean(optimized_sku.stockouts)
        st.write(f"Original service level: {original_sl:,.2%}")
        st.write(f"Optimized service level: {optimized_sl:,.2%}")
        st.write(f"Service level improvement: {optimized_sl - original_sl:,.2%}")
    elif metric == "Maximize Inventory Turnover":
        original_turnover = calculate_inventory_turnover(original_sku)
        optimized_turnover = calculate_inventory_turnover(optimized_sku)
        st.write(f"Original inventory turnover: {original_turnover:,.2f}")
        st.write(f"Optimized inventory turnover: {optimized_turnover:,.2f}")
        st.write(f"Turnover improvement: {(optimized_turnover - original_turnover) / original_turnover:,.2%}")

def calculate_inventory_turnover(sku):
    return np.sum(sku.demand_evolution) / np.mean(sku.inventory_evolution)

def plot_inventory_comparison(original_sku, optimized_sku):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=original_sku.inventory_evolution, mode='lines', name='Original Inventory'))
    fig.add_trace(go.Scatter(y=optimized_sku.inventory_evolution, mode='lines', name='Optimized Inventory'))
    fig.update_layout(title='Inventory Level Comparison', xaxis_title='Time', yaxis_title='Inventory Level')
    return fig

