import streamlit as st
import pandas as pd
import numpy as np
from forecast import (
    ExponentialSmoothingForecast, 
    NormalDistributionForecast, 
    ARIMAForecast,
    HoltWintersAdvancedForecast,
    SeasonalARIMAForecast,
    EnsembleForecast,
    MovingAverageTrendForecast,
    SimpleExpSmoothingDrift,
    CrostonForecast
)
from inventory import SSPeriodicReview, RQContinuousReview, BaseStockModel, NewsvendorModel,ModifiedContinuousReview
from sku import *
from app_utils import *
import plotly.graph_objects as go
import warnings

# Import the other tabs
from tabs.tab_inventory import display_inventory
from tabs.tab_supply_demand import display_supply_demand
from tabs.tab_download import display_download_results
from tabs.tab_historical import display_historical_analysis_tab
from tabs.tab_forecast import display_forecast
from tabs.tab_management import display_management
from tabs.tab_montecarlo_sim import display_montecarlo_sim
from tabs.tab_optimization import display_optimization

st.set_page_config(layout="wide", page_title="Single SKU Analysis", page_icon="📦")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.start_run = False
    st.session_state.simulation_run = False
    st.session_state.sensitivity_results = None
    st.session_state.sku = None
    st.session_state.updated_supply_plan = None
    st.session_state.excel_file = None
    st.session_state.optimization_run = False
    st.session_state.selected_sku = None
    st.session_state.params = None

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['SKU', 'Date'])
    return df

def run_simulation(df, selected_sku, selected_forecast_model, selected_inventory_model, periods, start_time, params, forecast_models, inventory_models):
    with st.spinner("Running simulation"):
        try:
            # If Best Model is selected, evaluate models first
            if selected_forecast_model == "Best Model (Auto-select)":
                with st.spinner("Evaluating forecast models..."):
                    try:
                        sku_data = df[df['SKU'] == selected_sku]
                        data = generate_weekly_time_series(sku_data, selected_sku)
                        
                        # Remove the "Best Model" option from the models to evaluate
                        models_to_evaluate = {k: v for k, v in forecast_models.items() if v is not None}
                        best_model = select_best_model(data, params['periods'], models_to_evaluate)
                        forecast_models["Best Model (Auto-select)"] = best_model
                    except Exception as e:
                        st.error(f"Error selecting best model: {str(e)}")
                        st.error("Falling back to Normal Distribution")
                        selected_forecast_model = "Normal Distribution"
            
            # Run the simulation with selected or best model
            sku = run_inventory_system(
                df, selected_sku, params, 
                forecast_models[selected_forecast_model], 
                inventory_models[selected_inventory_model], 
                periods, start_time
            )
            st.session_state.sku = sku
            st.session_state.simulation_run = True
            
        except Exception as e:
            st.error(f"An error occurred during the simulation: {str(e)}")
            raise e

def display_logo_and_title():
    logo_base64 = get_base64_of_bin_file("logo.svg")
    st.markdown(
    f"""
    <style>
    .logo-title {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        position: relative;
    }}
    .logo {{
        height: 4rem;
        position: absolute;
        left: 0;
    }}
    .title {{
        color: #262730;
        font-size: 2rem;
        margin: 0;
        text-align: center;
        flex: 1;
    }}
    </style>
    <div class="logo-title">
        <img src="data:image/svg+xml;base64,{logo_base64}" alt="Company Logo" class="logo">
        <h1 class="title">📦 Supply Chain Management</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

def display_input_section(df):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("File uploaded successfully!")
            st.session_state.df = df
    
    with col2:
        if df is not None:
            skus = df['SKU'].unique()
            selected_sku = st.selectbox("Select SKU", skus)
            st.session_state.selected_sku = selected_sku
            if st.button("Run Simulation", key="run_simulation"):
                st.session_state.start_run = True
        else:
            st.info("Please upload an Excel file to begin.")
    
    return df, selected_sku if df is not None else None

def select_best_model(data, periods, forecast_models):
    """Compare all models and select the best one based on multiple metrics"""
    print("\nComparing forecast models...")
    model_scores = {}
    
    # Split data for validation
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Save original dates
    train_dates = train_data.index
    test_dates = test_data.index
    
    # Convert to array and handle zeros
    train_values = train_data.values
    test_values = test_data.values
    
    if (train_values == 0).any():
        mean_value = train_values[train_values > 0].mean()
        print(f"Found zeros in training data. Temporarily replacing with mean ({mean_value:.2f}) for model evaluation")
        train_values = np.where(train_values == 0, mean_value, train_values)
    
    for name, model in forecast_models.items():
        if name == "Best Model (Auto-select)":
            continue
            
        try:
            print(f"\nEvaluating {name}...")
            # Create temporary series with proper dates for model
            temp_train = pd.Series(train_values, index=train_dates)
            
            # Fit and forecast on validation period
            forecast = model.forecast(temp_train, len(test_data))
            
            # Extract forecast values and ensure alignment
            forecast_values = forecast['mean'].values
            lower_values = forecast['lower'].values
            upper_values = forecast['upper'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(test_values - forecast_values))
            rmse = np.sqrt(np.mean((test_values - forecast_values)**2))
            
            # Calculate MAPE only for non-zero actual values
            non_zero_mask = test_values != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((test_values[non_zero_mask] - forecast_values[non_zero_mask]) / 
                                    test_values[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            # Calculate stability (lower variation is better)
            stability = np.std(np.diff(forecast_values))
            
            # Calculate coverage
            coverage = np.mean((test_values >= lower_values) & 
                             (test_values <= upper_values)) * 100
            
            # Combine metrics into a single score (lower is better)
            score = (
                0.3 * (mae / np.mean(test_values)) +  # Normalized MAE
                0.3 * (rmse / np.mean(test_values)) +  # Normalized RMSE
                0.2 * (mape / 100 if not np.isnan(mape) else 1.0) +  # Normalized MAPE
                0.1 * (stability / np.mean(test_values)) +  # Normalized stability
                0.1 * (1 - coverage/100)  # Coverage error
            )
            
            model_scores[name] = {
                'model': model,
                'score': score,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'stability': stability,
                'coverage': coverage
            }
            
            print(f"{name} metrics:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}%" if not np.isnan(mape) else "MAPE: N/A")
            print(f"Stability: {stability:.2f}")
            print(f"Coverage: {coverage:.2f}%")
            print(f"Overall score: {score:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    
    if not model_scores:
        raise ValueError("No models could be evaluated successfully")
    
    # Select best model
    best_model_name = min(model_scores.items(), key=lambda x: x[1]['score'])[0]
    best_model = model_scores[best_model_name]['model']
    
    # Display comparison results
    st.subheader("Model Comparison Results")
    comparison_df = pd.DataFrame({
        name: {
            'MAE': f"{scores['mae']:.2f}",
            'RMSE': f"{scores['rmse']:.2f}",
            'MAPE': f"{scores['mape']:.2f}%" if not np.isnan(scores['mape']) else "N/A",
            'Stability': f"{scores['stability']:.2f}",
            'Coverage': f"{scores['coverage']:.2f}%",
            'Overall Score': f"{scores['score']:.4f}"
        }
        for name, scores in model_scores.items()
    }).T
    
    st.dataframe(comparison_df)
    st.success(f"Selected Best Model: {best_model_name}")
    
    return best_model

def display_model_selection():
    forecast_models = {
        "Best Model (Auto-select)": None,
        "Normal Distribution": NormalDistributionForecast(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        ),
        "Moving Average Trend": MovingAverageTrendForecast(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        ),
        "Simple Exp Smoothing": SimpleExpSmoothingDrift(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        ),
        "Seasonal ARIMA": SeasonalARIMAForecast(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        ),
        "Croston": CrostonForecast(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        ),
        "Ensemble": EnsembleForecast(
            zero_demand_strategy='rolling_mean',
            rolling_window=4,
            zero_train_strategy='mean'
        )
    }
    
    inventory_models = {
        "RQ Continuous Review": RQContinuousReview(),
        "SS Periodic Review": SSPeriodicReview(),
        "Base Stock Model": BaseStockModel(),
        "Newsvendor Model": NewsvendorModel(),
        "Modified Continuous Review": ModifiedContinuousReview()
    }

    col1, col2 = st.columns(2)
    with col1:
        selected_forecast_model = st.selectbox(
            "Select Forecast Model", 
            list(forecast_models.keys()),
            help="""
            - Best Model: Automatically selects the best performing model
            - Normal Distribution: Simple statistical forecasting
            - Moving Average Trend: Simple trend-based forecasting
            - Simple Exp Smoothing: Basic exponential smoothing with drift
            - Seasonal ARIMA: Advanced time series with seasonality
            - Croston: Intermittent demand forecasting
            - Ensemble: Combines multiple forecasting methods
            """
        )
    
    with col2:
        selected_inventory_model = st.selectbox(
            "Select Inventory Model", 
            list(inventory_models.keys()),
            index=len(inventory_models)-1
        )
    
    return selected_forecast_model, selected_inventory_model, forecast_models, inventory_models

def display_parameters(df):
    st.subheader("Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        periods = st.number_input("Number of Periods", min_value=1, value=52)
        lead_time_weeks = st.number_input("Lead Time (weeks)", min_value=1, value=1)
        service_level = st.number_input("Service Level", min_value=0.1, max_value=0.999, value=0.95, step=0.01)
        review_period = st.number_input("Review Period (weeks)", min_value=1, value=4, help="0 means continuous review, any positive integer means periodic review every n weeks")
    with col2:
        start_time = st.date_input("Start Time", min_value=df['Date'].min(), max_value=df['Date'].max(), value=df['Date'].min())
        order_cost = st.number_input("Order Cost", min_value=0.0, value=0.0, help="Cost of placing an order")
        holding_cost = st.number_input("Holding Cost (per unit per week)", min_value=0.0, value=0.1, help="Cost of holding one unit of inventory per week")
        #stockout_cost = st.number_input("Stockout Cost (per unit)", min_value=0, value=10)
    with col3:
        initial_inventory = st.number_input("Initial Inventory", min_value=0, value=100)
        cost = st.number_input("Product Cost", min_value=0.0, value=50.0, help="Cost of purchasing the product")
        price = st.number_input("Selling Price", min_value=0.0, value=100.0, help="Price at which the product is sold to customers")

    params = {
        'periods': periods,
        'lead_time_weeks': lead_time_weeks,
        'initial_inventory': initial_inventory,
        'service_level': service_level,
        'start_time': start_time,
        'order_cost': order_cost,
        'holding_cost': holding_cost,
        #'stockout_cost': stockout_cost,
        'cost': cost,
        'price': price,
        'review_period': review_period
    }
    return params, start_time

def display_results_tabs(df, selected_sku, selected_forecast_model, selected_inventory_model, params):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Historical Analysis", "Forecast", "Inventory Analysis", "Management Dashboard", 
        "Monte Carlo Simulations", "Supply and Demand Plans",
        "Parameter Optimization", "Download Results", "Information"
    ])
    

    with tab1:
        display_historical_analysis_tab(df, selected_sku)
    with tab2:
        display_forecast()
    with tab3:
        if st.session_state.simulation_run:
            display_inventory()
        else:
            st.warning("Please run the simulation to view inventory analysis results.")
    with tab4:
        display_management()
    with tab5:
        display_montecarlo_sim(
            selected_sku, selected_forecast_model, params, 
            st.session_state.forecast_models, st.session_state.inventory_models, 
            df, params['periods'], params['start_time'], selected_inventory_model, 
            params['initial_inventory']
        )
    with tab6:
        display_supply_demand()
    with tab7:
        display_optimization(st.session_state["sku"],selected_forecast_model, st.session_state.optimization_run)
    with tab8:
        display_download_results(selected_sku, selected_forecast_model, selected_inventory_model)
    with tab9:
        display_model_information(st.session_state.forecast_models, st.session_state.inventory_models)


def display_model_information(forecast_models, inventory_models):
    st.header("Model Information")
    st.subheader("Forecast Models")
    for model_name, model in forecast_models.items():
        st.markdown(f"**{model_name}**")
        if model_name == "Best Model (Auto-select)":
            st.markdown("""
            Automatically evaluates all available models and selects the best one based on:
            - MAE (Mean Absolute Error)
            - RMSE (Root Mean Square Error)
            - Forecast Stability
            - Prediction Interval Coverage
            
            The model with the best overall score is selected for forecasting.
            """)
        elif model is not None:
            st.markdown(model.get_description())
        st.markdown("---")

    st.subheader("Inventory Models")
    for model_name, model in inventory_models.items():
        st.markdown(f"**{model_name}**")
        st.markdown(model.get_description())
        st.markdown("---")

def main():
    display_logo_and_title()

    df, selected_sku = display_input_section(st.session_state.get('df'))

    if df is not None and selected_sku:
        selected_forecast_model, selected_inventory_model, forecast_models, inventory_models = display_model_selection()
        st.session_state.forecast_models = forecast_models
        st.session_state.inventory_models = inventory_models

        params, start_time = display_parameters(df)
        
        if st.session_state.start_run:
            run_simulation(df, selected_sku, selected_forecast_model, selected_inventory_model, 
                           params['periods'], start_time, params, forecast_models, inventory_models)
            st.session_state.start_run = False
        display_results_tabs(df, selected_sku, selected_forecast_model, selected_inventory_model, params)
        

if __name__ == "__main__":
    main()