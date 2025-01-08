import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_forecast(sku):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku.data.index, y=sku.data, mode='lines', name='Historical Demand'))
    fig.add_trace(go.Scatter(x=sku.forecast['mean'].index, y=sku.forecast['mean'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=sku.forecast['lower'].index, y=sku.forecast['lower'], fill=None, mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower CI'))
    fig.add_trace(go.Scatter(x=sku.forecast['upper'].index, y=sku.forecast['upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper CI'))
    fig.update_layout(title="Demand Forecast", xaxis_title="Date", yaxis_title="Quantity (units)")
    return fig

def calculate_forecast_metrics(forecast):
    """Calculate forecast metrics including trend and seasonality"""
    try:
        mean_forecast = forecast['mean']
        lower_ci = forecast['lower']
        upper_ci = forecast['upper']
        
        metrics = {}
        
        # Average Forecast
        metrics['Average Forecast'] = np.mean(mean_forecast)
        
        # Forecast Range
        metrics['Forecast Range'] = np.max(mean_forecast) - np.min(mean_forecast)
        
        # Average Confidence Interval Width
        metrics['Avg CI Width'] = np.mean(upper_ci - lower_ci)
        
        # Coefficient of Variation
        if np.mean(mean_forecast) > 0:
            metrics['Coefficient of Variation'] = np.std(mean_forecast) / np.mean(mean_forecast)
        else:
            metrics['Coefficient of Variation'] = np.nan
        
        # Trend calculation with simpler, more robust method
        try:
            # Calculate simple percentage change
            first_value = mean_forecast.iloc[0]
            last_value = mean_forecast.iloc[-1]
            if first_value > 0:  # Avoid division by zero
                total_change_pct = ((last_value - first_value) / first_value) * 100
                # Annualize the trend (assuming 52 weeks per year)
                metrics['Trend'] = total_change_pct * (52 / len(mean_forecast))
            else:
                metrics['Trend'] = 0
        except Exception as e:
            print(f"Warning: Could not calculate trend: {str(e)}")
            metrics['Trend'] = 0
        
        # Seasonality calculation with error handling
        try:
            if len(mean_forecast) >= 52:
                # Use simple ratio-to-moving-average method
                ma = mean_forecast.rolling(window=52, center=True).mean()
                if ma.mean() > 0:  # Avoid division by zero
                    seasonal_ratios = mean_forecast / ma
                    metrics['Seasonality Strength'] = (seasonal_ratios.max() - seasonal_ratios.min()) * 100
                else:
                    metrics['Seasonality Strength'] = 0
            else:
                metrics['Seasonality Strength'] = 0
        except Exception as e:
            print(f"Warning: Could not calculate seasonality: {str(e)}")
            metrics['Seasonality Strength'] = 0
        
        return metrics
    
    except Exception as e:
        print(f"Error calculating forecast metrics: {str(e)}")
        # Return default metrics if calculation fails
        return {
            'Average Forecast': np.mean(forecast['mean']),
            'Forecast Range': np.max(forecast['mean']) - np.min(forecast['mean']),
            'Avg CI Width': np.mean(forecast['upper'] - forecast['lower']),
            'Coefficient of Variation': 0,
            'Trend': 0,
            'Seasonality Strength': 0
        }

def display_forecast():
    if st.session_state.sku is not None:
        sku = st.session_state.sku
        
        st.header("Forecast Analysis")
        
        # Plot forecast
        fig = plot_forecast(sku)
        st.plotly_chart(fig, use_container_width=True, key="plot_forecast_analysis")
        
        # Calculate and display metrics
        forecast_metrics = calculate_forecast_metrics(sku.forecast)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Forecast", f"{forecast_metrics['Average Forecast']:.2f}")
            st.metric("Forecast Range", f"{forecast_metrics['Forecast Range']:.2f}")
        
        with col2:
            st.metric("Avg CI Width", f"{forecast_metrics['Avg CI Width']:.2f}")
            st.metric("Coefficient of Variation", f"{forecast_metrics['Coefficient of Variation']:.2%}")
        
        with col3:
            st.metric("Trend", f"{forecast_metrics['Trend']:.1f}%")
            st.metric("Seasonality Strength", f"{forecast_metrics['Seasonality Strength']:.1f}%")
        
        # Display forecast data
        st.subheader("Forecast Data")
        forecast_df = pd.DataFrame({
            'Date': sku.forecast['mean'].index,
            'Forecast': sku.forecast['mean'].values,
            'Lower CI': sku.forecast['lower'].values,
            'Upper CI': sku.forecast['upper'].values
        })
        st.dataframe(forecast_df, use_container_width=True)
        
    else:
        st.warning("Please run the simulation to view forecast results.")