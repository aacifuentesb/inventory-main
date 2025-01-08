# Inventory Management System

## Overview

This Inventory Management System is a Streamlit-based web application designed to help businesses optimize their inventory control processes. It provides powerful tools for demand forecasting, inventory policy calculation, and performance analysis.

## Features

- **Data Upload**: Easy Excel file upload for SKU data
- **Demand Forecasting**: Multiple forecasting models including Exponential Smoothing, ARIMA, and Normal Distribution
- **Inventory Policy Calculation**: Various inventory models including Periodic Review, Continuous Review, Base Stock, and Newsvendor
- **Interactive Visualizations**: Detailed plots for inventory levels, demand, and various performance metrics
- **Sensitivity Analysis**: Tools to understand how different parameters affect inventory performance
- **Performance Metrics**: Comprehensive set of KPIs including service level, profit, and inventory turnover
- **Supply and Demand Planning**: Visualization and analysis of supply plans and demand forecasts
- **Customizable Parameters**: Adjustable lead times, service levels, costs, and more

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/?
   cd inventory-management-system
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit app, use the following command in your terminal:

```
streamlit run app.py
```

The application will open in your default web browser.

## How to Use

1. **Upload Data**: Start by uploading your SKU data in Excel format. The file should contain columns for SKU, Date, and Quantity.

2. **Select SKU**: Choose the specific SKU you want to analyze from the dropdown menu.

3. **Set Parameters**: Adjust various parameters such as lead time, service level, and costs according to your business needs.

4. **Choose Models**: Select your preferred forecasting and inventory models.

5. **Run Simulation**: Click the "Run Simulation" button to generate forecasts and inventory policies.

6. **Analyze Results**: Navigate through different tabs to view forecasts, inventory levels, performance metrics, and more.

7. **Sensitivity Analysis**: Use the sensitivity analysis tool to understand how changes in parameters affect your inventory performance.

8. **Download Results**: Export your results and updated policies for further analysis or implementation.

## File Structure

- `app.py`: Main Streamlit application file
- `app_utils.py`: Utility functions for the application
- `forecast.py`: Forecasting models
- `inventory.py`: Inventory policy models
- `sku.py`: SKU class definition and related functions
- `requirements.txt`: List of Python dependencies
- `logo.svg`: Company logo file

## Dependencies

Main dependencies include:
- Streamlit
- Pandas
- NumPy
- Plotly
- Statsmodels

For a complete list, see `requirements.txt`.
