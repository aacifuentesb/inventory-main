import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class SKU:
    def __init__(self, sku_id, data, params, forecast_model, inventory_model,periods):
        self.sku_id = sku_id
        self.data = data
        self.params = params
        # Add stockout_cost
        if 'stockout_cost' not in self.params:
            self.params['stockout_cost'] = self.get_stockout_cost(1)
        self.forecast_model = forecast_model
        self.inventory_model = inventory_model
        self.periods = periods
        if 'review_period' not in self.params:
            self.params['review_period'] = 1  # Default to weekly review if not specified
        self.forecast = None
        self.inventory_policy = None
        self.inventory_evolution = None
        self.profit_evolution = None
        self.best_distribution = None
        self.distribution_params = None
        self.metrics = None
        self.demand_evolution = None
        self.order_evolution = None
        self.stockouts = None
        self.unfufilled_demand = None
        self.order_points = None
        self.orders_arriving = None
        self.orders_in_transit = None

    def get_stockout_cost(self,stockout_cost_percentage=0.1):
        '''Calculate the stockout cost as a percentage of the unit cost'''
        return self.params['cost'] * stockout_cost_percentage

    def simulate_inventory(self, demand, periods):
        (self.inventory_evolution, self.order_evolution, self.stockouts, 
         self.profit_evolution, self.orders_arriving, self.orders_in_transit, self.unfufilled_demand) = \
            self.inventory_model.simulate(demand, periods, self.params)
        
        self.demand_evolution = demand
        self.order_points = self.order_evolution > 0
        
        self.calculate_metrics()

    def update_inventory_policy(self, new_policy):
        if self.inventory_policy == new_policy:
            print("New policy is the same as the current policy. No update needed.")
            return

        self.inventory_policy = new_policy.copy()
        self.inventory_model.policy = new_policy.copy()
        self.simulate_inventory(self.forecast['mean'], len(self.forecast['mean']))
    
    def generate_forecast(self, periods):
        self.forecast = self.forecast_model.forecast(self.data, periods)

    def get_forecast_metrics(self):
        return self.forecast_model.get_forecast_metrics(self.data)

    def calculate_inventory_policy(self):
        self.inventory_policy = self.inventory_model.calculate(self.data.values, self.params)

    def calculate_metrics(self):
        self.metrics = {
            'overall': {
                'average_inventory': np.mean(self.inventory_evolution),
                'stockout_rate': np.mean(self.stockouts),
                'total_profit': np.sum(self.profit_evolution),
                'average_profit': np.mean(self.profit_evolution),
                'service_level': 1 - np.mean(self.stockouts),
                'inventory_turnover': np.sum(self.demand_evolution) / np.mean(self.inventory_evolution),
                'total_orders': np.sum(self.order_points),
                'average_order_size': np.mean(self.order_evolution[self.order_evolution > 0]),
            },
            'weekly': {
                'inventory': self.inventory_evolution,
                'profit': self.profit_evolution,
                'demand': self.demand_evolution,
                'stockouts': self.stockouts,
                'orders': self.order_evolution,
            }
        }

    def plot_results(self):
        fig = make_subplots(rows=4, cols=1, subplot_titles=('Demand Forecast', 'Inventory Evolution', 'Profit Evolution', 'Stockouts'))
        
        # Demand Forecast
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name='Historical Demand'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.forecast['mean'].index, y=self.forecast['mean'], mode='lines', name='Forecast', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.forecast['lower'].index, y=self.forecast['lower'], fill=None, mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower CI'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.forecast['upper'].index, y=self.forecast['upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper CI'), row=1, col=1)
        
        # Inventory Evolution
        fig.add_trace(go.Scatter(x=list(range(len(self.inventory_evolution))), y=self.inventory_evolution, mode='lines', name='Inventory Level'), row=2, col=1)
        if isinstance(self.inventory_policy, dict) and 'reorder_point' in self.inventory_policy:
            reorder_point = self.inventory_policy['reorder_point']
        else:
            reorder_point = self.inventory_policy
        fig.add_trace(go.Scatter(x=list(range(len(self.inventory_evolution))), y=[reorder_point]*len(self.inventory_evolution), mode='lines', name='Reorder Point', line=dict(dash='dash')), row=2, col=1)
        
        # Profit Evolution
        fig.add_trace(go.Scatter(x=list(range(len(self.profit_evolution))), y=self.profit_evolution, mode='lines', name='Profit'), row=3, col=1)
        
        # Stockouts
        fig.add_trace(go.Bar(x=list(range(len(self.stockouts))), y=self.stockouts.astype(int), name='Stockouts'), row=4, col=1)
        
        # Add markers for order points
        order_indices = np.where(self.order_points)[0]
        fig.add_trace(go.Scatter(x=order_indices, y=self.inventory_evolution[order_indices], mode='markers', name='Order Points', marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
        
        fig.update_layout(height=1200, title_text=f"SKU {self.sku_id} Analysis")
        return fig


def aggregate_weekly(df):
    df['Week'] = df['Date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
    return df.groupby(['SKU', 'Week'])['QTY'].sum().reset_index()

def generate_weekly_time_series(df, sku_id):
    """
    Generate weekly time series for a SKU, filling zeros from last sale to current date
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'SKU', and 'QTY' columns
    sku_id : str
        SKU identifier to filter data
        
    Returns:
    --------
    pd.Series
        Weekly aggregated time series with zeros filled up to current date
    """
    # Filter data for specific SKU
    sku_data = df[df['SKU'] == sku_id].copy()
    sku_data['Date'] = pd.to_datetime(sku_data['Date'])
    
    # Get the start date (first sale) and end date (current date)
    start_date = sku_data['Date'].min()
    current_date = pd.Timestamp.now()
    # Round to previous Monday to ensure consistent weekly boundaries
    current_date = current_date - pd.Timedelta(days=current_date.weekday())
    
    # Create complete date range from start to current date
    full_range = pd.date_range(start=start_date, end=current_date, freq='W-MON')
    
    # Aggregate by week and reindex to fill missing weeks with zeros
    sku_data.set_index('Date', inplace=True)
    weekly_data = sku_data['QTY'].resample('W-MON').sum()
    weekly_data = weekly_data.reindex(full_range, fill_value=0)
    
    return weekly_data

def run_inventory_system(df, sku_id, params, forecast_model, inventory_model, periods, start_time=None):
    """
    Run inventory system simulation for a SKU
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical data
    sku_id : str
        SKU identifier
    params : dict
        Parameters for the inventory system
    forecast_model : ForecastModel
        Forecasting model to use
    inventory_model : InventoryModel
        Inventory model to use
    periods : int
        Number of periods to forecast
    start_time : datetime, optional
        Start time for analysis. If None, uses all available data
        
    Returns:
    --------
    SKU
        SKU object with simulation results
    """
    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        df = df[df['Date'] >= start_time].copy()
    
    weekly_data = generate_weekly_time_series(df, sku_id)
    
    # Create SKU object and run simulation
    sku = SKU(sku_id, weekly_data, params, forecast_model, inventory_model, periods)
    sku.generate_forecast(periods)
    sku.calculate_inventory_policy()
    sku.simulate_inventory(sku.forecast['mean'], periods)
    
    return sku

