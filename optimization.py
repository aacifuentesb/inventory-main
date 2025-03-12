import numpy as np
from scipy.stats import uniform
import streamlit as st
from scipy import stats
from itertools import product
from tqdm import tqdm
import random
from inventory import RQContinuousReview, SSPeriodicReview, BaseStockModel, NewsvendorModel, ModifiedContinuousReview
import plotly.graph_objects as go
from stqdm import stqdm

#Stats

def monte_carlo_optimization(sku, num_param_combinations=200, num_demand_scenarios=1000, target_service_level=0.95, optimization_model='Soft Constraint', objective_function='Minimize Total Cost'):
    best_params = None
    best_metric_value = float('inf') if objective_function == 'Minimize Total Cost' else float('-inf')
    original_policy = sku.inventory_policy.copy()

    initial_policy = sku.inventory_model.policy
    initial_params = sku.params.copy()
    initial_params.update(initial_policy)
    param_ranges = get_param_ranges(sku)
    param_combinations = generate_random_params(sku, param_ranges, num_param_combinations)
    # add initial params to the list
    param_combinations.append(initial_params)
    demand_scenarios = generate_demand_scenarios(sku, num_demand_scenarios)
    if objective_function == 'Minimize Total Cost':
        top_10 = {10**(10)-i : None for i in range(10)}
    else:
        top_10 = {-10**(10)+i : None for i in range(10)}

    for params in stqdm(param_combinations, desc="Evaluating parameter combinations"):
        total_metric_value = 0
        total_service_level = 0

        inventory_policy = sku.inventory_model.calculate(sku.data.values, params)
        # Update only the keys that exist in the model's policy
        for key in inventory_policy.keys():
            if key in params:
                sku.inventory_model.policy[key] = params[key]
                sku.inventory_policy[key] = params[key]            
        
        valid_solution = True

        for scenario_index, demand in enumerate(demand_scenarios):
            inventory, orders, stockouts, profits, _, _,unfufilled_demand = sku.inventory_model.simulate(demand, len(demand), params)
            
            service_level = 1 - np.sum(stockouts) / len(demand)
            total_service_level += service_level
            
            if objective_function == 'Minimize Total Cost':
                metric_value = calculate_total_cost(sku, inventory, orders, stockouts,unfufilled_demand, params)
            elif objective_function == 'Maximize Service Level':
                metric_value = service_level
            elif objective_function == 'Maximize Profit':
                metric_value = np.sum(profits)
            
            if optimization_model == 'Hard Constraint' and service_level < target_service_level:
                valid_solution = True
                break
            elif optimization_model == 'Soft Constraint':
                target_shortfall = max(0, target_service_level - service_level)
                if objective_function == 'Minimize Total Cost':
                    stockout_penalty = target_shortfall  * 10 * params['stockout_cost'] * np.sum(unfufilled_demand)
                    metric_value += stockout_penalty
                elif objective_function == 'Maximize Profit':
                    stockout_penalty = target_shortfall * (params['price'] - params['cost']) * np.sum(unfufilled_demand)
                    metric_value -= stockout_penalty
            
            total_metric_value += metric_value

        if not valid_solution:
            continue
        
        avg_metric_value = total_metric_value / num_demand_scenarios
        avg_service_level = total_service_level / num_demand_scenarios
        if optimization_model == "Hard Constraint":
            if avg_service_level < target_service_level:
                valid_solution = False
                continue

        # Create dictionary only with keys that exist in params
        important_dict = {}
        for key in sku.inventory_policy.keys():
            if key in params:
                important_dict[key] = params[key]
        important_dict["avg_service_level"] = avg_service_level

        if objective_function == 'Minimize Total Cost':
            max_value = max(list(top_10.keys()))
            if avg_metric_value < max_value:
                del top_10[max_value]
                top_10[avg_metric_value] = important_dict
        else:
            min_value = min(list(top_10.keys()))
            if avg_metric_value > min_value:
                del top_10[min_value]
                top_10[avg_metric_value] = important_dict

        if objective_function == 'Minimize Total Cost':
            if avg_metric_value < best_metric_value:
                if (optimization_model == "Hard Constraint" and avg_service_level >= target_service_level) or optimization_model == "Soft Constraint":
                    best_metric_value = avg_metric_value
                    best_params = params.copy()
        else:  # 'Maximize Service Level' or 'Maximize Profit'
            if avg_metric_value > best_metric_value:
                if (optimization_model == "Hard Constraint" and avg_service_level >= target_service_level) or optimization_model == "Soft Constraint":
                    best_metric_value = avg_metric_value
                    best_params = params.copy()

    if best_params is None:
        return None, None, None

    # Change inventory policy back to original
    sku.inventory_policy = original_policy

    return best_params, best_metric_value, top_10

def calculate_total_cost(sku, inventory, orders, stockouts,unfufilled_demand, params):
    holding_cost = np.sum(inventory * params['holding_cost'])
    stockout_cost = np.sum(unfufilled_demand) * params['stockout_cost']
    ordering_cost = np.sum(orders > 0) * params['order_cost']
    inventory_cost = np.sum(inventory) * params['cost']
    total_cost = inventory_cost + holding_cost + stockout_cost + ordering_cost
    return total_cost

def generate_random_params(sku, param_ranges, num_combinations):
    param_combinations = []
    for _ in range(num_combinations):
        current_params = sku.params.copy()
        for param, (min_val, max_val) in param_ranges.items():
            min_val_positive = max(0, min_val)
            max_val_positive = max(0, max_val)
            current_params[param] = uniform.rvs(loc=min_val_positive, scale=max_val_positive - min_val_positive)
            if param in ["eoq", "reorder_point", "safety_stock", "order_up_to", "base_stock_level", "order_quantity"]:
                current_params[param] = int(current_params[param])
        param_combinations.append(current_params)
    return param_combinations

def generate_demand_scenarios(sku, num_scenarios):
    scenarios = []
    for _ in range(num_scenarios):
        sku.generate_forecast(sku.periods)
        demand = sku.forecast['mean'].values
        scenarios.append(demand)
    return scenarios

def calculate_total_cost_sku(sku):
    
    inventory = sku.inventory_evolution
    orders = sku.order_points
    stockouts = sku.stockouts
    params = sku.params
    unfufilled_demand = sku.unfufilled_demand
    holding_cost = np.sum(inventory * params['holding_cost'])
    stockout_cost = np.sum(unfufilled_demand) * params['stockout_cost']
    ordering_cost = np.sum(orders > 0) * params['order_cost']
    inventory_cost = np.sum(inventory) * params['cost']
    return inventory_cost + holding_cost + stockout_cost + ordering_cost

def generate_grid_params(sku, param_ranges, num_combinations):
    # Determine number of points for each parameter
    num_params = len(param_ranges)
    points_per_param = max(2, int(num_combinations ** (1 / num_params)))
    
    # Generate grid points for each parameter
    grid_points = {}
    for param, (min_val, max_val) in param_ranges.items():
        grid_points[param] = np.linspace(max(0, min_val), max_val, points_per_param)
    
    # Generate all combinations
    param_combinations = []
    for values in product(*grid_points.values()):
        current_params = sku.params.copy()
        for param, value in zip(param_ranges.keys(), values):
            current_params[param] = value
        param_combinations.append(current_params)
    
    # If we have more combinations than requested, randomly sample
    if len(param_combinations) > num_combinations:
        param_combinations = random.sample(param_combinations, num_combinations)
    
    return param_combinations

def get_param_ranges(sku):
    # Calculate key statistics from historical data
    demand_mean = np.mean(sku.data)
    demand_std = np.std(sku.data)
    lead_time = sku.params['lead_time_weeks']
    
    # Calculate some common values used across models
    lead_time_demand = demand_mean * lead_time
    lead_time_demand_std = demand_std * np.sqrt(lead_time)
    safety_stock_max = 3 * lead_time_demand_std  # 3 sigma for 99.7% service level
    
    if isinstance(sku.inventory_model, RQContinuousReview):
        annual_demand = np.sum(sku.data) * (52 / len(sku.data))
        #Min eoq: min(1,min(sku.data)), Max eoq: max(sku.data)*4
        min_eoq = min(1, min(sku.data))
        max_eoq = max(sku.data) * 4

        min_reorder_point = int(lead_time * min(sku.data) * 0.5)
        max_reorder_point = lead_time * max(sku.data) * 4 
        return {
            'eoq': (min_eoq, max_eoq),
            'reorder_point': (min_reorder_point, max_reorder_point)
        }
    
    elif isinstance(sku.inventory_model, SSPeriodicReview):
        review_period = sku.params['review_period']
        review_lead_time_demand = demand_mean * (lead_time + review_period)
        review_lead_time_demand_std = demand_std * np.sqrt(lead_time + review_period)
        safety_stock_max_review = 3 * review_lead_time_demand_std
        return {
            'safety_stock': (0, safety_stock_max_review),
            'reorder_point': (review_lead_time_demand, review_lead_time_demand + safety_stock_max_review),
            'order_up_to': (review_lead_time_demand + demand_mean * review_period, 
                            review_lead_time_demand + safety_stock_max_review + 2 * demand_mean * review_period)
        }
    
    elif isinstance(sku.inventory_model, BaseStockModel):
        return {
            'base_stock_level': (lead_time_demand, lead_time_demand + safety_stock_max + demand_mean)
        }
    
    elif isinstance(sku.inventory_model, NewsvendorModel):
        critical_ratio = (sku.params['price'] - sku.params['cost']) / sku.params['price']
        z = stats.norm.ppf(critical_ratio)
        return {
            'order_quantity': (demand_mean, demand_mean + z * demand_std)
        }
    
    elif isinstance(sku.inventory_model, ModifiedContinuousReview):
        # eoq, reorder_point,safety_stock
        min_eoq = min(1, min(sku.data)) * sku.params['review_period']
        max_eoq = max(sku.data) * sku.params["review_period"] * 2
        min_reorder_point = int(lead_time * min(sku.data))
        max_reorder_point = int(lead_time * max(sku.data)) * 2
        min_target = max_reorder_point
        max_target = max_reorder_point + max_eoq

        return {
            'eoq': (min_eoq, max_eoq),
            'reorder_point': (min_reorder_point, max_reorder_point),
            'safety_stock': (0, safety_stock_max),
            'target_level': (min_target, max_target)
        }

    else:
        raise ValueError("Unsupported inventory model type")

def calculate_metric(sku, inventory, orders, stockouts,unfufilled_demand, params, metric):
    if metric == "Minimize Total Cost":
        holding_cost = np.sum(inventory * params['holding_cost'])
        stockout_cost = np.sum(unfufilled_demand) * params['stockout_cost']
        ordering_cost = np.sum(orders > 0) * params['order_cost']
        total_cost = holding_cost + stockout_cost + ordering_cost
        return total_cost
    elif metric == "Maximize Service Level":
        return 1 - np.mean(stockouts)
    elif metric == "Maximize Inventory Turnover":
        return np.sum(sku.demand_evolution) / np.mean(inventory)
    else:
        raise ValueError("Unsupported optimization metric")

def plot_inventory_and_demand_opt(sku, one_axis_y=True):
    if one_axis_y:
        fig = go.Figure()
    else:
        pass
    
    # Add Inventory Level trace
    fig.add_trace(
        go.Scatter(x=list(range(len(sku.inventory_evolution))), 
                   y=sku.inventory_evolution, 
                   mode='lines', 
                   name='Inventory Level')
    )
    
    # Add Demand trace
    fig.add_trace(
        go.Scatter(x=list(range(len(sku.demand_evolution))), 
                   y=sku.demand_evolution, 
                   mode='lines', 
                   name='Demand', 
                   line=dict(dash='dot')),
        secondary_y=not one_axis_y if not one_axis_y else None
    )
    
    # Add Reorder Point trace
    if isinstance(sku.inventory_policy, dict) and 'reorder_point' in sku.inventory_policy:
        reorder_point = sku.inventory_policy['reorder_point']
    else:
        reorder_point = sku.inventory_policy
    
    fig.add_trace(
        go.Scatter(x=list(range(len(sku.inventory_evolution))), 
                   y=[reorder_point]*len(sku.inventory_evolution), 
                   mode='lines', 
                   name='Reorder Point', 
                   line=dict(dash='dash'))
    )
    
    # Add Order Points trace
    order_indices = np.where(sku.order_points)[0]
    fig.add_trace(
        go.Scatter(x=order_indices, 
                   y=sku.inventory_evolution[order_indices], 
                   mode='markers', 
                   name='Order Points', 
                   marker=dict(color='green', symbol='triangle-up', size=10))
    )
    
    # Update layout
    fig.update_layout(
        title="Inventory Evolution and Demand",
        xaxis_title="Time Period"
    )
    
    if one_axis_y:
        fig.update_yaxes(title_text="Quantity")
    else:
        fig.update_yaxes(title_text="Inventory Level", secondary_y=False)
        fig.update_yaxes(title_text="Demand", secondary_y=True)
    
    return fig

def calculate_metrics(original_sku, optimized_sku):
    metrics = {
        'original': {},
        'optimized': {}
    }
    
    for sku_type, sku in [('original', original_sku), ('optimized', optimized_sku)]:

        metrics[sku_type]['Total Cost ($)'] = calculate_total_cost_sku(sku)
        metrics[sku_type]['Service Level (%)'] = 1 - np.mean(sku.stockouts)
        metrics[sku_type]['Inventory Turnover'] = np.sum(sku.demand_evolution) / np.mean(sku.inventory_evolution)
        metrics[sku_type]['Average Inventory (units)'] = np.mean(sku.inventory_evolution)
    
    return metrics


