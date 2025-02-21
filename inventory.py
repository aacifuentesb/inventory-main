import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from itertools import product


class InventoryModel(ABC):
    def __init__(self):
        self.policy = {}

    @abstractmethod
    def calculate(self, demand, params):
        pass

    @abstractmethod
    def simulate(self, demand, periods, params):
        pass

    @staticmethod
    def get_description():
        return ""

    def update_policy(self, params):
        # Find the keys in params that are in the policy
        policy_keys = set(self.policy.keys())
        # Update the policy with the new parameters
        self.policy.update({k: v for k, v in params.items() if k in policy_keys})
        

    def optimize(self, demand, params, param_ranges):
        best_cost = float('inf')
        best_policy = None
        target_service_level = params['service_level']

        # Generate grid of parameter combinations
        param_combinations = product(*[np.linspace(r[0], r[1], 100) for r in param_ranges.values()])

        for params_values in param_combinations:
            current_params = dict(zip(param_ranges.keys(), params_values))
            current_params.update(params)

            # Generate demand forecast
            forecast_demand = np.random.normal(np.mean(demand), np.std(demand), params['periods'])

            # Simulate inventory
            inventory, orders, stockouts, profits, _, _, unfufilled_demand = self.simulate(forecast_demand, params['periods'], current_params)

            # Calculate total cost
            total_cost = (
                np.sum(inventory * params['holding_cost']) +
                np.sum(orders * params['order_cost']) +
                np.sum(unfufilled_demand) * params['stockout_cost']
            )

            # Calculate service level
            service_level = 1 - np.mean(stockouts)

            # Check if solution is feasible and better than current best
            if service_level >= target_service_level and total_cost < best_cost:
                best_cost = total_cost
                best_policy = current_params

        return best_policy

class RQContinuousReview(InventoryModel):
    def calculate(self, demand, params, opt = False):
        
        if not opt:
            annual_demand = np.sum(demand) * (52 / len(demand))
            annual_demand = max(annual_demand, 0.01)
            
            avg_weekly_demand = np.mean(demand)
            std_weekly_demand = np.std(demand)
            
            lead_time_demand = avg_weekly_demand * params['lead_time_weeks']
            lead_time_std = std_weekly_demand * np.sqrt(params['lead_time_weeks'])
            
            # Use provided EOQ if available, otherwise calculate
            if 'eoq' in params:
                eoq = params['eoq']
            else:
                safety_factor = params.get('safety_factor', 1.5)
                eoq = safety_factor * np.sqrt((2 * annual_demand * params['order_cost']) / params['holding_cost'])
                eoq = max(eoq, lead_time_demand)
            
            # Use provided reorder point if available, otherwise calculate
            if 'reorder_point' in params:
                reorder_point = params['reorder_point']
            else:
                z = stats.norm.ppf(params['service_level'])
                safety_stock = z * lead_time_std
                reorder_point = lead_time_demand + safety_stock
            
            self.policy = {
                'eoq': np.round(eoq),
                'reorder_point': np.round(reorder_point),
                'safety_stock': np.round(reorder_point - lead_time_demand)
            }
            #print("Calculated policy:", self.policy)
            return self.policy
        else:
            param_ranges = {
            'eoq': (np.mean(demand), np.mean(demand) * 10),
            'reorder_point': (np.mean(demand) * params['lead_time_weeks'], np.mean(demand) * params['lead_time_weeks'] * 3),
            'safety_stock': (0, np.mean(demand) * 2)
            }
        
            best_policy = self.optimize(demand, params, param_ranges)
            self.policy = {
                'eoq': np.round(best_policy['eoq']),
                'reorder_point': np.round(best_policy['reorder_point']),
                'safety_stock': np.round(best_policy['safety_stock'])
                                    
            }
            return self.policy

    def simulate(self, demand, periods, params):
        inventory = np.zeros(periods)
        orders = np.zeros(periods)
        profits = np.zeros(periods)
        stockouts = np.zeros(periods, dtype=bool)
        unfufilled_demand = np.zeros(periods)
        orders_arriving = np.zeros(periods)
        orders_in_transit = np.zeros(periods)

        inventory[0] = params['initial_inventory']
        lead_time = params['lead_time_weeks']
        review_period = params['review_period']
        
        # Check if we should order in the first period
        inventory_position = inventory[0] + sum(orders[max(0, 0-lead_time):0])
        if inventory_position <= self.policy['reorder_point']:
            orders[0] = self.policy['eoq']
            if lead_time < periods:
                orders_arriving[lead_time] += orders[0]
        
        for t in range(1, periods):
            # Only check for ordering on review periods after first order
            if t % review_period == 0:
                inventory_position = inventory[t-1] + sum(orders[max(0, t-lead_time):t])
                
                if inventory_position <= self.policy['reorder_point']:
                    orders[t] = self.policy['eoq']
                    if t + lead_time < periods:
                        orders_arriving[t + lead_time] += orders[t]
            
            received_order = orders_arriving[t]
            inventory[t] = max(0, inventory[t-1] + received_order - demand[t])
            
            if inventory[t] < demand[t]:
                stockouts[t] = True
                unfufilled_demand[t] = demand[t] - inventory[t]
            
            sales = min(inventory[t], demand[t])
            profits[t] = (sales * params['price'] - 
                          orders[t] * params['cost'] - 
                          inventory[t] * params['holding_cost']-
                          unfufilled_demand[t] * params['stockout_cost'])
            
            if lead_time > 1:
                orders_in_transit[t] = sum(orders[max(0, t-lead_time+1):t])
            else:
                orders_in_transit[t] = 0

        return inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand
    
    @staticmethod
    def get_description():
        return """
        Robust (r, Q) Continuous Review Policy

        This policy uses two main parameters:
        - r (reorder point): When the inventory position (on-hand + on-order) reaches this point, an order is placed.
        - Q (order quantity): The amount ordered each time, based on a modified Economic Order Quantity (EOQ).

        Key components:
        - Modified EOQ: Q = safety_factor * √((2AD) / h), where safety_factor > 1 for larger order quantities
        - Safety Stock: SS = z * σ * √L
        - Reorder Point: r = μL + SS

        Where:
        - A: fixed cost per order
        - D: annual demand
        - h: holding cost per unit per year
        - z: safety factor based on service level
        - σ: standard deviation of weekly demand
        - L: lead time in weeks
        - μL: average demand during lead time

        This robust version ensures larger order quantities and maintains a safety stock to reduce the risk of stockouts.
        """

class SSPeriodicReview(InventoryModel):
    def calculate(self, demand, params,opt = False):
        #print("SSPeriodicReview.calculate called with params:", params)
        if not opt:    
            demand_mean = np.mean(demand)
            demand_std = np.std(demand)
            review_period = params['review_period']
            lead_time = params['lead_time_weeks']
            
            if 'safety_stock' in params:
                safety_stock = params['safety_stock']
            else:
                z = stats.norm.ppf(params['service_level'])
                safety_stock = z * demand_std * np.sqrt(lead_time + review_period)
            
            if 'reorder_point' in params:
                reorder_point = params['reorder_point']
            else:
                reorder_point = demand_mean * (lead_time + review_period) + safety_stock
            
            if 'order_up_to' in params:
                order_up_to = params['order_up_to']
            else:
                order_up_to = reorder_point + demand_mean * review_period

            self.policy = {
                'safety_stock': np.round(safety_stock),
                'reorder_point': np.round(reorder_point),
                'order_up_to': np.round(order_up_to)
            }
            #print("Calculated policy:", self.policy)
            return self.policy
        else:
            param_ranges = {
            'safety_stock': (0, np.mean(demand) * params['lead_time_weeks'] * 2),
            'order_up_to': (np.mean(demand) * (params['lead_time_weeks'] + params['review_period'])*0.5,
                            np.mean(demand) * (params['lead_time_weeks'] + params['review_period']) * 6)
        }
        
            best_policy = self.optimize(demand, params, param_ranges)
            self.policy = {
                'safety_stock': np.round(best_policy['safety_stock']),
                'order_up_to': np.round(best_policy['order_up_to'])
            }
            return self.policy

    def simulate(self, demand, periods, params):
        inventory = np.zeros(periods)
        orders = np.zeros(periods)
        profits = np.zeros(periods)
        stockouts = np.zeros(periods, dtype=bool)
        unfufilled_demand = np.zeros(periods)
        orders_arriving = np.zeros(periods)
        orders_in_transit = np.zeros(periods)

        inventory[0] = params['initial_inventory']
        review_period = params['review_period']
        lead_time = params['lead_time_weeks']
        
        # Check if we should order in the first period
        inventory_position = inventory[0] + sum(orders[max(0, 0-lead_time):0])
        if inventory_position <= self.policy['reorder_point']:
            order_qty = self.policy['order_up_to'] - inventory_position
            orders[0] = max(order_qty, 0)
            if lead_time < periods:
                orders_arriving[lead_time] += orders[0]
        
        for t in range(1, periods):
            if t % review_period == 0:
                inventory_position = inventory[t-1] + sum(orders[max(0, t-lead_time):t])
                if inventory_position <= self.policy['reorder_point']:
                    order_qty = self.policy['order_up_to'] - inventory_position
                    orders[t] = max(order_qty, 0)
                    if t + lead_time < periods:
                        orders_arriving[t + lead_time] += orders[t]
            
            received_order = orders_arriving[t]
            inventory[t] = max(0, inventory[t-1] + received_order - demand[t])
            
            if inventory[t] < demand[t]:
                stockouts[t] = True
                unfufilled_demand[t] = demand[t] - inventory[t]
            
            sales = min(inventory[t], demand[t])
            profits[t] = (sales * params['price'] - 
                          orders[t] * params['cost'] - 
                          inventory[t] * params['holding_cost'] - 
                            unfufilled_demand[t] * params['stockout_cost'])
            
            if lead_time > 1:
                orders_in_transit[t] = sum(orders[max(0, t-lead_time+1):t])
            else:
                orders_in_transit[t] = 0

        return inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand

    @staticmethod
    def get_description():
        return """
        (s, S) Periodic Review Policy

        This policy uses two main parameters:
        - s (safety stock): The extra inventory kept to prevent stockouts during lead time.
        - S (order-up-to level): The target inventory level after ordering up to the reorder point.

        Key components:
        - Safety Stock: SS = z * σ * √(L + R)
        - Reorder Point: r = μ(L + R) + SS
        - Order-Up-To Level: S = r + μR

        Where:
        - z: safety factor based on service level
        - σ: standard deviation of weekly demand
        - L: lead time in weeks
        - R: review period in weeks
        - μ: average weekly demand

        This policy ensures that inventory is replenished up to a target level, with additional safety stock to prevent stockouts during lead time.
        """

class BaseStockModel(InventoryModel):
    def calculate(self, demand, params,opt = False):
        if not opt:
            if 'base_stock_level' in params:
                base_stock_level = params['base_stock_level']
            else:
                z = stats.norm.ppf(params['service_level'])
                lead_time_demand = np.mean(demand) * params['lead_time_weeks']
                lead_time_std = np.std(demand) * np.sqrt(params['lead_time_weeks'])
                base_stock_level = lead_time_demand + z * lead_time_std

            self.policy = {'base_stock_level': np.round(base_stock_level)}
            return self.policy
        else:
            param_ranges = {
            'base_stock_level': (np.mean(demand) * params['lead_time_weeks'],
                                 np.mean(demand) * params['lead_time_weeks'] * 3)
        }
        
            best_policy = self.optimize(demand, params, param_ranges)
            self.policy = {'base_stock_level': np.round(best_policy['base_stock_level'])}
            return self.policy

    def simulate(self, demand, periods, params):
        inventory = np.zeros(periods)
        orders = np.zeros(periods)
        profits = np.zeros(periods)
        stockouts = np.zeros(periods, dtype=bool)
        unfufilled_demand = np.zeros(periods)
        orders_arriving = np.zeros(periods)
        orders_in_transit = np.zeros(periods)

        inventory[0] = params['initial_inventory']
        lead_time = params['lead_time_weeks']
        review_period = params['review_period']
        
        # Check if we should order in the first period
        inventory_position = inventory[0] + sum(orders[max(0, 0-lead_time):0])
        order_qty = self.policy['base_stock_level'] - inventory_position
        orders[0] = max(order_qty, 0)
        if lead_time < periods:
            orders_arriving[lead_time] += orders[0]
        
        for t in range(1, periods):
            if t % review_period == 0:  # Only check for ordering on review periods
                inventory_position = inventory[t-1] + sum(orders[max(0, t-lead_time):t])
                order_qty = self.policy['base_stock_level'] - inventory_position
                orders[t] = max(order_qty, 0)
                
                if t + lead_time < periods:
                    orders_arriving[t + lead_time] += orders[t]
            
            received_order = orders_arriving[t]
            inventory[t] = max(0, inventory[t-1] + received_order - demand[t])
            
            if inventory[t] < demand[t]:
                stockouts[t] = True
                unfufilled_demand[t] = demand[t] - inventory[t]
            
            sales = min(inventory[t], demand[t])
            profits[t] = (sales * params['price'] - 
                          orders[t] * params['cost'] - 
                          inventory[t] * params['holding_cost']
                          - unfufilled_demand[t] * params['stockout_cost'])
            
            if lead_time > 1:
                orders_in_transit[t] = sum(orders[max(0, t-lead_time+1):t])
            else:
                orders_in_transit[t] = 0

        return inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand

    @staticmethod
    def get_description():
        return """
        Base Stock Policy

        The Base Stock policy uses a fixed inventory level to determine when to place orders.
        When the inventory position (on-hand + on-order) falls below the base stock level, an order is placed.

        Key components:
        - Base Stock Level: The fixed inventory level that triggers an order when inventory falls below it.

        This policy is useful for maintaining a consistent inventory level and ensuring that orders are placed in a timely manner.
        """

class NewsvendorModel(InventoryModel):
    def calculate(self, demand, params):
        #print("NewsvendorModel.calculate called with params:", params)
        
        if 'order_quantity' in params:
            order_quantity = params['order_quantity']
        else:
            critical_ratio = (params['price'] - params['cost']) / params['price']
            z = stats.norm.ppf(critical_ratio)
            order_quantity = np.mean(demand) + z * np.std(demand)

        self.policy = {'order_quantity': np.round(order_quantity)}
        #print("Calculated policy:", self.policy)
        return self.policy

    def simulate(self, demand, periods, params):
        inventory = np.zeros(periods)
        orders = np.zeros(periods)
        profits = np.zeros(periods)
        stockouts = np.zeros(periods, dtype=bool)
        unfufilled_demand = np.zeros(periods)
        orders_arriving = np.zeros(periods)
        orders_in_transit = np.zeros(periods)

        lead_time = params['lead_time_weeks']
        review_period = params['review_period']

        for t in range(periods):
            if t % review_period == 0:
                orders[t] = self.policy['order_quantity']
                if t + lead_time < periods:
                    orders_arriving[t + lead_time] += orders[t]
            
            received_order = orders_arriving[t]
            inventory[t] = received_order
            
            if inventory[t] < demand[t]:
                stockouts[t] = True
                unfufilled_demand[t] = demand[t] - inventory[t]
            
            sales = min(inventory[t], demand[t])
            profits[t] = (sales * params['price'] - 
                          orders[t] * params['cost'] - 
                          max(0, inventory[t] - demand[t]) * params['holding_cost']-
                            unfufilled_demand[t] * params['stockout_cost'])

            
            # Clear remaining inventory (perishable goods)
            inventory[t] = 0
            
            if lead_time > 1:
                orders_in_transit[t] = sum(orders[max(0, t-lead_time+1):t])
            else:
                orders_in_transit[t] = 0

        return inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand

    @staticmethod
    def get_description():
        return """
        Newsvendor Model

        The Newsvendor model is used for perishable goods with uncertain demand.
        The goal is to find the optimal order quantity that maximizes expected profit.
        
        Key components:
        - Order Quantity: Q = μ + zσ, where z is the critical ratio based on cost and price
        
        Where:
        - μ: mean demand
        - σ: standard deviation of demand
        - z: critical ratio based on cost and price
        
        This model is useful for optimizing order quantities for perishable goods with uncertain demand.
        """

class ModifiedContinuousReview(InventoryModel):
    def calculate(self, demand, params, opt=True):
        if not opt:
            return self._calculate_policy(demand, params)
        else:
            return self._optimize_policy(demand, params)

    def _calculate_policy(self, demand, params):
        # First validate we have enough historical data
        non_zero_sales = demand[demand > 0]
        if len(non_zero_sales) < 2:
            raise ValueError(f"Insufficient historical sales data. Found only {len(non_zero_sales)} non-zero sales points, minimum required is 2.")
            
        # Existing calculation logic
        annual_demand = np.sum(demand) * (52 / len(demand))
        annual_demand = max(annual_demand, 0.01)
        
        avg_weekly_demand = np.mean(demand)
        std_weekly_demand = np.std(demand)
        
        lead_time = params['lead_time_weeks']
        review_period = params['review_period']
        
        protected_period = lead_time + review_period
        protected_period_demand = avg_weekly_demand * protected_period
        protected_period_std = std_weekly_demand * np.sqrt(protected_period)
        
        # Calculate minimum EOQ based on average weekly demand and review period
        min_eoq = max(
            avg_weekly_demand * review_period,  # At least cover review period demand
            np.ceil(np.percentile(non_zero_sales, 25))  # Use 25th percentile of non-zero sales only
        )
        
        if 'eoq' in params:
            order_quantity = max(params['eoq'], min_eoq)
        else:
            if params['order_cost'] > 0:
                basic_eoq = np.sqrt((2 * params['order_cost'] * annual_demand) / params['holding_cost'])
                order_quantity = max(
                    min_eoq,
                    avg_weekly_demand * review_period * np.ceil(basic_eoq / (avg_weekly_demand * review_period))
                )
            else:
                z = stats.norm.ppf(params['service_level'])
                order_quantity = max(
                    min_eoq,
                    avg_weekly_demand * review_period,
                    z * protected_period_std
                )
        
        # Calculate safety stock first
        z = stats.norm.ppf(params['service_level'])
        safety_stock = z * protected_period_std
        
        # Then calculate reorder point using the safety stock
        if 'reorder_point' in params:
            reorder_point = max(params['reorder_point'], protected_period_demand)
        else:
            reorder_point = protected_period_demand + safety_stock  # ROP = D*L + SS
        
        self.policy = {
            'eoq': np.round(max(order_quantity, 1), 2),  # Ensure minimum of 1
            'reorder_point': np.round(reorder_point, 2),
            'safety_stock': np.round(safety_stock, 2)  # Use the directly calculated safety stock
        }
        
        return self.policy

    def _optimize_policy(self, demand, params):
        avg_weekly_demand = np.mean(demand)
        std_weekly_demand = np.std(demand)
        lead_time = params['lead_time_weeks']
        review_period = params['review_period']
        protected_period = lead_time + review_period
        protected_period_demand = avg_weekly_demand * protected_period
        protected_period_std = std_weekly_demand * np.sqrt(protected_period)

        mean_demand = np.mean(demand)
        param_ranges = {
            'eoq': (max(mean_demand * params['review_period'] * 0.5, 1), 
                    mean_demand * params['review_period'] * 10),
            'safety_stock': (0, protected_period_std * 3),  # Up to 3 sigma for safety stock
            'reorder_point': (protected_period_demand, protected_period_demand * 3)
        }
        
        best_cost = float('inf')
        best_policy = None
        target_service_level = params['service_level']

        # Generate grid of parameter combinations
        param_combinations = product(*[np.linspace(r[0], r[1], 10) for r in param_ranges.values()])

        for eoq, safety_stock, reorder_point in param_combinations:
            current_params = params.copy()
            current_params['eoq'] = np.round(eoq)
            current_params['safety_stock'] = np.round(safety_stock)
            current_params['reorder_point'] = np.round(reorder_point)

            self.policy = {
                'eoq': np.round(eoq),
                'safety_stock': np.round(safety_stock),
                'reorder_point': np.round(reorder_point)
            }

            # Generate demand forecast
            forecast_demand = np.random.normal(np.mean(demand), np.std(demand), params['periods'])

            # Simulate inventory
            inventory, orders, stockouts, profits, _, _, unfufilled_demand = self.simulate(forecast_demand, params['periods'], current_params)

            # Calculate total cost
            total_cost = (
                np.sum(inventory * params['holding_cost']) +
                np.sum(orders > 0) * params['order_cost'] +  # Fixed cost per order
                np.sum(unfufilled_demand) * params['stockout_cost']
            )

            # Calculate service level
            service_level = 1 - np.mean(stockouts)

            # Check if solution is feasible and better than current best
            if service_level >= target_service_level and total_cost < best_cost:
                best_cost = total_cost
                best_policy = self.policy.copy()

        if best_policy is None:
            # Use the initial policy if no feasible solution was found
            best_policy = self.policy

            
        self.policy = best_policy
        return self.policy

    def simulate(self, demand, periods, params):
        inventory = np.zeros(periods)
        orders = np.zeros(periods)
        profits = np.zeros(periods)
        stockouts = np.zeros(periods, dtype=bool)
        unfufilled_demand = np.zeros(periods)
        orders_arriving = np.zeros(periods)
        orders_in_transit = np.zeros(periods)

        inventory[0] = params['initial_inventory']
        lead_time = params['lead_time_weeks']
        review_period = params['review_period']

        # Check if we should order in the first period
        inventory_position = inventory[0] + sum(orders[max(0, 0-lead_time):0])
        if inventory_position <= self.policy['reorder_point']:
            order_quantity = self.policy['eoq']
            orders[0] = order_quantity
            if lead_time < periods:
                orders_arriving[lead_time] += order_quantity

        for t in range(1, periods):
            # Check for order arrival
            if t >= lead_time:
                inventory[t] = inventory[t-1] + orders[t-lead_time]
            else:
                inventory[t] = inventory[t-1]
            
            # Handle demand
            if inventory[t] < demand[t]:
                stockouts[t] = True
                sales = inventory[t]
                inventory[t] = 0
                unfufilled_demand[t] = demand[t] - sales
            else:
                sales = demand[t]
                inventory[t] -= sales
            
            # Calculate profits
            profits[t] = (sales * params['price'] - 
                          (orders[t] if t >= lead_time else 0) * params['cost'] - 
                          inventory[t] * params['holding_cost'] -
                          unfufilled_demand[t] * params['stockout_cost'])
            
            # Check for ordering (only on review periods after first order)
            if t % review_period == 0:
                inventory_position = (inventory[t] + 
                                      np.sum(orders[max(0, t-lead_time+1):t]))
                
                if inventory_position <= self.policy['reorder_point']:
                    order_quantity = self.policy['eoq']
                    orders[t] = order_quantity
                    if t + lead_time < periods:
                        orders_arriving[t + lead_time] += order_quantity
            
            # Update orders in transit
            orders_in_transit[t] = np.sum(orders[max(0, t-lead_time+1):t])

        return inventory, orders, stockouts, profits, orders_arriving, orders_in_transit, unfufilled_demand

    @staticmethod
    def get_description():
        return """
        Enhanced (r, Q) Periodic Review Policy

        This policy uses two main parameters:
        - r (reorder point): When the inventory position (on-hand + on-order) reaches this point, an order is placed.
        - Q (order quantity): The amount ordered each time, based on a modified Economic Order Quantity (EOQ).

        Key enhancements:
        1. Considers both lead time and review period in calculations.
        2. Uses a protected period (lead time + review period) for safety stock calculations.
        3. Adjusts EOQ to be a multiple of review period demand.
        4. Accounts for orders in transit in the inventory position calculation.
        5. Only places orders on review periods.

        Key components:
        - Modified EOQ: Q = ceil(√((2AD) / h) / (R * μ)) * R * μ
        - Safety Stock: SS = z * σ * √(L + R)
        - Reorder Point: r = μ(L + R) + SS

        Where:
        - A: fixed cost per order
        - D: annual demand
        - h: holding cost per unit per year
        - z: safety factor based on service level
        - σ: standard deviation of weekly demand
        - L: lead time in weeks
        - R: review period in weeks
        - μ: average weekly demand

        This enhanced version provides a more robust inventory management strategy by considering the interplay between review periods, lead times, and demand variability.
        """