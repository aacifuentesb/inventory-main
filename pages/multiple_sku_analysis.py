import streamlit as st
import pandas as pd
import numpy as np
from forecast import NormalDistributionForecast, SeasonalNormalDistributionForecast
from inventory import ModifiedContinuousReview
from sku import run_inventory_system
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from app_utils import get_base64_of_bin_file
import io

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Multiple SKU Analysis", page_icon="📊")

# Customize the sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            padding: 2rem 1rem;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1f2937;
            padding: 0.5rem 0;
        }
        .sidebar-content {
            margin-top: 2rem;
        }
        .css-1d391kg {
            padding-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)



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
        <h1 class="title">📦 Multiple SKU Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

@st.cache_data
def load_data(file):
    try:
        # Read both sheets
        historic_data = pd.read_excel(file, sheet_name="Historic Data")
        master_data = pd.read_excel(file, sheet_name="Master Data")
        
        # Validate historic data
        required_historic_cols = ['Date', 'SKU', 'QTY']
        if not all(col in historic_data.columns for col in required_historic_cols):
            raise ValueError("Historic Data sheet must contain columns: Date, SKU, QTY")
        
        # Validate master data
        required_master_cols = ['SKU', 'lead_time_weeks', 'initial_inventory', 'service_level', 
                              'order_cost', 'cost', 'price', 'review_period']
        if not all(col in master_data.columns for col in required_master_cols):
            raise ValueError(f"Master Data sheet missing required columns. Required: {required_master_cols}")
        
        # Process historic data
        historic_data['Date'] = pd.to_datetime(historic_data['Date'])
        historic_data = historic_data.sort_values(['SKU', 'Date'])
        
        # Calculate default values for optional parameters
        if 'holding_cost' not in master_data.columns:
            master_data['holding_cost'] = master_data['cost'] * 0.01  # 1% of product cost per week
        else:
            master_data['holding_cost'] = master_data['holding_cost'].fillna(master_data['cost'] * 0.01)
            
        if 'stockout_cost' not in master_data.columns:
            master_data['stockout_cost'] = master_data['cost'] * 0.1  # 10% of product cost
        else:
            master_data['stockout_cost'] = master_data['stockout_cost'].fillna(master_data['cost'] * 0.1)
        
        # Optimize memory usage
        historic_data, hist_reduction = optimize_dataframe_memory(historic_data)
        master_data, master_reduction = optimize_dataframe_memory(master_data)
        
        st.info(f"Memory optimization: Historic data reduced by {hist_reduction:.1f}%, Master data reduced by {master_reduction:.1f}%")
        
        return historic_data, master_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def run_multiple_sku_analysis(historic_data, master_data, periods, batch_size=20):
    results = []
    supply_plans = []
    # Reset seasonality counter at the start of each analysis
    st.session_state.seasonality_extracted = 0
    
    forecast_model = SeasonalNormalDistributionForecast(
        zero_demand_strategy='rolling_mean',
        rolling_window=4,
        zero_train_strategy='mean',
        seasonal_periods=4
    )
    
    inventory_model = ModifiedContinuousReview()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['lead_time_weeks', 'initial_inventory', 'service_level', 
                   'order_cost', 'cost', 'price', 'review_period', 
                   'holding_cost', 'stockout_cost']
    
    for col in numeric_cols:
        if col in master_data.columns:
            master_data[col] = pd.to_numeric(master_data[col], errors='coerce')
    
    # Convert QTY to numeric in historic data
    historic_data['QTY'] = pd.to_numeric(historic_data['QTY'], errors='coerce')
    
    skipped_skus = {
        'No Historical Data': [],
        'Insufficient Sales Data': [],
        'Processing Errors': []
    }
    
    # Calculate total number of batches
    total_skus = len(master_data)
    num_batches = (total_skus + batch_size - 1) // batch_size  # Ceiling division
    
    # Process SKUs in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_skus)
        
        batch_master_data = master_data.iloc[start_idx:end_idx]
        
        # Update progress bar for this batch
        batch_progress_start = batch_idx / num_batches
        batch_progress_end = (batch_idx + 1) / num_batches
        
        status_text.text(f"Processing batch {batch_idx + 1}/{num_batches} (SKUs {start_idx + 1}-{end_idx} of {total_skus})")
        
        batch_supply_plans = []  # Store supply plans for this batch
        
        # Process each SKU in the batch
        for i, (idx, sku_params) in enumerate(batch_master_data.iterrows()):
            try:
                # Calculate overall progress
                progress = batch_progress_start + (i + 1) / len(batch_master_data) * (batch_progress_end - batch_progress_start)
                progress_bar.progress(progress)
                status_text.text(f"Processing SKU {sku_params['SKU']} ({start_idx + i + 1}/{total_skus})")
                
                # Get SKU data
                sku_data = historic_data[historic_data['SKU'] == sku_params['SKU']].copy()
                
                # Skip if no historical data
                if len(sku_data) == 0:
                    skipped_skus['No Historical Data'].append(sku_params['SKU'])
                    continue
                    
                # Count non-zero sales points
                non_zero_sales = len(sku_data[sku_data['QTY'] > 0])
                if non_zero_sales < 2:
                    skipped_skus['Insufficient Sales Data'].append(
                        f"{sku_params['SKU']} (only {non_zero_sales} sales points)"
                    )
                    continue
                
                # Ensure all parameters are numeric
                params = {
                    'periods': int(periods),
                    'lead_time_weeks': int(sku_params['lead_time_weeks']),
                    'initial_inventory': int(sku_params['initial_inventory']),
                    'service_level': float(sku_params['service_level']),
                    'order_cost': float(sku_params['order_cost']),
                    'holding_cost': float(sku_params['holding_cost']),
                    'cost': float(sku_params['cost']),
                    'price': float(sku_params['price']),
                    'review_period': int(sku_params['review_period']),
                    'stockout_cost': float(sku_params['stockout_cost'])
                }
                
                # Get start time (first sale date)
                start_time = sku_data['Date'].min()
                
                # Run simulation
                sku = run_inventory_system(
                    sku_data, sku_params['SKU'], params,
                    forecast_model, inventory_model,
                    params['periods'], start_time
                )
                
                if sku is not None:
                    # Update seasonality counter if seasonality was extracted
                    if forecast_model.seasonality_extracted:
                        st.session_state.seasonality_extracted += 1
                    
                    # Collect results
                    results.append({
                        'SKU': sku_params['SKU'],
                        'Service Level': 1 - float(np.mean(sku.stockouts)),
                        'Average Inventory': float(np.mean(sku.inventory_evolution)),
                        'Total Cost': float(np.sum(sku.inventory_evolution * params['holding_cost'] + 
                                           sku.order_evolution * params['cost'])),
                        'Total Sales': float(np.sum(np.minimum(sku.inventory_evolution, sku.demand_evolution) * 
                                            params['price'])),
                        'Stockout Rate': float(np.mean(sku.stockouts)),
                        'Inventory Turnover': float(np.sum(sku.demand_evolution) / 
                                            np.mean(sku.inventory_evolution)) if np.mean(sku.inventory_evolution) > 0 else 0,
                        'EOQ': float(sku.inventory_policy.get('eoq', 0)),
                        'Reorder Point': float(sku.inventory_policy.get('reorder_point', 0)),
                        'Safety Stock': float(sku.inventory_policy.get('safety_stock', 0)),
                        'Seasonality Extracted': forecast_model.seasonality_extracted
                    })
                    
                    # Collect supply plan
                    supply_plan = pd.DataFrame({
                        'Date': sku.forecast['mean'].index,
                        'SKU': sku_params['SKU'],
                        'Forecast Demand': sku.forecast['mean'].values,
                        'Order Quantity': sku.order_evolution,
                        'Orders Arriving': sku.orders_arriving,
                        'Orders in Transit': sku.orders_in_transit,
                        'Inventory Level': sku.inventory_evolution,
                        'Stockouts': sku.stockouts
                    })
                    
                    # Optimize memory usage of the supply plan
                    supply_plan, _ = optimize_dataframe_memory(supply_plan)
                    batch_supply_plans.append(supply_plan)
                
            except Exception as e:
                skipped_skus['Processing Errors'].append(f"{sku_params['SKU']}: {str(e)}")
                continue
        
        # Combine batch supply plans and add to main list
        if batch_supply_plans:
            batch_combined = pd.concat(batch_supply_plans, axis=0)
            supply_plans.append(batch_combined)
            
        # Clear memory after each batch
        import gc
        gc.collect()
    
    progress_bar.empty()
    status_text.empty()
    
    # Display summary of skipped SKUs if any
    if any(skipped_skus.values()):
        with st.expander("⚠️ Summary of Skipped SKUs", expanded=False):
            for category, skus in skipped_skus.items():
                if skus:
                    st.markdown(f"**{category}** ({len(skus)} SKUs)")
                    for sku in skus:
                        st.markdown(f"- {sku}")
    
    if not results:
        st.error("No SKUs were processed successfully")
        return pd.DataFrame(), pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Optimize results dataframe
    results_df, results_reduction = optimize_dataframe_memory(results_df)
    
    # Combine all supply plans
    if supply_plans:
        supply_plans_df = pd.concat(supply_plans, axis=0)
        supply_plans_df, supply_reduction = optimize_dataframe_memory(supply_plans_df)
        st.info(f"Memory optimization: Results data reduced by {results_reduction:.1f}%, Supply plan data reduced by {supply_reduction:.1f}%")
    else:
        supply_plans_df = pd.DataFrame()
    
    return results_df, supply_plans_df

def plot_global_analysis(results_df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Service Level Distribution', 'Inventory Turnover Distribution',
                       'Cost vs Service Level', 'Sales vs Inventory')
    )
    
    fig.add_trace(
        go.Histogram(x=results_df['Service Level'], name="Service Level"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=results_df['Inventory Turnover'], name="Inventory Turnover"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=results_df['Service Level'], y=results_df['Total Cost'],
                  mode='markers', name="Cost vs Service Level"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=results_df['Average Inventory'], y=results_df['Total Sales'],
                  mode='markers', name="Sales vs Inventory"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Global Analysis Dashboard")
    return fig

def generate_excel_report(results_df, supply_plan_df, master_data):
    # Calculate Syntetos-Boylan categories for each SKU
    categorization_results = []
    historic_data = st.session_state.historic_data
    
    for sku in master_data['SKU']:
        sku_data = historic_data[historic_data['SKU'] == sku].copy()
        
        # Skip if no data
        if len(sku_data) == 0:
            categorization_results.append({
                'SKU': sku,
                'Category': 'Insufficient Data'
            })
            continue
            
        # Get non-zero demands
        non_zero_demands = sku_data[sku_data['QTY'] > 0]['QTY']
        total_periods = len(sku_data)
        
        if len(non_zero_demands) < 2:
            categorization_results.append({
                'SKU': sku,
                'Category': 'Insufficient Data'
            })
            continue
        
        # Calculate ADI
        adi = total_periods / len(non_zero_demands)
        
        # Calculate CV²
        cv2 = (non_zero_demands.std() / non_zero_demands.mean()) ** 2
        
        # Determine category
        if adi < 1.32:
            if cv2 < 0.49:
                category = 'Smooth'
            else:
                category = 'Erratic'
        else:
            if cv2 < 0.49:
                category = 'Intermittent'
            else:
                category = 'Lumpy'
        
        categorization_results.append({
            'SKU': sku,
            'Category': category
        })
    
    # Create DataFrame with categories
    categories_df = pd.DataFrame(categorization_results)
    
    # Merge categories with results
    results_df = results_df.merge(categories_df, on='SKU', how='left')
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary Results with categories
        results_df.to_excel(writer, sheet_name='Summary Results', index=False)
        
        # Supply Plan
        supply_plan_df.to_excel(writer, sheet_name='Supply Plan', index=False)
        
        # Parameters Used
        master_data.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Model Parameters
        model_params = pd.DataFrame({
            'Parameter': [
                'Forecast Model',
                'Forecast Zero Demand Strategy',
                'Forecast Rolling Window',
                'Forecast Zero Train Strategy',
                'Inventory Model',
                'Description'
            ],
            'Value': [
                'Normal Distribution',
                'rolling_mean',
                '4',
                'mean',
                'Modified Continuous Review',
                'The Modified Continuous Review model is used for inventory management. ' +
                'It combines features of continuous review with periodic adjustments. ' +
                'The forecast model uses Normal Distribution with rolling mean for zero demand handling.'
            ]
        })
        model_params.to_excel(writer, sheet_name='Model Parameters', index=False)
        
        # Create a template for future runs
        template = pd.DataFrame({
            'Parameter': ['Date', 'SKU', 'QTY'],
            'Description': [
                'Date of the transaction (YYYY-MM-DD)',
                'Stock Keeping Unit identifier',
                'Quantity of the transaction'
            ],
            'Required': ['Yes', 'Yes', 'Yes'],
            'Example': ['2024-01-01', 'SKU001', '100']
        })
        template.to_excel(writer, sheet_name='Historic Data Template', index=False)
        
        master_template = pd.DataFrame({
            'Parameter': master_data.columns,
            'Description': [
                'Stock Keeping Unit identifier',
                'Lead time in weeks',
                'Initial inventory level',
                'Target service level (0-1)',
                'Cost of placing an order',
                'Unit cost of the product',
                'Selling price per unit',
                'Review period in weeks',
                'Holding cost per unit per week (optional)',
                'Stockout cost per unit (optional)'
            ],
            'Required': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No'],
            'Example': ['SKU001', '2', '100', '0.95', '50', '100', '200', '1', '1', '20']
        })
        master_template.to_excel(writer, sheet_name='Master Data Template', index=False)
    
    processed_data = output.getvalue()
    return processed_data

def display_overall_metrics(results_df):
    st.subheader("Summary Metrics")
    
    # Summary metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average Service Level", f"{results_df['Service Level'].mean():.2%}")
    col2.metric("Average Inventory Turnover", f"{results_df['Inventory Turnover'].mean():.2f}")
    col3.metric("Total Cost", f"${results_df['Total Cost'].sum():,.2f}")
    col4.metric("Total Sales", f"${results_df['Total Sales'].sum():,.2f}")
    
    # Add seasonality success metric if available
    if 'seasonality_extracted' in st.session_state:
        success_rate = st.session_state.seasonality_extracted / len(results_df) * 100
        col5.metric("Seasonality Extraction", f"{success_rate:.1f}%", 
                   f"{st.session_state.seasonality_extracted}/{len(results_df)} SKUs")
    
    # Global analysis dashboard
    st.plotly_chart(plot_global_analysis(results_df), use_container_width=True)
    
    # Detailed results table with sorting and filtering
    st.subheader("Detailed Results")
    st.dataframe(
        results_df.style.format({
            'Service Level': '{:.2%}',
            'Average Inventory': '{:.1f}',
            'Total Cost': '${:,.2f}',
            'Total Sales': '${:,.2f}',
            'Stockout Rate': '{:.2%}',
            'Inventory Turnover': '{:.2f}',
            'EOQ': '{:.1f}',
            'Reorder Point': '{:.1f}',
            'Safety Stock': '{:.1f}'
        }),
        use_container_width=True
    )

def display_supply_demand_plan(supply_plan_df, results_df):
    st.subheader("Supply and Demand Analysis")
    
    # SKU filter
    selected_sku = st.selectbox(
        "Select SKU for detailed view",
        options=['All SKUs'] + list(results_df['SKU'].unique())
    )
    
    # Filter data based on selection
    if selected_sku != 'All SKUs':
        filtered_plan = supply_plan_df[supply_plan_df['SKU'] == selected_sku]
        filtered_results = results_df[results_df['SKU'] == selected_sku]
        title_suffix = f" - {selected_sku}"
    else:
        # When "All SKUs" is selected, limit to a subset of SKUs to prevent memory issues
        max_skus_to_display = 5  # Limit to 5 SKUs at a time
        
        if len(results_df['SKU'].unique()) > max_skus_to_display:
            # If there are more than max_skus_to_display, add pagination
            all_skus = sorted(results_df['SKU'].unique())
            
            # Add pagination controls
            col1, col2 = st.columns([3, 1])
            with col1:
                page_size = st.selectbox(
                    "SKUs per page:",
                    options=[5, 10, 20, 50],
                    index=0,
                    key="sku_page_size"
                )
            
            with col2:
                total_pages = (len(all_skus) + page_size - 1) // page_size
                page_number = st.number_input(
                    f"Page (1-{total_pages}):",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="sku_page_number"
                )
            
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(all_skus))
            
            selected_skus = all_skus[start_idx:end_idx]
            st.info(f"Showing SKUs {start_idx+1}-{end_idx} of {len(all_skus)}")
            
            filtered_plan = supply_plan_df[supply_plan_df['SKU'].isin(selected_skus)]
            filtered_results = results_df[results_df['SKU'].isin(selected_skus)]
            title_suffix = f" - Page {page_number}/{total_pages}"
        else:
            filtered_plan = supply_plan_df
            filtered_results = results_df
            title_suffix = " - All SKUs"
    
    # Check if the filtered data is too large
    if len(filtered_plan) > 100000:
        st.warning(f"The selected data contains {len(filtered_plan)} rows, which may cause memory issues. Consider selecting a specific SKU or reducing the date range.")
        
        # Sample the data to reduce memory usage
        filtered_plan = filtered_plan.sample(n=100000, random_state=42)
        st.info("Data has been sampled to 100,000 rows to improve performance.")
    
    # Aggregate data by date
    daily_plan = filtered_plan.groupby('Date').agg({
        'Forecast Demand': 'sum',
        'Order Quantity': 'sum',
        'Orders Arriving': 'sum',
        'Inventory Level': 'sum',
        'Stockouts': 'sum'
    }).reset_index()
    
    # Sort by date to ensure temporal correctness
    daily_plan = daily_plan.sort_values('Date')
    
    # Create combined plot
    fig = go.Figure()
    
    # Add Forecast Demand
    fig.add_trace(go.Scatter(
        x=daily_plan['Date'],
        y=daily_plan['Forecast Demand'],
        mode='lines',
        name='Forecast Demand',
        line=dict(dash='dash')
    ))
    
    # Add Inventory Level
    fig.add_trace(go.Scatter(
        x=daily_plan['Date'],
        y=daily_plan['Inventory Level'],
        mode='lines',
        name='Inventory Level',
        line=dict(color='blue')
    ))
    
    # Add Order Points as triangles
    order_indices = daily_plan['Order Quantity'] > 0
    if order_indices.any():
        fig.add_trace(go.Scatter(
            x=daily_plan.loc[order_indices, 'Date'],
            y=daily_plan.loc[order_indices, 'Inventory Level'],
            mode='markers',
            name='Order Points',
            marker=dict(
                color='green',
                symbol='triangle-up',
                size=12
            ),
            hovertemplate="Order Quantity: %{text}<br>Date: %{x}<br>Inventory Level: %{y}<extra></extra>",
            text=daily_plan.loc[order_indices, 'Order Quantity']
        ))
    
    # Add Orders Arriving as diamonds
    arriving_indices = daily_plan['Orders Arriving'] > 0
    if arriving_indices.any():
        fig.add_trace(go.Scatter(
            x=daily_plan.loc[arriving_indices, 'Date'],
            y=daily_plan.loc[arriving_indices, 'Orders Arriving'],
            mode='markers',
            name='Orders Arriving',
            marker=dict(
                color='purple',
                symbol='diamond',
                size=8
            ),
            hovertemplate="Arriving Quantity: %{y}<br>Date: %{x}<extra></extra>"
        ))
    
    # Add Reorder Point and Safety Stock if viewing a single SKU
    if selected_sku != 'All SKUs' and len(filtered_results) == 1:
        sku_policy = dict(zip(results_df['SKU'], 
                            zip(results_df['EOQ'], 
                                results_df['Reorder Point'],
                                results_df['Safety Stock']))).get(selected_sku)
        if sku_policy:
            _, reorder_point, safety_stock = sku_policy
            # Add Reorder Point line
            fig.add_trace(go.Scatter(
                x=daily_plan['Date'],
                y=[reorder_point] * len(daily_plan),
                mode='lines',
                name='Reorder Point',
                line=dict(dash='dot', color='red')
            ))
            # Add Safety Stock line
            fig.add_trace(go.Scatter(
                x=daily_plan['Date'],
                y=[safety_stock] * len(daily_plan),
                mode='lines',
                name='Safety Stock',
                line=dict(dash='dot', color='orange')
            ))
            
            # Add annotations for both lines
            fig.add_annotation(
                x=daily_plan['Date'].iloc[-1],
                y=reorder_point,
                text=f"Reorder Point: {reorder_point:.0f}",
                showarrow=False,
                yshift=10,
                xshift=10
            )
            fig.add_annotation(
                x=daily_plan['Date'].iloc[-1],
                y=safety_stock,
                text=f"Safety Stock: {safety_stock:.0f}",
                showarrow=False,
                yshift=-10,
                xshift=10
            )
    
    fig.update_layout(
        title=f"Supply and Demand Plan{title_suffix}",
        xaxis_title="Date",
        yaxis_title="Quantity",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600  # Make the plot taller
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Supply Plan Table with pagination
    st.subheader("Supply Plan Details")
    
    # Sort the data
    sorted_plan = filtered_plan.sort_values('Date')
    
    # Add pagination for the table
    rows_per_page = st.selectbox(
        "Rows per page:",
        options=[10, 25, 50, 100],
        index=1,  # Default to 25
        key="supply_plan_rows_per_page"
    )
    
    total_pages = (len(sorted_plan) + rows_per_page - 1) // rows_per_page
    
    if total_pages > 1:
        page_number = st.number_input(
            f"Page (1-{total_pages}):",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="supply_plan_page_number"
        )
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(sorted_plan))
        
        st.info(f"Showing rows {start_idx+1}-{end_idx} of {len(sorted_plan)}")
        display_plan = sorted_plan.iloc[start_idx:end_idx]
    else:
        display_plan = sorted_plan
    
    st.dataframe(
        display_plan.style.format({
            'Forecast Demand': '{:.0f}',
            'Order Quantity': '{:.0f}',
            'Orders Arriving': '{:.0f}',
            'Inventory Level': '{:.0f}',
            'Stockouts': '{:.0f}'
        }),
        use_container_width=True
    )

def display_forecast_analysis(results_df, supply_plan_df):
    st.subheader("Forecast Analysis")
    
    # Forecast Accuracy Metrics
    accuracy_metrics = pd.DataFrame({
        'SKU': results_df['SKU'],
        'MAPE': np.random.uniform(5, 15, len(results_df)),  # Replace with actual MAPE calculation
        'MAE': np.random.uniform(10, 30, len(results_df)),  # Replace with actual MAE calculation
        'RMSE': np.random.uniform(15, 40, len(results_df))  # Replace with actual RMSE calculation
    })
    
    # Display metrics
    st.markdown("### Forecast Accuracy Metrics")
    st.dataframe(
        accuracy_metrics.style.format({
            'MAPE': '{:.2f}%',
            'MAE': '{:.1f}',
            'RMSE': '{:.1f}'
        }),
        use_container_width=True
    )
    
    # Financial Impact Analysis
    st.markdown("### Financial Impact Analysis")
    financial_metrics = pd.DataFrame({
        'SKU': results_df['SKU'],
        'Revenue': results_df['Total Sales'],
        'Cost': results_df['Total Cost'],
        'Profit': results_df['Total Sales'] - results_df['Total Cost'],
        'Profit Margin': (results_df['Total Sales'] - results_df['Total Cost']) / results_df['Total Sales']
    })
    
    st.dataframe(
        financial_metrics.style.format({
            'Revenue': '${:,.2f}',
            'Cost': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Profit Margin': '{:.2%}'
        }),
        use_container_width=True
    )
    
    # Plot profit distribution
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=financial_metrics['Profit'],
        name='Profit Distribution'
    ))
    fig.update_layout(
        title="Profit Distribution Across SKUs",
        yaxis_title="Profit ($)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def display_download_section(results_df, supply_plan_df, master_data):
    st.subheader("Download Analysis Results")
    
    # Generate Excel report
    excel_data = generate_excel_report(results_df, supply_plan_df, master_data)
    
    # Add download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Complete Analysis",
            data=excel_data,
            file_name="multiple_sku_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download all analysis results including summary, supply plan, and parameters"
        )
    
    with col2:
        # Generate CSV for quick view
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📑 Download Summary (CSV)",
            data=csv,
            file_name="summary_results.csv",
            mime="text/csv",
            help="Download summary results in CSV format for quick viewing"
        )
    
    # Preview of what will be downloaded
    st.markdown("### Preview of Download Contents")
    
    tab1, tab2, tab3 = st.tabs(["Summary Results", "Supply Plan", "Parameters"])
    
    with tab1:
        st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.dataframe(supply_plan_df, use_container_width=True)
    
    with tab3:
        st.dataframe(master_data, use_container_width=True)

def display_weekly_analysis(supply_plan_df, results_df):
    st.subheader("Weekly Analysis and Decision Support")
    
    # SKU filter
    selected_sku = st.selectbox(
        "Select SKU for weekly analysis",
        options=['All SKUs'] + list(results_df['SKU'].unique()),
        key='weekly_sku'
    )
    
    # Filter data based on selection
    if selected_sku != 'All SKUs':
        filtered_plan = supply_plan_df[supply_plan_df['SKU'] == selected_sku]
        filtered_results = results_df[results_df['SKU'] == selected_sku]
        title_suffix = f" - {selected_sku}"
    else:
        # When "All SKUs" is selected, limit to a subset of SKUs to prevent memory issues
        max_skus_to_display = 5  # Limit to 5 SKUs at a time
        
        if len(results_df['SKU'].unique()) > max_skus_to_display:
            # If there are more than max_skus_to_display, add pagination
            all_skus = sorted(results_df['SKU'].unique())
            
            # Add pagination controls
            col1, col2 = st.columns([3, 1])
            with col1:
                page_size = st.selectbox(
                    "SKUs per page:",
                    options=[5, 10, 20, 50],
                    index=0,
                    key="weekly_sku_page_size"
                )
            
            with col2:
                total_pages = (len(all_skus) + page_size - 1) // page_size
                page_number = st.number_input(
                    f"Page (1-{total_pages}):",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="weekly_sku_page_number"
                )
            
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(all_skus))
            
            selected_skus = all_skus[start_idx:end_idx]
            st.info(f"Showing SKUs {start_idx+1}-{end_idx} of {len(all_skus)}")
            
            filtered_plan = supply_plan_df[supply_plan_df['SKU'].isin(selected_skus)]
            filtered_results = results_df[results_df['SKU'].isin(selected_skus)]
            title_suffix = f" - Page {page_number}/{total_pages}"
        else:
            filtered_plan = supply_plan_df
            filtered_results = results_df
            title_suffix = " - All SKUs"
    
    # Check if the filtered data is too large
    if len(filtered_plan) > 100000:
        st.warning(f"The selected data contains {len(filtered_plan)} rows, which may cause memory issues. Consider selecting a specific SKU or reducing the date range.")
        
        # Sample the data to reduce memory usage
        filtered_plan = filtered_plan.sample(n=100000, random_state=42)
        st.info("Data has been sampled to 100,000 rows to improve performance.")
    
    # Create a dictionary of inventory policies for quick lookup
    inventory_policies = dict(zip(results_df['SKU'], 
                                zip(results_df['EOQ'], 
                                    results_df['Reorder Point'],
                                    results_df['Safety Stock'])))
    
    # Group by week
    filtered_plan['Week'] = filtered_plan['Date'].dt.strftime('%Y-W%W')
    weekly_plan = filtered_plan.groupby(['Week', 'SKU']).agg({
        'Forecast Demand': 'sum',
        'Order Quantity': 'sum',
        'Orders Arriving': 'sum',
        'Inventory Level': 'mean',
        'Stockouts': 'sum'
    }).reset_index()
    
    # Calculate additional metrics with proper handling of edge cases
    weekly_plan['Stock Coverage (weeks)'] = np.where(
        weekly_plan['Forecast Demand'] > 0,
        weekly_plan['Inventory Level'] / weekly_plan['Forecast Demand'],
        np.where(weekly_plan['Inventory Level'] > 0, 999, 0)  # Use 999 for high inventory with no demand, 0 for no inventory
    )
    
    # Calculate recommended order quantity based on the inventory policy
    def get_order_decision(row):
        coverage = row['Stock Coverage (weeks)']
        sku_policy = inventory_policies.get(row['SKU'])
        
        if sku_policy is None:
            return '⚠️ No policy found'
            
        eoq, reorder_point, safety_stock = sku_policy
        
        if pd.isna(coverage) or coverage == 0:
            if row['Inventory Level'] == 0:
                return f'🔴 Place Order (EOQ: {eoq:.0f} units)'
            return '⚠️ Check Forecast'
            
        inventory_position = row['Inventory Level'] + row['Orders Arriving']
        
        if inventory_position <= reorder_point:
            return f'🔴 Place Order (EOQ: {eoq:.0f} units)'
        elif coverage < 2:
            return f'🟡 Monitor (Below target, ROP: {reorder_point:.0f})'
        else:
            return '🟢 Sufficient'

    weekly_plan['Order Decision'] = weekly_plan.apply(get_order_decision, axis=1)
    
    # Create weekly status dashboard
    st.markdown("### Weekly Status Dashboard")
    
    # Week selector
    weeks = sorted(weekly_plan['Week'].unique())
    selected_week = st.select_slider(
        "Select Week",
        options=weeks,
        value=weeks[0] if weeks else None
    )
    
    if selected_week:
        week_data = weekly_plan[weekly_plan['Week'] == selected_week]
        
        # Weekly metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Forecast Demand",
            f"{week_data['Forecast Demand'].sum():,.0f}",
            f"{week_data['Forecast Demand'].sum() - filtered_plan.groupby('Week')['Forecast Demand'].sum().shift(1).get(selected_week, 0):+,.0f}"
        )
        col2.metric(
            "Current Inventory",
            f"{week_data['Inventory Level'].mean():,.0f}",
            f"{week_data['Stock Coverage (weeks)'].mean():.1f} weeks"
        )
        col3.metric(
            "Orders Arriving",
            f"{week_data['Orders Arriving'].sum():,.0f}"
        )
        col4.metric(
            "Stockouts",
            f"{week_data['Stockouts'].sum():,.0f}"
        )
        
        # Decision support table with pagination
        st.markdown("### Weekly Decisions")
        
        # Get the decision table for the selected week
        decision_table = week_data[['SKU', 'Forecast Demand', 'Inventory Level', 
                                  'Stock Coverage (weeks)', 'Order Decision']].copy()
        
        # Format the Stock Coverage column
        decision_table['Stock Coverage (weeks)'] = decision_table['Stock Coverage (weeks)'].apply(
            lambda x: f"{x:.1f}" if x < 999 and x > 0 else "N/A" if x == 0 else ">999"
        )
        
        # Add pagination if there are many SKUs
        if len(decision_table) > 10:
            rows_per_page = st.selectbox(
                "Rows per page:",
                options=[5, 10, 20, 50],
                index=1,  # Default to 10
                key="weekly_decision_rows_per_page"
            )
            
            total_pages = (len(decision_table) + rows_per_page - 1) // rows_per_page
            
            page_number = st.number_input(
                f"Page (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="weekly_decision_page_number"
            )
            
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(decision_table))
            
            st.info(f"Showing rows {start_idx+1}-{end_idx} of {len(decision_table)}")
            display_table = decision_table.iloc[start_idx:end_idx]
        else:
            display_table = decision_table
        
        # Create a style function that only applies to the Order Decision column
        def style_order_decision(row):
            color_map = {
                '🔴': 'background-color: #ffcccc',
                '🟡': 'background-color: #ffffcc',
                '🟢': 'background-color: #ccffcc',
                '⚠️': 'background-color: #ffe5cc'
            }
            decision_icon = row['Order Decision'][0]  # Get the first character (emoji)
            return [color_map.get(decision_icon, '')] * len(row)
        
        # Apply styling with number formatting
        styled_table = display_table.style\
            .format({
                'Forecast Demand': '{:,.0f}',
                'Inventory Level': '{:,.0f}'
            })\
            .apply(style_order_decision, axis=1)
        
        st.dataframe(styled_table, use_container_width=True)
        
        # Recommendations - limit to critical items only
        st.markdown("### Recommendations")
        
        # Filter to show only critical items (red and yellow)
        critical_items = display_table[display_table['Order Decision'].str.contains('🔴|🟡|⚠️')]
        
        if len(critical_items) > 0:
            for _, row in critical_items.iterrows():
                sku_policy = inventory_policies.get(row['SKU'])
                if sku_policy:
                    eoq, reorder_point, safety_stock = sku_policy
                    if '🔴' in row['Order Decision']:
                        st.warning(
                            f"SKU {row['SKU']}: {row['Order Decision']}\n" +
                            f"- Current Inventory: {row['Inventory Level']:,.0f}\n" +
                            f"- Reorder Point: {reorder_point:,.0f}\n" +
                            f"- Safety Stock: {safety_stock:,.0f}"
                        )
                    elif '🟡' in row['Order Decision']:
                        st.info(
                            f"SKU {row['SKU']}: Stock coverage is moderate ({row['Stock Coverage (weeks)']} weeks).\n" +
                            f"- Current Inventory: {row['Inventory Level']:,.0f}\n" +
                            f"- Reorder Point: {reorder_point:,.0f}"
                        )
                    elif '⚠️' in row['Order Decision']:
                        st.warning(f"SKU {row['SKU']}: Unable to calculate stock coverage. Please check forecast and inventory data.")
        else:
            st.success("No critical items requiring immediate attention for this week.")
            
        # Clear memory
        import gc
        gc.collect()

def display_syntetos_categorization(historic_data, master_data):
    st.subheader("Syntetos-Boylan Categorization")
    
    # Add explanation
    st.markdown("""
    ### About Syntetos-Boylan Categorization
    
    This categorization helps identify demand patterns based on:
    
    - **ADI** (Average Demand Interval): Average time between non-zero demands
    - **CV²** (Squared Coefficient of Variation): Measures relative variability of non-zero demands
    
    Categories:
    - 🟦 **Smooth**: Regular demand with low variability (ADI < 1.32, CV² < 0.49)
    - 🟨 **Erratic**: Regular demand with high variability (ADI < 1.32, CV² ≥ 0.49)
    - 🟩 **Intermittent**: Irregular demand with low variability (ADI ≥ 1.32, CV² < 0.49)
    - 🟥 **Lumpy**: Irregular demand with high variability (ADI ≥ 1.32, CV² ≥ 0.49)
    - ⬜ **Insufficient Data**: Less than 2 demand points
    """)
    
    # Calculate metrics for each SKU
    categorization_results = []
    
    for sku in master_data['SKU']:
        sku_data = historic_data[historic_data['SKU'] == sku].copy()
        
        # Skip if no data
        if len(sku_data) == 0:
            categorization_results.append({
                'SKU': sku,
                'ADI': None,
                'CV2': None,
                'Category': 'Insufficient Data',
                'Non-zero Demands': 0,
                'Total Periods': 0,
                'Mean Demand': 0
            })
            continue
            
        # Get non-zero demands
        non_zero_demands = sku_data[sku_data['QTY'] > 0]['QTY']
        total_periods = len(sku_data)
        
        if len(non_zero_demands) < 2:
            categorization_results.append({
                'SKU': sku,
                'ADI': None,
                'CV2': None,
                'Category': 'Insufficient Data',
                'Non-zero Demands': len(non_zero_demands),
                'Total Periods': total_periods,
                'Mean Demand': non_zero_demands.mean() if len(non_zero_demands) > 0 else 0
            })
            continue
        
        # Calculate ADI
        adi = total_periods / len(non_zero_demands)
        
        # Calculate CV²
        cv2 = (non_zero_demands.std() / non_zero_demands.mean()) ** 2
        
        # Determine category
        if adi < 1.32:
            if cv2 < 0.49:
                category = 'Smooth'
            else:
                category = 'Erratic'
        else:
            if cv2 < 0.49:
                category = 'Intermittent'
            else:
                category = 'Lumpy'
        
        categorization_results.append({
            'SKU': sku,
            'ADI': adi,
            'CV2': cv2,
            'Category': category,
            'Non-zero Demands': len(non_zero_demands),
            'Total Periods': total_periods,
            'Mean Demand': non_zero_demands.mean()
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(categorization_results)
    
    # Display summary statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        n_smooth = len(results_df[results_df['Category'] == 'Smooth'])
        st.metric("🟦 Smooth", n_smooth)
        
    with col2:
        n_erratic = len(results_df[results_df['Category'] == 'Erratic'])
        st.metric("🟨 Erratic", n_erratic)
        
    with col3:
        n_intermittent = len(results_df[results_df['Category'] == 'Intermittent'])
        st.metric("🟩 Intermittent", n_intermittent)
        
    with col4:
        n_lumpy = len(results_df[results_df['Category'] == 'Lumpy'])
        st.metric("🟥 Lumpy", n_lumpy)
        
    with col5:
        n_insufficient = len(results_df[results_df['Category'] == 'Insufficient Data'])
        st.metric("⬜ Insufficient Data", n_insufficient)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add regions with valid CSS colors
    fig.add_shape(type="rect", x0=0, x1=1.32, y0=0, y1=0.49,
                  fillcolor="lightblue", opacity=0.3, line=dict(width=0))
    fig.add_shape(type="rect", x0=0, x1=1.32, y0=0.49, y1=max(results_df['CV2'].max()*1.1, 2),
                  fillcolor="lightyellow", opacity=0.3, line=dict(width=0))
    fig.add_shape(type="rect", x0=1.32, x1=max(results_df['ADI'].max()*1.1, 3), y0=0, y1=0.49,
                  fillcolor="lightgreen", opacity=0.3, line=dict(width=0))
    fig.add_shape(type="rect", x0=1.32, x1=max(results_df['ADI'].max()*1.1, 3), y0=0.49, y1=max(results_df['CV2'].max()*1.1, 2),
                  fillcolor="lightpink", opacity=0.3, line=dict(width=0))
    
    # Add points for each category
    categories = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']
    colors = ['royalblue', 'orange', 'forestgreen', 'red']
    
    for category, color in zip(categories, colors):
        mask = results_df['Category'] == category
        fig.add_trace(go.Scatter(
            x=results_df[mask]['ADI'],
            y=results_df[mask]['CV2'],
            mode='markers',
            name=category,
            marker=dict(color=color, size=8),
            text=results_df[mask]['SKU'],
            hovertemplate="SKU: %{text}<br>ADI: %{x:.2f}<br>CV²: %{y:.2f}"
        ))
    
    # Add lines for categorization
    fig.add_shape(type="line", x0=1.32, x1=1.32, y0=0, y1=max(results_df['CV2'].max()*1.1, 2),
                  line=dict(dash="dash", color="gray"))
    fig.add_shape(type="line", x0=0, x1=max(results_df['ADI'].max()*1.1, 3), y0=0.49, y1=0.49,
                  line=dict(dash="dash", color="gray"))
    
    fig.update_layout(
        title="Demand Pattern Categorization",
        xaxis_title="ADI (Average Demand Interval)",
        yaxis_title="CV² (Squared Coefficient of Variation)",
        showlegend=True,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed results table
    st.subheader("Detailed Results")
    
    # Format the DataFrame for display
    display_df = results_df.copy()
    display_df['ADI'] = display_df['ADI'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    display_df['CV2'] = display_df['CV2'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    display_df['Mean Demand'] = display_df['Mean Demand'].apply(lambda x: f"{x:.2f}")
    
    # Add color coding for categories
    def color_category(val):
        colors = {
            'Smooth': 'background-color: lightblue',
            'Erratic': 'background-color: lightyellow',
            'Intermittent': 'background-color: lightgreen',
            'Lumpy': 'background-color: lightpink',
            'Insufficient Data': 'background-color: lightgray'
        }
        return colors.get(val, '')
    
    styled_df = display_df.style.applymap(color_category, subset=['Category'])
    st.dataframe(styled_df, use_container_width=True)

def optimize_dataframe_memory(df):
    """
    Optimize memory usage of a DataFrame by downcasting numeric columns
    and converting object columns to categories when appropriate.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Process columns by dtype
    for col in df.columns:
        col_type = df[col].dtype
        
        # Numeric columns
        if col_type in ['int64', 'float64']:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Integer columns
            if col_type == 'int64':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Float columns
            elif col_type == 'float64':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        # Object columns (strings)
        elif col_type == 'object':
            # Convert to category if less than 50% unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem * 100
    
    return df, reduction

def main():
    # Initialize session state variables
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    
    display_logo_and_title()
    ()
    
    st.markdown("""
    ## Multiple SKU Analysis
    
    Upload an Excel file with two sheets:
    1. **Historic Data**: Sales history (Date, SKU, QTY)
    2. **Master Data**: Parameters for each SKU
    """)
    
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        # Process file only if it's new or hasn't been processed
        file_contents = uploaded_file.getvalue()
        current_file_hash = hash(file_contents)
        
        if not st.session_state.file_processed or \
           ('current_file_hash' not in st.session_state) or \
           (st.session_state.current_file_hash != current_file_hash):
            
            historic_data, master_data = load_data(uploaded_file)
            
            if historic_data is not None and master_data is not None:
                st.session_state.historic_data = historic_data
                st.session_state.master_data = master_data
                st.session_state.current_file_hash = current_file_hash
                st.session_state.file_processed = True
                st.success("File loaded successfully!")
        
        if st.session_state.file_processed:
            # Display data preview
            st.subheader("Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Historic Data**")
                st.dataframe(st.session_state.historic_data.head(), use_container_width=True)
            with col2:
                st.markdown("**Master Data**")
                st.dataframe(st.session_state.master_data.head(), use_container_width=True)
            
            # Analysis parameters
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get forecast periods
                periods = st.number_input("Forecast Periods (weeks)", min_value=1, value=52)
            
            with col2:
                # Add batch size control
                total_skus = len(st.session_state.master_data)
                batch_size = st.slider(
                    "SKUs per batch (lower for less memory usage)",
                    min_value=5,
                    max_value=min(50, total_skus),
                    value=min(20, total_skus),
                    step=5,
                    help="Process SKUs in smaller batches to reduce memory usage. Lower values use less memory but may take longer."
                )
            
            # Memory usage warning
            if total_skus > 50:
                st.warning(f"Your dataset contains {total_skus} SKUs, which may require significant memory. "
                          "If you encounter memory errors, try reducing the batch size.")
            
            # Create a placeholder for warnings
            warnings_placeholder = st.empty()
            
            if st.button("Run Analysis"):
                st.session_state.analysis_run = True
                
                with st.spinner("Running analysis for all SKUs..."):
                    # Clear previous warnings
                    warnings_placeholder.empty()
                    
                    # Run analysis within the warnings placeholder
                    with warnings_placeholder.container():
                        results_df, supply_plan_df = run_multiple_sku_analysis(
                            st.session_state.historic_data,
                            st.session_state.master_data,
                            periods,
                            batch_size
                        )
                    
                    # Store results in session state
                    st.session_state.results_df = results_df
                    st.session_state.supply_plan_df = supply_plan_df
                    st.session_state.show_results = True
            
            # Display results if they exist and should be shown
            if st.session_state.show_results and 'results_df' in st.session_state:
                results_df = st.session_state.results_df
                supply_plan_df = st.session_state.supply_plan_df
                
                if not results_df.empty:
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📊 Overall Metrics",
                        "📈 Supply & Demand Plan",
                        "🔮 Forecast Analysis",
                        "📅 Weekly Analysis",
                        "🎯 Categorization",
                        "💾 Download Results"
                    ])
                    
                    with tab1:
                        display_overall_metrics(results_df)
                    
                    with tab2:
                        display_supply_demand_plan(supply_plan_df, results_df)
                    
                    with tab3:
                        display_forecast_analysis(results_df, supply_plan_df)
                    
                    with tab4:
                        display_weekly_analysis(supply_plan_df, results_df)
                    
                    with tab5:
                        display_syntetos_categorization(st.session_state.historic_data, st.session_state.master_data)
                    
                    with tab6:
                        display_download_section(results_df, supply_plan_df, st.session_state.master_data)

if __name__ == "__main__":
    main() 