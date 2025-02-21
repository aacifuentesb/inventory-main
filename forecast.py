import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from abc import ABC, abstractmethod
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings



class ForecastModel(ABC):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        """
        Initialize forecast model with handling strategies
        
        Parameters:
        -----------
        zero_demand_strategy : str
            Strategy to handle zero demand values in forecast ('rolling_mean', 'global_mean', 'previous_value')
        rolling_window : int
            Window size for rolling mean calculation
        zero_train_strategy : str
            Strategy to handle zeros in training data ('mean', 'median', 'previous', 'next')
        use_mape : bool
            Whether to include MAPE in model evaluation
        strip_indices : bool
            Whether to strip indices before forecasting and add them back after
        """
        self.zero_demand_strategy = zero_demand_strategy
        self.rolling_window = rolling_window
        self.zero_train_strategy = zero_train_strategy
        self.use_mape = use_mape
        self.strip_indices = strip_indices

    def validate_data(self, data):
        """Validate and print information about the time series data"""
        print("\nTime Series Data Analysis:")
        print("-" * 50)
        print(f"Data Shape: {data.shape}")
        print(f"\nFirst few rows:")
        print(data.head())
        print(f"\nLast few rows:")
        print(data.tail())
        
        print("\nTime Series Properties:")
        print(f"Start date: {data.index.min()}")
        print(f"End date: {data.index.max()}")
        print(f"Number of weeks: {len(data)}")
        print(f"Frequency detected: {pd.infer_freq(data.index)}")
        
        print("\nData Statistics:")
        print(f"Mean value: {data.mean():.2f}")
        print(f"Std dev: {data.std():.2f}")
        print(f"Min value: {data.min():.2f}")
        print(f"Max value: {data.max():.2f}")
        print(f"Number of zeros: {(data == 0).sum()}")
        print(f"Number of negative values: {(data < 0).sum()}")
        
        # Calculate week-over-week changes
        weekly_changes = data.diff()
        print("\nWeek-over-Week Changes:")
        print(f"Mean change: {weekly_changes.mean():.2f}")
        print(f"Max increase: {weekly_changes.max():.2f}")
        print(f"Max decrease: {weekly_changes.min():.2f}")
        
        return len(data) >= 4  # Minimum required length for forecasting

    def handle_zero_demand(self, forecast_series):
        """Handle zero values in forecast based on selected strategy"""
        if self.zero_demand_strategy == 'rolling_mean':
            rolling_mean = forecast_series.rolling(window=self.rolling_window, min_periods=1).mean()
            forecast_series = forecast_series.where(forecast_series != 0, rolling_mean)
        elif self.zero_demand_strategy == 'global_mean':
            global_mean = forecast_series.mean()
            forecast_series = forecast_series.where(forecast_series != 0, global_mean)
        elif self.zero_demand_strategy == 'previous_value':
            forecast_series = forecast_series.replace(0, method='ffill')
            # If first value is 0, use next non-zero value
            if forecast_series.iloc[0] == 0:
                forecast_series = forecast_series.replace(0, method='bfill')
        
        return forecast_series

    def prepare_data(self, data):
        """Prepare data for forecasting by optionally stripping indices"""
        if self.strip_indices:
            # Save original index information
            self._original_freq = pd.infer_freq(data.index)
            self._last_date = data.index[-1]
            # Return values only
            return data.values
        return data

    def create_forecast_index(self, periods):
        """Create date index for forecast results"""
        if self.strip_indices:
            return pd.date_range(
                start=self._last_date + pd.Timedelta(days=7),
                periods=periods,
                freq=self._original_freq or 'W'
            )
        return None

    def format_forecast(self, forecast_values, periods):
        """Format forecast results with proper index"""
        if self.strip_indices:
            date_range = self.create_forecast_index(periods)
            if isinstance(forecast_values, dict):
                return {
                    'mean': pd.Series(forecast_values['mean'], index=date_range),
                    'lower': pd.Series(forecast_values['lower'], index=date_range),
                    'upper': pd.Series(forecast_values['upper'], index=date_range)
                }
            else:
                return pd.Series(forecast_values, index=date_range)
        return forecast_values

    def calculate_metrics(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        metrics = {
            'mae': np.mean(np.abs(actual - predicted)),
            'rmse': np.sqrt(np.mean((actual - predicted) ** 2)),
        }
        
        if self.use_mape and np.all(actual != 0):
            metrics['mape'] = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return metrics

    @abstractmethod
    def forecast(self, data, periods):
        """
        Generate forecast
        
        Parameters:
        -----------
        data : pd.Series
            Historical data
        periods : int
            Number of periods to forecast
            
        Returns:
        --------
        dict
            Dictionary containing 'mean', 'lower', and 'upper' forecasts
        """
        # Implementation in derived classes should use:
        # 1. data = self.prepare_data(data)
        # 2. ... generate forecast ...
        # 3. return self.format_forecast(forecast_result, periods)
        pass

    @staticmethod
    def get_description():
        return ""

    def get_forecast_metrics(self, data, test_size=0.3):
        """
        Calculate forecast metrics by splitting the data into train and test sets.
        
        :param data: pandas Series with datetime index
        :param test_size: float, proportion of data to use as test set
        :return: dict with metrics
        """
        train_size = int(len(data) * (1 - test_size))
        train, test = data[:train_size], data[train_size:]
        
        # Forecast for the test period
        forecast_results = self.forecast(train, len(test))
        forecast = forecast_results['mean']

        # Align forecast index with test data index
        forecast = pd.Series(forecast.values, index=test.index)

        # Calculate metrics
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mase = mae / np.mean(np.abs(np.diff(train)))
        # agg_diff = percentage difference in total sales between the forecast and the actual
        aggregated_diff = (np.abs((np.sum(test) - np.sum(forecast))) / np.sum(test))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MASE': mase,
            'MAPE': aggregated_diff

        }
        
       
class ExponentialSmoothingForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)

    def forecast(self, data, periods):
        try:
            model = ExponentialSmoothing(data, seasonal_periods=52, trend='add', seasonal='add')
        except:
            model = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='add')
        fit = model.fit()
        forecast = fit.forecast(periods)
        
        # Handle zero values
        forecast = self.handle_zero_demand(forecast)
        
        # Calculate confidence intervals manually
        resid = fit.resid
        sigma = np.sqrt(np.sum(resid**2) / (len(resid) - 1))
        lower = forecast - 1.96 * sigma
        upper = forecast + 1.96 * sigma

        # Make forecast, lower, and upper non-negative
        forecast = np.maximum(forecast, 0)
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)

        # Round up to the nearest integer
        forecast = np.round(forecast)
        lower = np.round(lower)
        upper = np.round(upper)
         
        return {'mean': forecast, 'lower': lower, 'upper': upper}

    @staticmethod
    def get_description():
        return '''
        Exponential Smoothing forecast is a popular method for time series forecasting.
        It is a generalization of simple exponential smoothing. 
        Mathematical Formulation:
        The model is defined by the following equations:
        St = αYt + (1-α)St-1
        Lt = β(St - St-1) + (1-β)Lt-1
        Ft+m = St + mLt
        where:
        St: smoothed observation at time t
        Lt: smoothed trend at time t
        Ft+m: forecast m periods ahead
        α: smoothing parameter for the level
        β: smoothing parameter for the trend
        The model is fit to the data and used to make forecasts.
        '''
    

class NormalDistributionForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)

    def forecast(self, data, periods):
        # Calculate mean and standard deviation
        mean = np.mean(data)
        std = np.std(data)

        # Generate multiple forecasts and take the best one
        num_attempts = 10
        best_forecast = None
        min_zeros = float('inf')

        for _ in range(num_attempts):
            # Generate forecast
            forecast = np.random.normal(mean, std, periods)
            
            # Make non-negative first
            forecast = np.maximum(forecast, 0)
            
            # Count zeros before handling them
            zero_count = np.sum(forecast == 0)
            
            if zero_count < min_zeros:
                min_zeros = zero_count
                best_forecast = forecast

        # Convert to series for zero handling
        last_date = data.index[-1]
        date_range = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=periods, freq='W')
        forecast_series = pd.Series(best_forecast, index=date_range)
        
        # Handle any remaining zeros
        forecast_series = self.handle_zero_demand(forecast_series)
        
        # Calculate confidence intervals
        confidence = 0.95
        degrees_of_freedom = len(data) - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
        margin_of_error = t_value * (std / np.sqrt(len(data)))

        lower = forecast_series - margin_of_error
        upper = forecast_series + margin_of_error

        # Make non-negative and round
        forecast_series = np.round(forecast_series)
        lower = np.maximum(np.round(lower), 0)
        upper = np.maximum(np.round(upper), 0)

        return {
            'mean': forecast_series,
            'lower': pd.Series(lower, index=date_range),
            'upper': pd.Series(upper, index=date_range)
        }

    @staticmethod
    def get_description():
        return '''
        Normal Distribution forecast generates random numbers from a normal distribution.
        The mean and standard deviation are calculated from the historical data.
        The forecast is generated by sampling from the normal distribution.
        '''


class SeasonalNormalDistributionForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True,
                 seasonal_periods=4):  # 4 weeks per month
        """
        Initialize Seasonal Normal Distribution forecast model
        
        Parameters:
        -----------
        seasonal_periods : int
            Number of weeks to consider as one seasonal cycle (default 4 for monthly)
        """
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)
        self.seasonal_periods = seasonal_periods
        self.seasonality_extracted = False  # Track if seasonality was successfully extracted

    def extract_seasonality(self, data):
        """
        Extract seasonal patterns from the data
        
        Parameters:
        -----------
        data : pd.Series
            Historical time series data
            
        Returns:
        --------
        seasonal_factors : np.array
            Seasonal factors for each week in the seasonal cycle
        """
        try:
            print("\nAttempting to extract seasonality...")
            
            # Ensure we have enough data for decomposition
            if len(data) < 2 * self.seasonal_periods:
                print("Insufficient data for seasonality extraction")
                self.seasonality_extracted = False
                return np.ones(self.seasonal_periods)
            
            # Handle zeros and missing values
            data_for_decomp = data.copy()
            non_zero_mean = data_for_decomp[data_for_decomp > 0].mean()
            
            if pd.isna(non_zero_mean) or non_zero_mean == 0:
                print("No valid non-zero values found")
                self.seasonality_extracted = False
                return np.ones(self.seasonal_periods)
            
            # Replace zeros with small values
            small_value = max(non_zero_mean * 0.1, 0.1)
            data_for_decomp = data_for_decomp.replace(0, small_value)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(data_for_decomp)):
                print("Data contains infinite values")
                self.seasonality_extracted = False
                return np.ones(self.seasonal_periods)
            
            print(f"Data range: {data_for_decomp.min():.2f} to {data_for_decomp.max():.2f}")
            
            try:
                # Perform seasonal decomposition with error handling
                decomposition = seasonal_decompose(
                    data_for_decomp,
                    period=self.seasonal_periods,
                    model='additive'
                )
                
                # Extract seasonal component
                seasonal = decomposition.seasonal
                if seasonal is None or len(seasonal) == 0:
                    print("Seasonal decomposition failed to produce seasonal component")
                    self.seasonality_extracted = False
                    return np.ones(self.seasonal_periods)
                
                # Calculate average seasonal factors for each period
                seasonal_factors = np.zeros(self.seasonal_periods)
                for i in range(self.seasonal_periods):
                    factors = seasonal[i::self.seasonal_periods]
                    if len(factors) > 0:
                        seasonal_factors[i] = np.mean(factors)
                
                # Convert additive factors to multiplicative
                base_level = np.mean(data_for_decomp)
                multiplicative_factors = (base_level + seasonal_factors) / base_level
                
                # Ensure factors are positive and finite
                multiplicative_factors = np.clip(multiplicative_factors, 0.1, 10.0)
                
                # Normalize factors
                multiplicative_factors = multiplicative_factors / np.mean(multiplicative_factors)
                
                # Check if we found significant seasonality
                variation = np.std(multiplicative_factors)
                print(f"Seasonal variation: {variation:.3f}")
                
                if variation > 0.05:  # 5% threshold
                    print("✅ Seasonality successfully extracted")
                    self.seasonality_extracted = True
                else:
                    print("No significant seasonality found")
                    self.seasonality_extracted = False
                    return np.ones(self.seasonal_periods)
                
                return multiplicative_factors
                
            except Exception as e:
                print(f"Error during decomposition: {str(e)}")
                self.seasonality_extracted = False
                return np.ones(self.seasonal_periods)
            
        except Exception as e:
            print(f"Error in seasonality extraction: {str(e)}")
            self.seasonality_extracted = False
            return np.ones(self.seasonal_periods)

    def forecast(self, data, periods):
        try:
            print("\nStarting Seasonal Normal Distribution forecast...")
            
            # Reset seasonality flag
            self.seasonality_extracted = False
            
            # Extract seasonal factors
            seasonal_factors = self.extract_seasonality(data)
            print(f"Seasonal factors detected: {seasonal_factors}")
            print(f"Seasonality extracted: {self.seasonality_extracted}")
            
            # Calculate mean and standard deviation of deseasonalized data
            deseasonalized_data = data.values / np.tile(
                seasonal_factors, 
                len(data) // self.seasonal_periods + 1
            )[:len(data)]
            
            mean = np.mean(deseasonalized_data)
            std = np.std(deseasonalized_data)
            
            print(f"Base distribution - Mean: {mean:.2f}, Std: {std:.2f}")
            
            # Generate multiple forecasts and take the best one
            num_attempts = 10
            best_forecast = None
            min_zeros = float('inf')
            
            for attempt in range(num_attempts):
                # Generate base forecast
                base_forecast = np.random.normal(mean, std, periods)
                
                # Apply seasonality
                seasonal_indices = np.arange(periods) % self.seasonal_periods
                forecast = base_forecast * seasonal_factors[seasonal_indices]
                
                # Make non-negative
                forecast = np.maximum(forecast, 0)
                
                # Count zeros
                zero_count = np.sum(forecast == 0)
                
                if zero_count < min_zeros:
                    min_zeros = zero_count
                    best_forecast = forecast
            
            # Convert to series for zero handling
            last_date = data.index[-1]
            date_range = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=periods,
                freq='W'
            )
            forecast_series = pd.Series(best_forecast, index=date_range)
            
            # Handle any remaining zeros
            forecast_series = self.handle_zero_demand(forecast_series)
            
            # Calculate confidence intervals considering seasonality
            confidence = 0.95
            degrees_of_freedom = len(data) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
            
            # Adjust margin of error for seasonality
            seasonal_indices = np.arange(periods) % self.seasonal_periods
            seasonal_std = std * seasonal_factors[seasonal_indices]
            margin_of_error = t_value * (seasonal_std / np.sqrt(len(data)))
            
            lower = forecast_series - margin_of_error
            upper = forecast_series + margin_of_error
            
            # Make non-negative and round
            forecast_series = np.round(forecast_series)
            lower = np.maximum(np.round(lower), 0)
            upper = np.maximum(np.round(upper), 0)
            
            return {
                'mean': forecast_series,
                'lower': pd.Series(lower, index=date_range),
                'upper': pd.Series(upper, index=date_range)
            }
            
        except Exception as e:
            print(f"\n❌ Seasonal Normal Distribution forecast failed: {str(e)}")
            raise ValueError(f"Seasonal Normal Distribution forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Seasonal Normal Distribution Forecast: A sophisticated forecasting model that combines 
        statistical distribution with seasonal pattern recognition.
        
        Core Components:
        1. Seasonal Pattern Detection
           - Uses advanced decomposition techniques
           - Identifies monthly patterns in weekly data
           - Adapts to varying seasonal strengths
        
        2. Statistical Foundation
           - Based on Normal (Gaussian) distribution
           - Incorporates mean and variance from historical data
           - Adjusts for seasonal variations
        
        3. Advanced Features
           - Automatic seasonality detection and validation
           - Multiple forecast generation with zero-handling
           - Seasonally-adjusted confidence intervals
           - Robust handling of intermittent demand
        
        Best Used For:
        ✓ Products with regular seasonal patterns
        ✓ Medium to long-term forecasting (4-52 weeks)
        ✓ Items with moderate to high sales volume
        ✓ Retail and consumer goods
        
        Advantages:
        + Captures both level and seasonal components
        + Provides reliable confidence intervals
        + Handles zero values intelligently
        + Adapts to changing seasonal patterns
        
        Limitations:
        - Requires sufficient historical data (>8 weeks)
        - May not capture sudden trend changes
        - Assumes relatively stable seasonal patterns
        
        Technical Details:
        • Seasonality Detection: Uses additive decomposition
        • Distribution: Normal with seasonal adjustments
        • Confidence Intervals: t-distribution based
        • Zero Handling: Multiple strategies available
        '''


class ARIMAForecast(ForecastModel):
    def forecast(self, data, periods):
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        df['unique_id'] = "Inventory"
        model = StatsForecast(models=[AutoARIMA(season_length=4)], freq='W')
        model.fit(df)
        forecast_df = model.predict(h=periods, level=[95])

        new_df = forecast_df.copy()
        new_df = new_df[['ds','AutoARIMA','AutoARIMA-lo-95','AutoARIMA-hi-95']]
        new_df.columns = ['ds','mean','lower','upper']
        new_df.set_index('ds', inplace=True)
        
        # Handle zero values
        new_df['mean'] = self.handle_zero_demand(new_df['mean'])
        
        new_df['mean'] = new_df['mean'].apply(lambda x: max(0,x))
        new_df['lower'] = new_df['lower'].apply(lambda x: max(0,x))
        new_df['upper'] = new_df['upper'].apply(lambda x: max(0,x))

        # Round up to the nearest integer
        new_df['mean'] = np.round(new_df['mean'])
        new_df['lower'] = np.round(new_df['lower'])
        new_df['upper'] = np.round(new_df['upper'])

        return {
            'mean': new_df['mean'], 
            'lower': new_df['lower'], 
            'upper': new_df['upper']
        }


    @staticmethod
    def get_description():
        # Detailed description of the model
        return '''
        ARIMA model is a popular statistical method for time series forecasting.
        It is a generalization of the simpler AutoRegressive Moving Average (ARMA) model.
        Mathematical Formulation:
        ARIMA(p, d, q) model is defined by three parameters:
        p: The number of lag observations included in the model, also called the lag order.
        d: The number of times that the raw observations are differenced, also called the degree of differencing.
        q: The size of the moving average window, also called the order of moving average.

        The model is fit to the data and used to make forecasts.
        '''
     

class HoltWintersAdvancedForecast(ForecastModel):
    def forecast(self, data, periods):
        try:
            print("\nStarting Holt-Winters forecast...")
            print(f"Input data shape: {data.shape}")
            print(f"Forecasting {periods} periods ahead")
            
            # Try to detect seasonality
            if len(data) >= 52:
                seasonal_periods = 52  # Weekly data, annual seasonality
            elif len(data) >= 13:
                seasonal_periods = 13  # Quarterly seasonality
            else:
                seasonal_periods = 4   # Monthly seasonality
            
            print(f"Using seasonal period: {seasonal_periods}")

            # Create a copy of data for fitting (replace zeros with mean for fitting only)
            fit_data = data.copy()
            if (fit_data == 0).any():
                mean_value = fit_data[fit_data > 0].mean()
                print(f"Found {(fit_data == 0).sum()} zeros in data. Temporarily replacing with mean ({mean_value:.2f}) for model fitting.")
                fit_data = fit_data.replace(0, mean_value)

            # Fit multiple models with different parameters and select the best one
            models = [
                ExponentialSmoothing(
                    fit_data, 
                    seasonal_periods=seasonal_periods,
                    trend=trend,
                    seasonal=seasonal,
                    damped=damped
                )
                for trend, seasonal, damped in [
                    ('add', 'add', True),
                    ('add', 'add', False),
                    ('add', None, True),  # No seasonality option
                    ('add', None, False)  # No seasonality option
                ]
            ]

            best_aic = float('inf')
            best_model = None
            best_fit = None
            best_config = None

            for i, model in enumerate(models):
                try:
                    config = ['add-add-damped', 'add-add', 'add-none-damped', 'add-none'][i]
                    print(f"Trying Holt-Winters configuration: {config}")
                    fit = model.fit(optimized=True, remove_bias=True)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_model = model
                        best_fit = fit
                        best_config = config
                        print(f"New best model found: {config} (AIC: {fit.aic:.2f})")
                except Exception as e:
                    print(f"Configuration {config} failed: {str(e)}")
                    continue

            if best_fit is None:
                print("❌ Could not fit any Holt-Winters model to the data")
                raise ValueError("No Holt-Winters model could be fit to the data")

            print(f"✅ Selected configuration: {best_config}")

            # Generate forecast
            raw_forecast = best_fit.forecast(periods)
            print(f"Raw forecast shape: {raw_forecast.shape}")
            print("Raw forecast values:", raw_forecast[:5], "...")  # Print first few values
            
            # Create proper date range for forecast
            date_range = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=7),
                periods=periods, 
                freq='W'
            )
            print(f"Created date range: {date_range[0]} to {date_range[-1]}")
            
            # Convert forecast to Series with proper index and handle NaN values
            forecast = pd.Series(raw_forecast, index=date_range)
            forecast = forecast.fillna(method='ffill')  # Forward fill any NaN values
            if forecast.isna().any():  # If any NaN values remain, fill with mean
                forecast = forecast.fillna(forecast.mean())
            print(f"Forecast series shape: {forecast.shape}")
            
            # Handle zero values in forecast
            forecast = self.handle_zero_demand(forecast)
            
            # Calculate confidence intervals using residual bootstrap
            resid = best_fit.resid
            sigma = np.sqrt(np.sum(resid**2) / (len(resid) - 1))
            
            # Generate confidence intervals
            lower = forecast - 1.96 * sigma
            upper = forecast + 1.96 * sigma
            
            # Make non-negative and round
            forecast = np.maximum(np.round(forecast), 0)
            lower = np.maximum(np.round(lower), 0)
            upper = np.maximum(np.round(upper), 0)
            
            result = {
                'mean': forecast,
                'lower': pd.Series(lower, index=date_range),
                'upper': pd.Series(upper, index=date_range)
            }
            
            print("\nForecast Summary:")
            print(f"Mean forecast range: {result['mean'].min():.2f} to {result['mean'].max():.2f}")
            print(f"Forecast length: {len(result['mean'])}")
            print("First few forecasted values:", result['mean'].head().values)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Holt-Winters forecast failed: {str(e)}")
            raise ValueError(f"Holt-Winters forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Advanced Holt-Winters forecasting with automatic model selection.
        
        Key features:
        - Automatic seasonality detection
        - Multiple model variations tested
        - Handles zero values by temporary replacement
        - Model selection based on AIC
        - Bootstrap-based confidence intervals
        - Integer-based forecasting
        
        The model automatically selects the best combination of:
        - Trend: additive
        - Seasonality: additive or none
        - Damping: with or without
        Based on the data characteristics.
        '''


class SeasonalARIMAForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)
        
    def forecast(self, data, periods):
        try:
            print("\nStarting Seasonal ARIMA forecast...")
            
            # Prepare data
            data_values = self.prepare_data(data)
            
            # Create DataFrame for statsforecast
            df = pd.DataFrame({
                'ds': range(len(data_values)),  # Use numeric index
                'y': data_values,
                'unique_id': "Inventory"
            })
            
            # Initialize AutoARIMA with seasonal settings
            model = AutoARIMA(
                season_length=52,  # Weekly data
                start_p=1,
                start_q=1,
                max_p=2,
                max_q=2,
                max_P=1,
                max_Q=1,
                max_order=4,
                max_d=1,
                max_D=1,
                stepwise=True,
                nmodels=10,
                seasonal=True
            )
            
            # Create StatsForecast instance
            sf = StatsForecast(
                models=[model],
                freq='W',
                n_jobs=-1  # Use all available cores
            )
            
            # Fit and predict
            sf.fit(df)
            forecast = sf.predict(h=periods, level=[95])
            
            # Process forecast
            forecast_df = forecast.copy()
            forecast_df = forecast_df[['ds', 'AutoARIMA', 'AutoARIMA-lo-95', 'AutoARIMA-hi-95']]
            forecast_df.columns = ['ds', 'mean', 'lower', 'upper']
            forecast_df.set_index('ds', inplace=True)
            
            # Handle zero values
            forecast_df['mean'] = self.handle_zero_demand(forecast_df['mean'])
            
            # Make non-negative and round
            for col in ['mean', 'lower', 'upper']:
                forecast_df[col] = np.maximum(np.round(forecast_df[col]), 0)
            
            # Create proper date index
            date_range = self.create_forecast_index(periods)
            
            return {
                'mean': pd.Series(forecast_df['mean'].values, index=date_range),
                'lower': pd.Series(forecast_df['lower'].values, index=date_range),
                'upper': pd.Series(forecast_df['upper'].values, index=date_range)
            }
            
        except Exception as e:
            print(f"\n❌ Seasonal ARIMA forecast failed: {str(e)}")
            raise ValueError(f"Seasonal ARIMA forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Seasonal ARIMA (AutoRegressive Integrated Moving Average) with automatic parameter selection.
        
        Features:
        - Automatically selects best ARIMA parameters
        - Handles weekly seasonality (52 periods)
        - Uses stepwise selection for efficiency
        - Provides confidence intervals
        - Handles both trend and seasonality
        
        Good for:
        - Data with clear seasonal patterns
        - Series with trends
        - Medium to long-term forecasting
        '''


class EnsembleForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)
        self.models = [
            HoltWintersAdvancedForecast(),
            SeasonalARIMAForecast(),
            ExponentialSmoothingForecast()
        ]
        
    def forecast(self, data, periods):
        if not self.validate_data(data):
            warnings.warn("❌ Insufficient data for Ensemble forecast")
            raise ValueError("Insufficient data for forecasting")
            
        try:
            # Convert data to array for consistent handling
            data_values = data.values
            date_index = data.index
            
            # Get forecasts from all models
            forecasts = []
            for model in self.models:
                try:
                    # Create temporary series with simple index for model
                    temp_data = pd.Series(data_values, index=range(len(data_values)))
                    forecast = model.forecast(temp_data, periods)
                    # Extract just the values from the forecast
                    forecast_values = forecast['mean'].values
                    forecasts.append(forecast_values)
                    print(f"Successfully added forecast from {model.__class__.__name__}")
                except Exception as e:
                    print(f"Failed to get forecast from {model.__class__.__name__}: {str(e)}")
                    continue
            
            if not forecasts:
                raise ValueError("No models produced valid forecasts")
            
            # Stack forecasts and calculate ensemble
            stacked_forecasts = np.vstack(forecasts)
            
            # Calculate weighted average based on recent performance
            weights = np.ones(len(forecasts)) / len(forecasts)  # Equal weights initially
            
            # Calculate ensemble forecast
            ensemble_forecast = np.average(stacked_forecasts, axis=0, weights=weights)
            
            # Calculate confidence intervals
            forecast_std = np.std(stacked_forecasts, axis=0)
            lower = ensemble_forecast - 1.96 * forecast_std
            upper = ensemble_forecast + 1.96 * forecast_std
            
            # Create date range for results
            date_range = pd.date_range(
                start=date_index[-1] + pd.Timedelta(days=7),
                periods=periods, 
                freq='W'
            )
            
            # Convert back to series with proper dates
            forecast_series = pd.Series(ensemble_forecast, index=date_range)
            forecast_series = self.handle_zero_demand(forecast_series)
            
            # Make non-negative and round
            forecast_series = np.maximum(np.round(forecast_series), 0)
            lower = np.maximum(np.round(lower), 0)
            upper = np.maximum(np.round(upper), 0)
            
            return {
                'mean': forecast_series,
                'lower': pd.Series(lower, index=date_range),
                'upper': pd.Series(upper, index=date_range)
            }
            
        except Exception as e:
            print(f"Ensemble forecast failed: {str(e)}")
            raise ValueError(f"Ensemble forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Ensemble Forecast combines multiple forecasting methods:
        - Holt-Winters Advanced
        - Seasonal ARIMA
        - Exponential Smoothing
        
        Key features:
        - Combines strengths of multiple models
        - Robust to individual model failures
        - Provides balanced predictions
        - Works with simple arrays for computation
        - Adds dates only at the end
        '''


class MovingAverageTrendForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)

    def forecast(self, data, periods):
        try:
            print("\nStarting Moving Average with Trend forecast...")
            
            # Prepare data
            data_values = self.prepare_data(data)
            
            # Calculate moving average (last 8 weeks)
            ma_window = min(8, len(data_values))
            ma = pd.Series(data_values).rolling(window=ma_window, min_periods=1).mean()
            
            # Calculate trend from moving average
            trend = (ma.iloc[-1] - ma.iloc[-ma_window]) / ma_window
            print(f"Detected trend: {trend:.2f} units per period")
            
            # Generate forecast
            last_ma = ma.iloc[-1]
            forecast_values = np.array([last_ma + trend * i for i in range(1, periods + 1)])
            
            # Calculate prediction intervals based on historical volatility
            volatility = pd.Series(data_values).pct_change().std()
            std_dev = forecast_values * volatility
            
            forecast_result = {
                'mean': forecast_values,
                'lower': forecast_values - (1.96 * std_dev),
                'upper': forecast_values + (1.96 * std_dev)
            }
            
            # Make non-negative and round
            for key in forecast_result:
                forecast_result[key] = np.maximum(np.round(forecast_result[key]), 0)
            
            # Format with proper index
            return self.format_forecast(forecast_result, periods)
            
        except Exception as e:
            print(f"\n❌ Moving Average forecast failed: {str(e)}")
            raise ValueError(f"Moving Average forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Moving Average with Trend Forecast
        
        A simple but effective method that:
        1. Calculates 8-week moving average
        2. Estimates trend from recent data
        3. Projects forward using trend
        4. Adjusts confidence intervals based on historical volatility
        
        Good for stable demand patterns with gradual trends.
        '''


class SimpleExpSmoothingDrift(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True):
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)

    def forecast(self, data, periods):
        try:
            print("\nStarting Simple Exponential Smoothing with Drift forecast...")
            
            # Calculate exponential smoothing (alpha = 0.3 for stability)
            alpha = 0.3
            smoothed = pd.Series(index=data.index, dtype=float)
            smoothed.iloc[0] = data.iloc[0]
            
            for t in range(1, len(data)):
                smoothed.iloc[t] = alpha * data.iloc[t] + (1 - alpha) * smoothed.iloc[t-1]
            
            # Calculate drift (average change)
            drift = (data.iloc[-1] - data.iloc[0]) / (len(data) - 1)
            print(f"Detected drift: {drift:.2f} units per period")
            
            # Generate forecast
            last_smoothed = smoothed.iloc[-1]
            forecast_values = np.array([last_smoothed + drift * i for i in range(1, periods + 1)])
            
            # Create date range
            date_range = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=7),
                periods=periods, 
                freq='W'
            )
            
            # Convert to series and handle zeros
            forecast = pd.Series(forecast_values, index=date_range)
            forecast = self.handle_zero_demand(forecast)
            
            # Calculate prediction intervals
            errors = data - smoothed
            rmse = np.sqrt(np.mean(errors ** 2))
            error_growth = np.sqrt(np.arange(1, periods + 1))  # Error grows with horizon
            
            lower = forecast - (1.96 * rmse * error_growth)
            upper = forecast + (1.96 * rmse * error_growth)
            
            # Make non-negative and round
            forecast = np.maximum(np.round(forecast), 0)
            lower = np.maximum(np.round(lower), 0)
            upper = np.maximum(np.round(upper), 0)
            
            return {
                'mean': forecast,
                'lower': pd.Series(lower, index=date_range),
                'upper': pd.Series(upper, index=date_range)
            }
            
        except Exception as e:
            print(f"\n❌ Simple Exponential Smoothing forecast failed: {str(e)}")
            raise ValueError(f"Simple Exponential Smoothing forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Simple Exponential Smoothing with Drift
        
        Combines exponential smoothing with drift adjustment:
        1. Applies exponential smoothing (α = 0.3)
        2. Calculates long-term drift
        3. Projects forward with drift adjustment
        4. Provides growing prediction intervals
        
        Good for data with both level changes and long-term trends.
        '''


class CrostonForecast(ForecastModel):
    def __init__(self, zero_demand_strategy='rolling_mean', rolling_window=4, 
                 zero_train_strategy='mean', use_mape=True, strip_indices=True,
                 alpha=0.1, beta=0.1):
        """
        Initialize Croston's method for intermittent demand forecasting.
        
        Parameters:
        -----------
        alpha : float
            Smoothing parameter for demand sizes (0 < alpha < 1)
        beta : float
            Smoothing parameter for intervals (0 < beta < 1)
        """
        super().__init__(zero_demand_strategy, rolling_window, 
                        zero_train_strategy, use_mape, strip_indices)
        self.alpha = alpha
        self.beta = beta

    def croston_method(self, data, periods):
        """
        Implementation of Croston's method for intermittent demand.
        
        Assumptions:
        1. Non-zero demands are independent and normally distributed
        2. Demand intervals are independent and geometrically distributed
        3. Demand sizes and intervals are mutually independent
        """
        # Ensure data is numpy array
        demand = np.asarray(data)
        n = len(demand)
        
        # Initialize arrays for demand sizes and intervals
        q = np.zeros(n)  # Demand sizes
        p = np.zeros(n)  # Intervals between demands
        
        # Find first non-zero demand for initialization
        first_idx = np.where(demand > 0)[0][0]
        q[first_idx] = demand[first_idx]
        p[first_idx] = first_idx + 1
        
        # Process historical data
        last_demand_idx = first_idx
        for t in range(first_idx + 1, n):
            if demand[t] > 0:
                # Update demand size estimate
                q[t] = self.alpha * demand[t] + (1 - self.alpha) * q[last_demand_idx]
                # Update interval estimate
                interval = t - last_demand_idx
                p[t] = self.beta * interval + (1 - self.beta) * p[last_demand_idx]
                last_demand_idx = t
            else:
                q[t] = q[last_demand_idx]
                p[t] = p[last_demand_idx]
        
        # Generate forecast
        final_q = q[last_demand_idx]  # Last demand size estimate
        final_p = p[last_demand_idx]  # Last interval estimate
        
        # Calculate point forecast (demand rate)
        point_forecast = final_q / final_p
        
        # Generate forecast for future periods
        forecast = np.array([point_forecast] * periods)
        
        # Calculate prediction intervals
        # Using historical non-zero demand variability
        non_zero_demands = demand[demand > 0]
        std_dev = np.std(non_zero_demands)
        cv = std_dev / np.mean(non_zero_demands) if len(non_zero_demands) > 0 else 1
        
        # Wider intervals for longer horizons
        error_growth = np.sqrt(np.arange(1, periods + 1))
        lower = forecast * (1 - 1.96 * cv * error_growth)
        upper = forecast * (1 + 1.96 * cv * error_growth)
        
        return forecast, lower, upper

    def forecast(self, data, periods):
        try:
            print("\nStarting Croston forecast...")
            
            # Prepare data
            data_values = self.prepare_data(data)
            
            # Generate forecast
            forecast_values, lower_values, upper_values = self.croston_method(data_values, periods)
            
            # Create date range
            date_range = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=7),
                periods=periods, 
                freq='W'
            )
            
            # Convert to series
            forecast_series = pd.Series(forecast_values, index=date_range)
            lower_series = pd.Series(lower_values, index=date_range)
            upper_series = pd.Series(upper_values, index=date_range)
            
            # Make non-negative and round
            forecast_series = np.maximum(np.round(forecast_series), 0)
            lower_series = np.maximum(np.round(lower_series), 0)
            upper_series = np.maximum(np.round(upper_series), 0)
            
            return {
                'mean': forecast_series,
                'lower': lower_series,
                'upper': upper_series
            }
            
        except Exception as e:
            print(f"\n❌ Croston forecast failed: {str(e)}")
            raise ValueError(f"Croston forecast failed: {str(e)}")

    @staticmethod
    def get_description():
        return '''
        Croston's method for intermittent demand forecasting.
        
        Key features:
        - Specifically designed for intermittent demand
        - Separates demand size from intervals
        - Uses two smoothing parameters:
          * α for demand sizes
          * β for intervals between demands
        - Provides growing prediction intervals
        - Handles zero values appropriately
        
        Assumptions:
        1. Non-zero demands are independent and normally distributed
        2. Demand intervals are independent and geometrically distributed
        3. Demand sizes and intervals are mutually independent
        
        Best for:
        - Spare parts demand
        - Irregular ordering patterns
        - Many zero-demand periods
        '''
