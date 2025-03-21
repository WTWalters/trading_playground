import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import the required modules from our system
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from src.market_analysis.parameter_management.risk_manager import RiskManager
from src.market_analysis.parameter_management.integration import ParameterIntegration
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer
from src.market_analysis.parameter_management.risk_controls import AdaptiveRiskControls

class MarketRegimeSimulator:
    """Generates synthetic market data with different regime characteristics for testing."""

    def __init__(self, seed=42):
        """Initialize the simulator with a random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)

    def generate_regime_data(self, regime_type, length=200, start_date=None):
        """
        Generate synthetic price data for a specific market regime.

        Parameters:
        -----------
        regime_type : str
            The type of regime to simulate ('low_vol', 'high_vol', 'trending', 'mean_reverting', 'crisis')
        length : int
            Number of data points to generate
        start_date : datetime
            Starting date for the time series

        Returns:
        --------
        pd.DataFrame
            DataFrame with synthetic market data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=length)

        dates = [start_date + timedelta(days=i) for i in range(length)]

        # Base parameters
        price_level = 100.0
        prices = [price_level]

        # Generate price data based on regime type
        if regime_type == 'low_vol':
            volatility = 0.005
            for _ in range(1, length):
                price_change = np.random.normal(0, volatility)
                prices.append(prices[-1] * (1 + price_change))

        elif regime_type == 'high_vol':
            volatility = 0.025
            for _ in range(1, length):
                price_change = np.random.normal(0, volatility)
                prices.append(prices[-1] * (1 + price_change))

        elif regime_type == 'trending':
            trend = 0.002  # Daily upward trend
            volatility = 0.008
            for _ in range(1, length):
                price_change = trend + np.random.normal(0, volatility)
                prices.append(prices[-1] * (1 + price_change))

        elif regime_type == 'mean_reverting':
            mean_level = price_level
            reversion_strength = 0.05
            volatility = 0.01
            for _ in range(1, length):
                reversion = reversion_strength * (mean_level - prices[-1]) / prices[-1]
                price_change = reversion + np.random.normal(0, volatility)
                prices.append(prices[-1] * (1 + price_change))

        elif regime_type == 'crisis':
            # Start with normal, then crash, then high volatility recovery
            volatility_normal = 0.008
            crash_start = int(length * 0.3)
            crash_end = int(length * 0.5)

            for i in range(1, length):
                if i < crash_start:
                    # Normal period
                    price_change = np.random.normal(0.0005, volatility_normal)
                elif i < crash_end:
                    # Crash period - strong negative bias with high volatility
                    crash_intensity = -0.015
                    price_change = np.random.normal(crash_intensity, 0.03)
                else:
                    # Recovery period - high volatility with slight positive bias
                    price_change = np.random.normal(0.001, 0.025)

                prices.append(prices[-1] * (1 + price_change))

        else:
            raise ValueError(f"Unknown regime type: {regime_type}")

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': [0] + [(prices[i]/prices[i-1] - 1) for i in range(1, len(prices))],
            'regime': regime_type
        })

        df.set_index('date', inplace=True)
        return df

    def generate_transition_data(self, regime_sequence, regime_lengths, start_date=None):
        """
        Generate synthetic data that transitions between different market regimes.

        Parameters:
        -----------
        regime_sequence : list
            List of regime types to transition between
        regime_lengths : list
            List of lengths for each regime period
        start_date : datetime
            Starting date for the time series

        Returns:
        --------
        pd.DataFrame
            DataFrame with synthetic market data including regime transitions
        """
        if len(regime_sequence) != len(regime_lengths):
            raise ValueError("regime_sequence and regime_lengths must have the same length")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=sum(regime_lengths))

        all_data = []
        current_date = start_date

        for i, (regime, length) in enumerate(zip(regime_sequence, regime_lengths)):
            regime_data = self.generate_regime_data(regime, length, current_date)
            all_data.append(regime_data)
            current_date = current_date + timedelta(days=length)

        # Combine all regime data
        combined_data = pd.concat(all_data)

        return combined_data

    def generate_correlation_breakdown(self, length=300, correlation_shift_point=150, start_date=None):
        """
        Generate synthetic pair data where correlation breaks down midway.

        Parameters:
        -----------
        length : int
            Total length of the data series
        correlation_shift_point : int
            Point at which correlation starts to break down
        start_date : datetime
            Starting date for the time series

        Returns:
        --------
        tuple of pd.DataFrame
            Two DataFrames representing asset pairs with correlation breakdown
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=length)

        dates = [start_date + timedelta(days=i) for i in range(length)]

        # Common market factor
        market_factor = np.cumsum(np.random.normal(0.0005, 0.01, length))

        # Asset-specific factors
        asset1_specific = np.cumsum(np.random.normal(0, 0.005, length))
        asset2_specific = np.cumsum(np.random.normal(0, 0.005, length))

        # Before breakdown - assets are highly correlated
        asset1_before = 100 * np.exp(0.7 * market_factor[:correlation_shift_point] +
                                     0.3 * asset1_specific[:correlation_shift_point])
        asset2_before = 50 * np.exp(0.7 * market_factor[:correlation_shift_point] +
                                    0.3 * asset2_specific[:correlation_shift_point])

        # After breakdown - correlation significantly reduces
        asset1_after = asset1_before[-1] * np.exp(0.3 * market_factor[correlation_shift_point:] +
                                                 0.7 * asset1_specific[correlation_shift_point:])
        asset2_after = asset2_before[-1] * np.exp(0.7 * market_factor[correlation_shift_point:] +
                                                 0.7 * asset2_specific[correlation_shift_point:])

        # Combine before and after
        asset1_prices = np.concatenate([asset1_before, asset1_after])
        asset2_prices = np.concatenate([asset2_before, asset2_after])

        # Create DataFrames
        asset1_df = pd.DataFrame({
            'date': dates,
            'price': asset1_prices,
            'returns': [0] + [(asset1_prices[i]/asset1_prices[i-1] - 1) for i in range(1, len(asset1_prices))]
        })

        asset2_df = pd.DataFrame({
            'date': dates,
            'price': asset2_prices,
            'returns': [0] + [(asset2_prices[i]/asset2_prices[i-1] - 1) for i in range(1, len(asset2_prices))]
        })

        asset1_df.set_index('date', inplace=True)
        asset2_df.set_index('date', inplace=True)

        # Add a correlation column calculated over rolling windows
        returns1 = asset1_df['returns']
        returns2 = asset2_df['returns']

        # Calculate rolling 30-day correlation
        rolling_corr = pd.Series(
            [returns1.iloc[max(0, i-30):i+1].corr(returns2.iloc[max(0, i-30):i+1])
             if i >= 5 else np.nan for i in range(len(returns1))],
            index=asset1_df.index
        )

        asset1_df['correlation'] = rolling_corr
        asset2_df['correlation'] = rolling_corr

        # Add regime indicator
        asset1_df['regime'] = 'correlated'
        asset2_df['regime'] = 'correlated'

        asset1_df.loc[asset1_df.index[correlation_shift_point:], 'regime'] = 'decorrelated'
        asset2_df.loc[asset2_df.index[correlation_shift_point:], 'regime'] = 'decorrelated'

        return asset1_df, asset2_df


class TestRegimeTransition(unittest.TestCase):
    """Test the system's ability to adapt to different market regimes and transitions."""

    def setUp(self):
        """Set up the testing environment."""
        self.simulator = MarketRegimeSimulator(seed=42)
        self.output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize our adaptive parameter management components
        self.regime_detector = EnhancedRegimeDetector(
            lookback_period=30,
            volatility_threshold_high=0.015,
            volatility_threshold_low=0.008
        )

        self.risk_controls = AdaptiveRiskControls(
            default_max_position_size=0.1,
            default_max_pair_exposure=0.2,
            default_stop_loss=0.05
        )

        self.position_sizer = KellyPositionSizer(
            default_fraction=0.5,
            max_kelly_fraction=0.8,
            min_kelly_fraction=0.1
        )

        self.risk_manager = RiskManager(
            risk_controls=self.risk_controls,
            position_sizer=self.position_sizer
        )

        self.parameter_integration = ParameterIntegration(
            regime_detector=self.regime_detector,
            risk_manager=self.risk_manager
        )

    def test_regime_detection_accuracy(self):
        """Test if the system correctly identifies different market regimes."""
        # Generate data for different regimes
        regimes = ['low_vol', 'high_vol', 'trending', 'mean_reverting', 'crisis']
        for regime in regimes:
            with self.subTest(regime=regime):
                # Generate regime-specific data
                data = self.simulator.generate_regime_data(regime, length=120)

                # Run the regime detector
                detected_regimes = []
                for i in range(30, len(data)):
                    window = data.iloc[i-30:i]
                    detected_regime = self.regime_detector.detect_regime(window['returns'])
                    detected_regimes.append(detected_regime)

                # Check the most frequently detected regime matches the expected
                from collections import Counter
                regime_counts = Counter(detected_regimes)
                most_common_regime = regime_counts.most_common(1)[0][0]

                # Mapping between simulated regime types and detector regime types
                regime_mapping = {
                    'low_vol': 'normal',
                    'high_vol': 'volatile',
                    'trending': 'trending',
                    'mean_reverting': 'mean_reverting',
                    'crisis': 'volatile'  # Crisis should be detected as volatile
                }

                expected_regime = regime_mapping[regime]

                # Verify detection (allowing some flexibility)
                if regime == 'crisis':
                    # For crisis regime, we expect a mix of volatile and trending regimes
                    common_regimes = [r[0] for r in regime_counts.most_common(2)]
                    self.assertTrue(
                        'volatile' in common_regimes,
                        f"Crisis regime not properly detected as volatile. Got: {common_regimes}"
                    )
                else:
                    # For other regimes, check if the most common detection matches expected
                    self.assertEqual(
                        most_common_regime, expected_regime,
                        f"Expected {expected_regime} regime, got {most_common_regime}"
                    )

    def test_parameter_adaptation(self):
        """Test if the system adapts parameters correctly for different regimes."""
        # Generate data for different regimes
        regimes = ['low_vol', 'high_vol', 'trending', 'mean_reverting']

        for regime in regimes:
            with self.subTest(regime=regime):
                # Generate regime-specific data
                data = self.simulator.generate_regime_data(regime, length=120)

                # Set up initial parameters
                initial_params = {
                    'max_position_size': 0.1,
                    'stop_loss': 0.05,
                    'kelly_fraction': 0.5
                }

                # Run through the data and track parameter changes
                param_history = []

                for i in range(30, len(data)):
                    window = data.iloc[i-30:i]

                    # Get adapted parameters
                    params = self.parameter_integration.adapt_parameters(
                        window['returns'],
                        window['returns'].std(),
                        0.8,  # dummy sharpe ratio
                        0.6,  # dummy win rate
                        initial_params
                    )

                    param_history.append(params)

                # Check if parameters were adapted appropriately
                avg_position_size = np.mean([p['max_position_size'] for p in param_history])
                avg_stop_loss = np.mean([p['stop_loss'] for p in param_history])

                # Verify parameter adaptation logic
                if regime == 'high_vol' or regime == 'crisis':
                    # High volatility should reduce position size and tighten stops
                    self.assertLess(
                        avg_position_size, initial_params['max_position_size'],
                        f"Position size not reduced for {regime} regime"
                    )
                    self.assertLess(
                        avg_stop_loss, initial_params['stop_loss'],
                        f"Stop loss not tightened for {regime} regime"
                    )
                elif regime == 'low_vol':
                    # Low volatility might allow larger positions
                    self.assertGreaterEqual(
                        avg_position_size, initial_params['max_position_size'],
                        f"Position size not maintained or increased for {regime} regime"
                    )

    def test_regime_transition_adaptation(self):
        """Test if the system adapts smoothly during regime transitions."""
        # Generate data with regime transitions
        regime_sequence = ['low_vol', 'high_vol', 'low_vol']
        regime_lengths = [60, 60, 60]

        transition_data = self.simulator.generate_transition_data(
            regime_sequence, regime_lengths
        )

        # Initialize parameters
        params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5
        }

        # Run through the data and track parameter changes
        param_history = []
        detected_regimes = []

        for i in range(30, len(transition_data)):
            window = transition_data.iloc[i-30:i]

            # Detect regime
            regime = self.regime_detector.detect_regime(window['returns'])
            detected_regimes.append(regime)

            # Get adapted parameters
            params = self.parameter_integration.adapt_parameters(
                window['returns'],
                window['returns'].std(),
                0.8,  # dummy sharpe ratio
                0.6,  # dummy win rate
                params  # Use the current parameters
            )

            param_history.append(params)

        # Convert to DataFrame for analysis
        param_df = pd.DataFrame(param_history, index=transition_data.index[30:])
        param_df['detected_regime'] = detected_regimes
        param_df['actual_regime'] = transition_data.iloc[30:]['regime'].values

        # Check for smooth transitions
        # Calculate the day-to-day change in parameters
        param_df['position_size_change'] = param_df['max_position_size'].diff().abs()

        # Maximum allowed daily parameter change (avoid abrupt changes)
        max_allowed_daily_change = 0.03

        excessive_changes = param_df['position_size_change'] > max_allowed_daily_change
        self.assertLessEqual(
            excessive_changes.sum(), 3,  # Allow up to 3 instances of larger changes
            "Too many abrupt parameter changes during regime transitions"
        )

        # Verify that parameters adapt in the expected direction during transitions
        # Extract parameters from each regime period
        low_vol_1_params = param_df[param_df['actual_regime'] == 'low_vol'].iloc[:30]
        high_vol_params = param_df[param_df['actual_regime'] == 'high_vol']
        low_vol_2_params = param_df[param_df['actual_regime'] == 'low_vol'].iloc[-30:]

        # Check that position sizes decrease during high volatility
        self.assertGreater(
            low_vol_1_params['max_position_size'].mean(),
            high_vol_params['max_position_size'].mean(),
            "Position size not properly reduced during transition to high volatility"
        )

        # Check that position sizes increase when volatility decreases
        self.assertGreater(
            low_vol_2_params['max_position_size'].mean(),
            high_vol_params['max_position_size'].mean(),
            "Position size not properly increased during transition to low volatility"
        )

        # Create visualization of the transition
        plt.figure(figsize=(12, 8))

        # Plot 1: Price and Regime
        plt.subplot(3, 1, 1)
        plt.plot(transition_data.index, transition_data['price'])

        # Shade backgrounds by actual regime
        regime_change_points = transition_data['regime'].ne(transition_data['regime'].shift()).cumsum()
        for regime_id in regime_change_points.unique():
            regime_data = transition_data[regime_change_points == regime_id]
            start, end = regime_data.index[0], regime_data.index[-1]
            color = 'lightgreen' if regime_data['regime'].iloc[0] == 'low_vol' else 'lightcoral'
            plt.axvspan(start, end, color=color, alpha=0.3)

        plt.title('Price Movement with Regime Transitions')
        plt.ylabel('Price')

        # Plot 2: Parameter Changes
        plt.subplot(3, 1, 2)
        plt.plot(param_df.index, param_df['max_position_size'], label='Position Size')
        plt.plot(param_df.index, param_df['stop_loss'], label='Stop Loss')
        plt.title('Parameter Adaptation')
        plt.ylabel('Parameter Value')
        plt.legend()

        # Plot 3: Detected Regimes
        plt.subplot(3, 1, 3)

        # Create a numeric representation of detected regimes
        regime_map = {'normal': 0, 'volatile': 1, 'trending': 2, 'mean_reverting': 3}
        regime_numeric = [regime_map.get(r, -1) for r in detected_regimes]

        plt.plot(param_df.index, regime_numeric, 'o-')
        plt.yticks([0, 1, 2, 3], ['Normal', 'Volatile', 'Trending', 'Mean Reverting'])
        plt.title('Detected Market Regimes')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_transition_test.png'))
        plt.close()

        return param_df  # Return for use in other tests

    def test_performance_maintenance(self):
        """
        Test if the system maintains at least 60% of its performance during regime transitions
        compared to steady-state performance in each regime.
        """
        # First, measure performance in stable regimes
        regimes = ['low_vol', 'high_vol']
        stable_performance = {}

        # Simple synthetic strategy performance calculation
        def calculate_performance(data, params_history):
            """Calculate synthetic trading performance based on regime detection and parameter adaptation."""
            # This is a simplified performance model assuming:
            # - Better parameters in the right regime yield better performance
            # - Performance is higher when position sizing is appropriate for volatility
            returns = data['returns'].iloc[30:].values
            volatility = np.array([r['volatility'] if 'volatility' in r else 0.01 for r in params_history])
            position_sizes = np.array([p['max_position_size'] for p in params_history])
            stop_losses = np.array([p['stop_loss'] for p in params_history])

            # Synthetic performance model:
            # - Rewards lower position sizes during high volatility
            # - Rewards higher position sizes during low volatility
            # - Penalties for mismatched risk management

            vol_adjusted_pos_size = position_sizes * (0.01 / np.maximum(volatility, 0.001))
            risk_reward_ratio = position_sizes / np.maximum(stop_losses, 0.01)

            # Weighted performance score
            performance = returns * vol_adjusted_pos_size * (1.0 - np.abs(risk_reward_ratio - 2.0) * 0.1)

            # Return cumulative performance
            return np.cumsum(performance)[-1]

        # Measure performance in stable regimes
        for regime in regimes:
            # Generate stable regime data
            data = self.simulator.generate_regime_data(regime, length=120)

            # Initialize parameters
            params = {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'kelly_fraction': 0.5
            }

            # Run through the data and get adapted parameters
            param_history = []

            for i in range(30, len(data)):
                window = data.iloc[i-30:i]

                # Get adapted parameters
                current_volatility = window['returns'].std() * np.sqrt(252)  # Annualized

                params = self.parameter_integration.adapt_parameters(
                    window['returns'],
                    current_volatility,
                    0.8,  # dummy sharpe ratio
                    0.6,  # dummy win rate
                    params
                )

                params['volatility'] = current_volatility
                param_history.append(params)

            # Calculate performance with optimal parameters
            performance = calculate_performance(data, param_history)
            stable_performance[regime] = performance

        # Now test performance during transitions
        regime_sequence = ['low_vol', 'high_vol', 'low_vol']
        regime_lengths = [60, 60, 60]

        transition_data = self.simulator.generate_transition_data(
            regime_sequence, regime_lengths
        )

        # Initialize parameters
        params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5
        }

        # Run through the transition data
        param_history = []

        for i in range(30, len(transition_data)):
            window = transition_data.iloc[i-30:i]
            current_volatility = window['returns'].std() * np.sqrt(252)

            # Get adapted parameters
            params = self.parameter_integration.adapt_parameters(
                window['returns'],
                current_volatility,
                0.8,  # dummy sharpe ratio
                0.6,  # dummy win rate
                params
            )

            params['volatility'] = current_volatility
            param_history.append(params)

        # Split the transition data by regime
        transition_regimes = {}
        for regime in regimes:
            mask = transition_data.iloc[30:]['regime'] == regime
            transition_regimes[regime] = {
                'data': transition_data.iloc[30:][mask],
                'params': [param_history[i] for i, m in enumerate(mask) if m]
            }

        # Calculate performance during each regime in transition
        transition_performance = {}
        for regime, data in transition_regimes.items():
            if len(data['data']) > 0:  # Ensure there's data for this regime
                performance = calculate_performance(data['data'], data['params'])
                transition_performance[regime] = performance

        # Check if performance maintained at least 60% during transitions
        for regime in regimes:
            if regime in transition_performance:
                performance_ratio = transition_performance[regime] / stable_performance[regime]

                self.assertGreaterEqual(
                    performance_ratio, 0.6,
                    f"Performance during {regime} transition below 60% of stable performance. "
                    f"Ratio: {performance_ratio:.2f}"
                )

                print(f"{regime} regime - Performance maintenance ratio: {performance_ratio:.2f}")

        # Create visualization of performance comparison
        plt.figure(figsize=(10, 6))

        # Bar chart of performance comparison
        regimes_to_plot = [r for r in regimes if r in transition_performance]
        stable_values = [stable_performance[r] for r in regimes_to_plot]
        transition_values = [transition_performance[r] for r in regimes_to_plot]

        x = np.arange(len(regimes_to_plot))
        width = 0.35

        plt.bar(x - width/2, stable_values, width, label='Stable Regime')
        plt.bar(x + width/2, transition_values, width, label='During Transition')

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xticks(x, regimes_to_plot)
        plt.ylabel('Performance')
        plt.title('Performance Comparison: Stable vs. Transition')
        plt.legend()

        # Add percentage labels
        for i, (stable, trans) in enumerate(zip(stable_values, transition_values)):
            ratio = trans / stable if stable != 0 else 0
            plt.annotate(f"{ratio:.0%}",
                         xy=(i + width/2, trans + 0.01),
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_maintenance_test.png'))
        plt.close()

    def test_correlation_breakdown(self):
        """Test if the system detects and adapts to correlation breakdowns in pairs."""
        # Generate synthetic pair data with correlation breakdown
        asset1_data, asset2_data = self.simulator.generate_correlation_breakdown(
            length=300, correlation_shift_point=150
        )

        # Calculate spread/ratio for the pair
        # For simplicity, we'll use a simple ratio of prices
        spread_data = asset1_data.copy()
        spread_data['spread'] = asset1_data['price'] / asset2_data['price']
        spread_data['spread_returns'] = spread_data['spread'].pct_change()

        # Run the pair through our adaptive system
        params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'max_pair_exposure': 0.2
        }

        param_history = []
        correlation_history = []

        # Process data with a sliding window
        window_size = 30
        for i in range(window_size, len(spread_data)):
            window_asset1 = asset1_data.iloc[i-window_size:i]
            window_asset2 = asset2_data.iloc[i-window_size:i]
            window_spread = spread_data.iloc[i-window_size:i]

            # Calculate correlation
            correlation = window_asset1['returns'].corr(window_asset2['returns'])
            correlation_history.append(correlation)

            # Detect regime based on spread behavior
            regime = self.regime_detector.detect_regime(window_spread['spread_returns'])

            # Get adapted parameters with correlation information
            params = self.parameter_integration.adapt_parameters(
                window_spread['spread_returns'],
                window_spread['spread_returns'].std(),
                0.8,  # dummy sharpe ratio
                0.6,  # dummy win rate
                params,
                correlation=correlation
            )

            param_history.append(params)

        # Convert to DataFrame for analysis
        param_df = pd.DataFrame(param_history, index=spread_data.index[window_size:])
        param_df['correlation'] = correlation_history
        param_df['actual_regime'] = spread_data.iloc[window_size:]['regime'].values

        # Check for appropriate parameter adjustments when correlation breaks down
        correlated_params = param_df[param_df['actual_regime'] == 'correlated']
        decorrelated_params = param_df[param_df['actual_regime'] == 'decorrelated']

        # Position size should decrease when correlation breaks down
        self.assertGreater(
            correlated_params['max_position_size'].mean(),
            decorrelated_params['max_position_size'].mean(),
            "Position size not properly reduced during correlation breakdown"
        )

        # Pair exposure should decrease when correlation breaks down
        self.assertGreater(
            correlated_params['max_pair_exposure'].mean(),
            decorrelated_params['max_pair_exposure'].mean(),
            "Pair exposure not properly reduced during correlation breakdown"
        )

        # Stop loss should tighten when correlation breaks down
        self.assertLess(
            decorrelated_params['stop_loss'].mean(),
            correlated_params['stop_loss'].mean(),
            "Stop loss not properly tightened during correlation breakdown"
        )

        # Check for smooth transition (no abrupt changes)
        param_df['position_size_change'] = param_df['max_position_size'].diff().abs()
        excessive_changes = param_df['position_size_change'] > 0.03

        self.assertLessEqual(
            excessive_changes.sum(), 5,  # Allow a few instances of larger changes
            "Too many abrupt parameter changes during correlation breakdown"
        )

        # Create visualization of the correlation breakdown test
        self._plot_correlation_breakdown(asset1_data, asset2_data, param_df)

        return param_df

    def _plot_correlation_breakdown(self, asset1_data, asset2_data, param_df):
        """Helper method to visualize correlation breakdown test results."""
        plt.figure(figsize=(12, 10))

        # Plot 1: Asset Prices
        plt.subplot(4, 1, 1)
        plt.plot(asset1_data.index, asset1_data['price'] / asset1_data['price'].iloc[0],
                 label='Asset 1 (Normalized)')
        plt.plot(asset2_data.index, asset2_data['price'] / asset2_data['price'].iloc[0],
                 label='Asset 2 (Normalized)')

        # Shade backgrounds by correlation regime
        regime_change_points = asset1_data['regime'].ne(asset1_data['regime'].shift()).cumsum()
        for regime_id in regime_change_points.unique():
            regime_data = asset1_data[regime_change_points == regime_id]
            start, end = regime_data.index[0], regime_data.index[-1]
            color = 'lightgreen' if regime_data['regime'].iloc[0] == 'correlated' else 'lightcoral'
            plt.axvspan(start, end, color=color, alpha=0.3)

        plt.title('Asset Prices with Correlation Breakdown')
        plt.ylabel('Normalized Price')
        plt.legend()

        # Plot 2: Rolling Correlation
        plt.subplot(4, 1, 2)
        plt.plot(asset1_data.index, asset1_data['correlation'])
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.title('30-Day Rolling Correlation')
        plt.ylabel('Correlation')

        # Plot 3: Parameter Changes
        plt.subplot(4, 1, 3)
        plt.plot(param_df.index, param_df['max_position_size'], label='Position Size')
        plt.plot(param_df.index, param_df['max_pair_exposure'], label='Pair Exposure')
        plt.plot(param_df.index, param_df['stop_loss'], label='Stop Loss')
        plt.title('Parameter Adaptation')
        plt.ylabel('Parameter Value')
        plt.legend()

        # Plot 4: Ratio/Spread Behavior
        plt.subplot(4, 1, 4)
        spread = asset1_data['price'] / asset2_data['price']
        plt.plot(asset1_data.index, spread)
        plt.title('Price Ratio (Spread)')
        plt.ylabel('Ratio')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_breakdown_test.png'))
        plt.close()

    def _plot_regime_detection_results(self, data, detected_regimes, actual_regimes=None):
        """Helper method to visualize regime detection results."""
        plt.figure(figsize=(12, 8))

        # Plot 1: Price
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data['price'])
        plt.title('Price Movement')
        plt.ylabel('Price')

        # Plot 2: Returns Volatility
        plt.subplot(3, 1, 2)
        rolling_vol = data['returns'].rolling(window=30).std() * np.sqrt(252)  # Annualized
        plt.plot(data.index, rolling_vol)
        plt.title('30-Day Rolling Volatility (Annualized)')
        plt.ylabel('Volatility')

        # Plot 3: Regime Detection
        plt.subplot(3, 1, 3)

        # Create a numeric representation of detected regimes
        regime_map = {'normal': 0, 'volatile': 1, 'trending': 2, 'mean_reverting': 3}

        # Plot detected regimes
        if detected_regimes:
            regime_numeric = [regime_map.get(r, -1) for r in detected_regimes]
            plt.plot(data.index[-len(regime_numeric):], regime_numeric, 'o-', label='Detected')

        # Plot actual regimes if provided
        if actual_regimes:
            # Map actual regimes to numeric values
            actual_regime_map = {
                'low_vol': 0,
                'high_vol': 1,
                'trending': 2,
                'mean_reverting': 3,
                'crisis': 1  # Map crisis to volatile (1)
            }
            actual_numeric = [actual_regime_map.get(r, -1) for r in actual_regimes]
            plt.plot(data.index[-len(actual_numeric):], actual_numeric, 'x--', alpha=0.6, label='Actual')

        plt.yticks([0, 1, 2, 3], ['Normal/Low Vol', 'Volatile/High Vol', 'Trending', 'Mean Reverting'])
        plt.title('Market Regimes')
        if detected_regimes and actual_regimes:
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_detection_results.png'))
        plt.close()

    def test_end_to_end_integration(self):
        """
        Test the complete adaptive parameter system with a complex multi-regime scenario.
        This test validates that all components work together correctly.
        """
        # Generate a complex scenario with multiple regime changes
        regime_sequence = ['low_vol', 'trending', 'high_vol', 'crisis', 'low_vol']
        regime_lengths = [60, 60, 60, 60, 60]

        complex_data = self.simulator.generate_transition_data(
            regime_sequence, regime_lengths
        )

        # Initialize parameters
        params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'max_pair_exposure': 0.2
        }

        # Run through the data
        param_history = []
        regime_history = []
        performance_history = []

        # Simple performance model
        cumulative_pnl = 0
        position = 0

        window_size = 30
        for i in range(window_size, len(complex_data)):
            window = complex_data.iloc[i-window_size:i]
            current_returns = window['returns']
            current_volatility = current_returns.std() * np.sqrt(252)

            # Get the current price and return
            current_price = complex_data.iloc[i]['price']
            current_return = complex_data.iloc[i]['returns']

            # Detect regime
            regime = self.regime_detector.detect_regime(current_returns)
            regime_history.append(regime)

            # Calculate a simulated Sharpe ratio and win rate based on recent performance
            recent_sharpe = (current_returns.mean() / current_returns.std()) * np.sqrt(252) if current_returns.std() > 0 else 0
            recent_win_rate = (current_returns > 0).mean()

            # Get adapted parameters
            params = self.parameter_integration.adapt_parameters(
                current_returns,
                current_volatility,
                recent_sharpe,
                recent_win_rate,
                params
            )

            param_history.append(params)

            # Simple trading simulation
            # If no position, decide whether to enter based on regime
            if position == 0:
                if regime in ['normal', 'trending']:
                    # Enter a long position sized according to adapted parameters
                    position = params['max_position_size']
                    entry_price = current_price
                    stop_level = entry_price * (1 - params['stop_loss'])
            # If in position, check if we should exit
            elif position > 0:
                # Check if stop loss hit
                if current_price < stop_level:
                    # Exit position at stop
                    trade_return = (stop_level / entry_price - 1) * position
                    cumulative_pnl += trade_return
                    position = 0
                # Or take profit if good return achieved
                elif current_price > entry_price * 1.1:  # 10% profit target
                    trade_return = 0.1 * position  # 10% gain on position size
                    cumulative_pnl += trade_return
                    position = 0

            # Track performance
            if position > 0:
                # Mark-to-market P&L
                daily_pnl = current_return * position
                cumulative_pnl += daily_pnl

            performance_history.append(cumulative_pnl)

        # Convert to DataFrames for analysis
        param_df = pd.DataFrame(param_history, index=complex_data.index[window_size:])
        param_df['detected_regime'] = regime_history
        param_df['actual_regime'] = complex_data.iloc[window_size:]['regime'].values
        param_df['cumulative_pnl'] = performance_history

        # Evaluate adaptation effectiveness
        # 1. Check if regime detection is reasonably accurate
        regime_accuracy = {}
        for actual_regime in ['low_vol', 'high_vol', 'trending', 'crisis']:
            mask = param_df['actual_regime'] == actual_regime
            if mask.sum() > 0:
                detected = param_df[mask]['detected_regime'].value_counts()
                most_common = detected.index[0] if len(detected) > 0 else None

                # Expected mapping
                expected_mapping = {
                    'low_vol': 'normal',
                    'high_vol': 'volatile',
                    'trending': 'trending',
                    'crisis': 'volatile'
                }

                expected = expected_mapping[actual_regime]
                accuracy = detected[expected] / mask.sum() if expected in detected else 0
                regime_accuracy[actual_regime] = accuracy

        # Require at least 60% accuracy for most regimes
        for regime, accuracy in regime_accuracy.items():
            min_required = 0.5 if regime == 'crisis' else 0.6  # Lower threshold for crisis
            self.assertGreaterEqual(
                accuracy, min_required,
                f"Regime detection accuracy for {regime} below threshold: {accuracy:.2f}"
            )

        # 2. Check if parameters adapt as expected during volatile periods
        high_vol_mask = param_df['actual_regime'].isin(['high_vol', 'crisis'])
        low_vol_mask = param_df['actual_regime'] == 'low_vol'

        if high_vol_mask.sum() > 0 and low_vol_mask.sum() > 0:
            high_vol_position_size = param_df[high_vol_mask]['max_position_size'].mean()
            low_vol_position_size = param_df[low_vol_mask]['max_position_size'].mean()

            self.assertLess(
                high_vol_position_size, low_vol_position_size,
                "Position sizing not properly adjusted for volatility"
            )

        # 3. Check overall performance
        final_pnl = performance_history[-1]
        self.assertGreater(
            final_pnl, 0,
            f"End-to-end system failed to generate positive performance: {final_pnl:.4f}"
        )

        # Create comprehensive visualization
        plt.figure(figsize=(12, 15))

        # Plot 1: Price with Regime Background
        plt.subplot(5, 1, 1)
        plt.plot(complex_data.index, complex_data['price'])

        # Shade backgrounds by actual regime
        regime_change_points = complex_data['regime'].ne(complex_data['regime'].shift()).cumsum()
        for regime_id in regime_change_points.unique():
            regime_data = complex_data[regime_change_points == regime_id]
            start, end = regime_data.index[0], regime_data.index[-1]

            # Color coding for regimes
            colors = {
                'low_vol': 'lightgreen',
                'high_vol': 'lightcoral',
                'trending': 'lightskyblue',
                'crisis': 'lightpink',
                'mean_reverting': 'wheat'
            }

            regime_type = regime_data['regime'].iloc[0]
            color = colors.get(regime_type, 'lightgray')

            plt.axvspan(start, end, color=color, alpha=0.3)

            # Add regime label
            middle = start + (end - start) / 2
            plt.text(middle, complex_data['price'].max() * 0.95,
                     regime_type, ha='center', fontsize=9)

        plt.title('Asset Price with Market Regimes')
        plt.ylabel('Price')

        # Plot 2: Volatility
        plt.subplot(5, 1, 2)
        rolling_vol = complex_data['returns'].rolling(window=30).std() * np.sqrt(252)
        plt.plot(complex_data.index, rolling_vol)
        plt.title('30-Day Rolling Volatility (Annualized)')
        plt.ylabel('Volatility')

        # Plot 3: Detected vs Actual Regimes
        plt.subplot(5, 1, 3)

        # Create a numeric representation of regimes
        regime_map = {'normal': 0, 'volatile': 1, 'trending': 2, 'mean_reverting': 3, 'unknown': -1}
        actual_regime_map = {
            'low_vol': 0,
            'high_vol': 1,
            'trending': 2,
            'mean_reverting': 3,
            'crisis': 4
        }

        # Plot detected regimes
        detected_numeric = [regime_map.get(r, -1) for r in regime_history]
        plt.plot(param_df.index, detected_numeric, 'o-', label='Detected')

        # Plot actual regimes
        actual_numeric = [actual_regime_map.get(r, -1) for r in param_df['actual_regime']]
        plt.plot(param_df.index, actual_numeric, 'x--', alpha=0.6, label='Actual')

        plt.yticks([-1, 0, 1, 2, 3, 4],
                   ['Unknown', 'Normal/Low Vol', 'Volatile/High Vol', 'Trending', 'Mean Reverting', 'Crisis'])
        plt.title('Detected vs Actual Market Regimes')
        plt.legend()

        # Plot 4: Parameter Adaptation
        plt.subplot(5, 1, 4)
        plt.plot(param_df.index, param_df['max_position_size'], label='Position Size')
        plt.plot(param_df.index, param_df['stop_loss'], label='Stop Loss')
        plt.plot(param_df.index, param_df['kelly_fraction'], label='Kelly Fraction')
        plt.title('Parameter Adaptation')
        plt.ylabel('Parameter Value')
        plt.legend()

        # Plot 5: Cumulative Performance
        plt.subplot(5, 1, 5)
        plt.plot(param_df.index, param_df['cumulative_pnl'])
        plt.title('Cumulative P&L')
        plt.ylabel('P&L')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'end_to_end_test.png'))
        plt.close()

        return param_df

if __name__ == "__main__":
    unittest.main()
