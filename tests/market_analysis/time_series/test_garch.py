"""
Test suite for GARCH models.

Tests various GARCH implementations against known properties and behaviors:
- Parameter estimation accuracy
- Volatility forecasting
- Model diagnostics
- Different market regimes
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from market_analysis.time_series.garch import GARCHModel


@pytest.fixture
def simulated_garch_returns():
    """
    Generate returns from a known GARCH(1,1) process.
    True parameters: omega=0.001, alpha=0.1, beta=0.85
    """
    np.random.seed(42)
    T = 1000
    omega, alpha, beta = 0.001, 0.1, 0.85
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
    
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    return pd.Series(returns, index=dates)


@pytest.fixture
def simulated_leverage_returns():
    """
    Generate returns exhibiting leverage effects (asymmetric volatility).
    Negative returns should increase volatility more than positive returns.
    """
    np.random.seed(43)
    T = 1000
    omega, alpha, gamma, beta = 0.001, 0.05, 0.08, 0.85
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, T):
        leverage = returns[t-1] < 0
        sigma2[t] = omega + (alpha + gamma * leverage) * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
    
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    return pd.Series(returns, index=dates)


@pytest.fixture
def garch_model(simulated_garch_returns):
    """Create standard GARCH model instance."""
    return GARCHModel(returns=simulated_garch_returns, model_type='standard')


def test_garch_parameter_estimation(garch_model):
    """Test GARCH(1,1) parameter estimation accuracy."""
    # True parameters
    true_params = {'omega': 0.001, 'alpha': 0.1, 'beta': 0.85}
    
    # Fit model
    estimated_params = garch_model.fit()
    
    # Check parameter accuracy (within reasonable bounds)
    for param, true_value in true_params.items():
        assert abs(estimated_params[param] - true_value) < 0.05
    
    # Check stationarity constraint
    assert estimated_params['alpha'] + estimated_params['beta'] < 1
    
    # Check positive variance constraint
    assert estimated_params['omega'] > 0


def test_egarch_leverage_effect():
    """Test EGARCH model's ability to capture leverage effects."""
    returns = simulated_leverage_returns()
    model = GARCHModel(returns=returns, model_type='egarch')
    params = model.fit()
    
    # Gamma parameter should be negative (leverage effect)
    assert params['gamma'] < 0
    
    # Check asymmetric response
    volatility = model.volatility
    neg_shocks = returns[returns < 0]
    pos_shocks = returns[returns > 0]
    
    # Volatility after negative shocks should be higher
    vol_after_neg = volatility[returns.shift(1) < 0].mean()
    vol_after_pos = volatility[returns.shift(1) > 0].mean()
    assert vol_after_neg > vol_after_pos


def test_gjr_garch_asymmetry():
    """Test GJR-GARCH model's asymmetric volatility response."""
    returns = simulated_leverage_returns()
    model = GARCHModel(returns=returns, model_type='gjr')
    params = model.fit()
    
    # Gamma parameter should be positive (increased volatility after negative returns)
    assert params['gamma'] > 0
    
    # Test news impact curve asymmetry
    volatility = model.volatility
    assert volatility[returns.shift(1) < 0].std() > volatility[returns.shift(1) > 0].std()


def test_component_garch_persistence():
    """Test Component GARCH long-term/short-term decomposition."""
    returns = simulated_garch_returns()
    model = GARCHModel(returns=returns, model_type='component')
    params = model.fit()
    
    # Check component model properties
    assert 0 < params['rho'] < 1  # Long-term persistence
    assert params['phi'] > 0  # Long-term component impact
    
    # Long-term component should be more persistent
    assert params['rho'] > params['alpha'] + params['beta']


def test_volatility_forecasting(garch_model):
    """Test volatility forecasting functionality."""
    # Fit model
    garch_model.fit()
    
    # Generate forecasts
    horizon = 10
    forecasts, std_errs = garch_model.forecast(horizon=horizon)
    
    # Basic forecast properties
    assert len(forecasts) == horizon
    assert len(std_errs) == horizon
    assert (forecasts > 0).all()
    assert (std_errs > 0).all()
    
    # Uncertainty should increase with horizon
    assert std_errs.is_monotonic_increasing
    
    # Forecast convergence to unconditional volatility
    long_horizon = 100
    long_forecasts, _ = garch_model.forecast(horizon=long_horizon)
    unconditional_vol = np.sqrt(garch_model.parameters[0] / 
                               (1 - garch_model.parameters[1] - garch_model.parameters[2]))
    assert abs(long_forecasts.iloc[-1] - unconditional_vol) < 0.01


def test_model_diagnostics(garch_model):
    """Test model diagnostic statistics."""
    # Fit model
    garch_model.fit()
    stats = garch_model.get_model_stats()
    
    # Check required statistics
    required_stats = [
        'log_likelihood', 'aic', 'bic', 'persistence',
        'unconditional_vol', 'residual_mean', 'residual_std',
        'residual_skew', 'residual_kurt',
        'ljung_box_residuals', 'ljung_box_squared'
    ]
    for stat in required_stats:
        assert stat in stats.index
    
    # Check basic properties
    assert stats['persistence'] < 1  # Stationarity
    assert abs(stats['residual_mean']) < 0.1  # Standardized residuals
    assert abs(stats['residual_std'] - 1) < 0.1  # Should be close to 1
    
    # Model selection criteria should be finite
    assert np.isfinite(stats['aic'])
    assert np.isfinite(stats['bic'])


def test_model_validation():
    """Test model validation and error handling."""
    # Test invalid model type
    with pytest.raises(ValueError):
        GARCHModel(returns=pd.Series([1, 2, 3]), model_type='invalid')
    
    # Test non-finite returns
    with pytest.raises(ValueError):
        GARCHModel(returns=pd.Series([1, np.nan, 3]))
    
    # Test forecasting before fitting
    model = GARCHModel(returns=pd.Series([1, 2, 3], index=pd.date_range('2024-01-01', periods=3)))
    with pytest.raises(RuntimeError):
        model.forecast()
    
    # Test statistics before fitting
    with pytest.raises(RuntimeError):
        model.get_model_stats()


def test_different_market_regimes():
    """Test GARCH models under different market regimes."""
    # Generate regime-switching returns
    np.random.seed(44)
    T = 1000
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    
    # Low volatility regime
    low_vol = np.random.normal(0, 0.01, T//2)
    # High volatility regime
    high_vol = np.random.normal(0, 0.03, T//2)
    
    returns = pd.Series(np.concatenate([low_vol, high_vol]), index=dates)
    
    # Fit models
    models = {
        'standard': GARCHModel(returns=returns, model_type='standard'),
        'egarch': GARCHModel(returns=returns, model_type='egarch'),
        'gjr': GARCHModel(returns=returns, model_type='gjr'),
        'component': GARCHModel(returns=returns, model_type='component')
    }
    
    for name, model in models.items():
        model.fit()
        volatility = model.volatility
        
        # Check regime detection
        low_regime_vol = volatility.iloc[:T//2].mean()
        high_regime_vol = volatility.iloc[T//2:].mean()
        assert high_regime_vol > 2 * low_regime_vol  # High vol should be notably higher