"""
Test suite for advanced GARCH models.

Tests specialized GARCH variants under different market conditions:
- IGARCH for persistent volatility
- TGARCH for asymmetric shocks
- FIGARCH for long memory effects
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from market_analysis.time_series.advanced_garch import IGARCHModel, TGARCHModel, FIGARCHModel


@pytest.fixture
def persistent_volatility_data():
    """
    Generate returns with highly persistent volatility.
    Simulates market conditions where IGARCH is appropriate.
    """
    np.random.seed(42)
    T = 1000
    
    # Generate persistent volatility process
    omega = 0.00001
    alpha = 0.15
    beta = 1 - alpha  # IGARCH constraint
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - beta)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
    
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    return pd.Series(returns, index=dates)


@pytest.fixture
def asymmetric_volatility_data():
    """
    Generate returns with asymmetric volatility response.
    Simulates leverage effect where TGARCH is appropriate.
    """
    np.random.seed(43)
    T = 1000
    
    # Generate process with leverage effects
    omega = 0.00001
    alpha = 0.05
    gamma = 0.10  # Asymmetry parameter
    beta = 0.85
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - beta)
    
    for t in range(1, T):
        threshold = returns[t-1] < 0
        sigma2[t] = omega + alpha * returns[t-1]**2 + gamma * threshold * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
    
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    return pd.Series(returns, index=dates)


@pytest.fixture
def long_memory_data():
    """
    Generate returns with long memory in volatility.
    Simulates conditions where FIGARCH is appropriate.
    """
    np.random.seed(44)
    T = 1000
    d = 0.4  # Fractional integration parameter
    
    # Generate fractionally integrated process
    noise = np.random.standard_normal(T)
    weights = np.array([(gamma(d + j) * gamma(1 - d)) / (gamma(j + 1) * gamma(d)) 
                       for j in range(100)])
    
    returns = np.zeros(T)
    for t in range(100, T):
        volatility = np.sum(weights * noise[t-100:t]**2)
        returns[t] = np.sqrt(volatility) * noise[t]
    
    dates = pd.date_range(start='2024-01-01', periods=T, freq='D')
    return pd.Series(returns, index=dates)


def test_igarch_estimation(persistent_volatility_data):
    """Test IGARCH model estimation on persistent volatility data."""
    model = IGARCHModel(returns=persistent_volatility_data)
    params = model.fit()
    
    # Check parameter constraints
    assert params['omega'] > 0
    assert 0 < params['alpha'] < 1
    
    # Test persistence constraint
    assert abs(model.get_persistence() - 1.0) < 1e-6
    
    # Test forecasting
    forecast, std_errs = model.forecast(horizon=5)
    assert len(forecast) == 5
    assert forecast.is_monotonic
    assert std_errs is not None


def test_tgarch_asymmetry(asymmetric_volatility_data):
    """Test TGARCH model's ability to capture asymmetric effects."""
    model = TGARCHModel(returns=asymmetric_volatility_data)
    params = model.fit()
    
    # Check asymmetry parameter
    assert params['gamma'] > 0  # Positive asymmetry for leverage effect
    
    # Calculate asymmetry ratio
    asymmetry = model.get_asymmetry()
    assert asymmetry > 0
    
    # Test volatility response to positive vs negative returns
    volatility = model.volatility
    neg_shock_vol = volatility[asymmetric_volatility_data.shift(1) < 0].mean()
    pos_shock_vol = volatility[asymmetric_volatility_data.shift(1) > 0].mean()
    assert neg_shock_vol > pos_shock_vol


def test_figarch_memory(long_memory_data):
    """Test FIGARCH model's long memory properties."""
    model = FIGARCHModel(returns=long_memory_data)
    params = model.fit()
    
    # Check fractional integration parameter
    d = model.get_long_memory_parameter()
    assert 0 < d < 1
    
    # Test memory horizon
    horizon = model.calculate_memory_horizon(threshold=0.01)
    assert horizon > 0
    assert isinstance(horizon, int)
    
    # Test autocorrelation decay
    volatility = model.volatility
    acf = pd.Series(volatility).autocorr(lag=20)
    assert acf > 0  # Should show significant long-range dependence


def test_igarch_constraints():
    """Test IGARCH parameter constraints and validation."""
    returns = pd.Series(np.random.randn(100))
    model = IGARCHModel(returns=returns)
    
    # Test parameter initialization
    init_params = model._initialize_parameters()
    assert len(init_params) == 2  # omega, alpha
    
    # Test constraints
    constraints = model._constraint_maker()
    assert len(constraints) > 0
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        model._garch_variance(np.array([-1, 0.5]), returns)  # Negative omega


def test_tgarch_volatility_response():
    """Test TGARCH volatility response to different shocks."""
    model = TGARCHModel(returns=asymmetric_volatility_data())
    model.fit()
    
    # Generate test shocks
    pos_shock = pd.Series([0.02] + [0]*10)  # Positive shock
    neg_shock = pd.Series([-0.02] + [0]*10)  # Negative shock
    
    # Calculate responses
    pos_response = model._garch_variance(model.parameters, pos_shock)
    neg_response = model._garch_variance(model.parameters, neg_shock)
    
    # Negative shocks should generate higher volatility
    assert neg_response[1] > pos_response[1]


def test_figarch_long_range_dependence():
    """Test FIGARCH long-range dependence properties."""
    model = FIGARCHModel(returns=long_memory_data())
    model.fit()
    
    # Test fractional differencing
    d = 0.4
    coef = model._fractional_differencing(d, 100)
    
    # Coefficients should decay hyperbolically
    assert coef[0] == 1
    assert all(coef[1:] > 0)
    assert np.all(np.diff(coef) < 0)  # Monotonically decreasing
    
    # Test long memory properties
    memory_horizon1 = model.calculate_memory_horizon(threshold=0.01)
    memory_horizon2 = model.calculate_memory_horizon(threshold=0.001)
    assert memory_horizon2 > memory_horizon1


def test_model_forecasting():
    """Test forecasting capabilities of all models."""
    returns = persistent_volatility_data()
    
    models = {
        'igarch': IGARCHModel(returns),
        'tgarch': TGARCHModel(returns),
        'figarch': FIGARCHModel(returns)
    }
    
    for name, model in models.items():
        model.fit()
        forecast, std_errs = model.forecast(horizon=10)
        
        # Basic forecast properties
        assert len(forecast) == 10
        assert len(std_errs) == 10
        assert all(forecast > 0)
        assert all(std_errs > 0)
        
        # Uncertainty should increase with horizon
        assert std_errs.is_monotonic_increasing


def test_model_diagnostics():
    """Test model diagnostic capabilities."""
    returns = persistent_volatility_data()
    
    for ModelClass in [IGARCHModel, TGARCHModel, FIGARCHModel]:
        model = ModelClass(returns)
        model.fit()
        
        stats = model.get_model_stats()
        
        # Check common diagnostics
        assert 'log_likelihood' in stats
        assert 'aic' in stats
        assert 'bic' in stats
        
        # Check residual properties
        assert abs(model.residuals.mean()) < 0.1
        assert abs(model.residuals.std() - 1) < 0.1


def test_edge_cases():
    """Test model behavior in edge cases."""
    # Test with constant returns
    const_returns = pd.Series(np.ones(100))
    
    for ModelClass in [IGARCHModel, TGARCHModel, FIGARCHModel]:
        model = ModelClass(returns=const_returns)
        with pytest.raises(RuntimeError):
            model.fit()  # Should fail due to lack of variation
    
    # Test with very short series
    short_returns = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError):
        IGARCHModel(returns=short_returns).fit()
    
    # Test with missing values
    returns_with_nan = pd.Series(np.random.randn(100))
    returns_with_nan[50] = np.nan
    with pytest.raises(ValueError):
        IGARCHModel(returns=returns_with_nan)