"""
GARCH modeling module for volatility analysis.

This module implements various GARCH models for volatility forecasting:
- GARCH(1,1)
- EGARCH for leverage effects
- GJR-GARCH for asymmetric volatility
- Component GARCH for long/short-term volatility
"""

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class GARCHModel:
    """
    GARCH model implementation with maximum likelihood estimation.
    
    Attributes:
        returns (pd.Series): Return series for modeling
        model_type (str): Type of GARCH model ('standard', 'egarch', 'gjr', 'component')
    """
    
    returns: pd.Series
    model_type: str = 'standard'
    
    def __post_init__(self):
        """Initialize model parameters and validate inputs."""
        self.validate_data()
        self.parameters = None
        self.volatility = None
        self.residuals = None
        
    def validate_data(self) -> None:
        """Validate input data requirements."""
        if not isinstance(self.returns, pd.Series):
            raise TypeError("Returns must be a pandas Series")
            
        if not np.isfinite(self.returns).all():
            raise ValueError("Returns contain non-finite values")
            
        valid_models = {'standard', 'egarch', 'gjr', 'component'}
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model type. Must be one of {valid_models}")
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize GARCH parameters with reasonable values."""
        if self.model_type == 'standard':
            # [omega, alpha, beta]
            return np.array([0.001, 0.1, 0.8])
        elif self.model_type == 'egarch':
            # [omega, alpha, gamma, beta]
            return np.array([0.001, 0.1, 0.0, 0.8])
        elif self.model_type == 'gjr':
            # [omega, alpha, gamma, beta]
            return np.array([0.001, 0.05, 0.05, 0.8])
        else:  # component
            # [omega, alpha, beta, phi, rho]
            return np.array([0.001, 0.1, 0.8, 0.01, 0.97])
    
    def _constraint_maker(self) -> list:
        """Create parameter constraints based on model type."""
        if self.model_type == 'standard':
            # Ensure stationarity and positive variance
            def constraint1(params):
                return 1 - params[1] - params[2]  # alpha + beta < 1
            def constraint2(params):
                return params[0]  # omega > 0
                
            return [
                {'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2}
            ]
        elif self.model_type in {'egarch', 'gjr'}:
            # Persistence constraint
            def constraint1(params):
                return 1 - params[1] - params[3]  # alpha + beta < 1
                
            return [{'type': 'ineq', 'fun': constraint1}]
        else:  # component
            def constraint1(params):
                return 1 - params[1] - params[2]  # alpha + beta < 1
            def constraint2(params):
                return 1 - params[4]  # rho < 1
                
            return [
                {'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2}
            ]
    
    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate GARCH variances based on model type."""
        T = len(returns)
        variance = np.zeros(T)
        variance[0] = np.var(returns)
        
        if self.model_type == 'standard':
            omega, alpha, beta = params
            for t in range(1, T):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
                
        elif self.model_type == 'egarch':
            omega, alpha, gamma, beta = params
            log_var = np.log(np.ones(T) * np.var(returns))
            for t in range(1, T):
                z = returns[t-1] / np.sqrt(np.exp(log_var[t-1]))
                log_var[t] = omega + alpha * (abs(z) - np.sqrt(2/np.pi)) + gamma * z + beta * log_var[t-1]
            variance = np.exp(log_var)
            
        elif self.model_type == 'gjr':
            omega, alpha, gamma, beta = params
            for t in range(1, T):
                leverage = returns[t-1] < 0
                variance[t] = omega + (alpha + gamma * leverage) * returns[t-1]**2 + beta * variance[t-1]
                
        else:  # component
            omega, alpha, beta, phi, rho = params
            q = np.var(returns) * np.ones(T)  # long-term component
            for t in range(1, T):
                q[t] = omega + rho * (q[t-1] - omega) + phi * (returns[t-1]**2 - q[t-1])
                variance[t] = q[t] + alpha * (returns[t-1]**2 - q[t-1]) + beta * (variance[t-1] - q[t-1])
                
        return variance
    
    def _log_likelihood(self, params: np.ndarray) -> float:
        """Calculate negative log-likelihood for optimization."""
        variance = self._garch_variance(params, self.returns.values)
        ll = -0.5 * np.sum(np.log(variance) + self.returns.values**2 / variance)
        return -ll  # Minimize negative log-likelihood
    
    def fit(self, method: str = 'SLSQP') -> Dict[str, float]:
        """
        Fit GARCH model using maximum likelihood estimation.
        
        Args:
            method: Optimization method for scipy.optimize.minimize
            
        Returns:
            Dict[str, float]: Fitted parameters
        """
        initial_params = self._initialize_parameters()
        constraints = self._constraint_maker()
        
        result = minimize(
            self._log_likelihood,
            initial_params,
            method=method,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise RuntimeError(f"GARCH fitting failed: {result.message}")
            
        self.parameters = result.x
        self.volatility = pd.Series(
            np.sqrt(self._garch_variance(self.parameters, self.returns.values)),
            index=self.returns.index
        )
        self.residuals = self.returns / self.volatility
        
        # Create parameter dictionary
        param_names = {
            'standard': ['omega', 'alpha', 'beta'],
            'egarch': ['omega', 'alpha', 'gamma', 'beta'],
            'gjr': ['omega', 'alpha', 'gamma', 'beta'],
            'component': ['omega', 'alpha', 'beta', 'phi', 'rho']
        }
        
        return dict(zip(param_names[self.model_type], self.parameters))
    
    def forecast(self, horizon: int = 1) -> Tuple[pd.Series, pd.Series]:
        """
        Forecast volatility for specified horizon.
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            Tuple[pd.Series, pd.Series]: Point forecasts and forecast standard errors
        """
        if self.parameters is None:
            raise RuntimeError("Model must be fit before forecasting")
            
        last_var = self.volatility.iloc[-1] ** 2
        forecasts = np.zeros(horizon)
        std_errs = np.zeros(horizon)
        
        if self.model_type == 'standard':
            omega, alpha, beta = self.parameters
            for h in range(horizon):
                forecasts[h] = omega + (alpha + beta) * last_var
                last_var = forecasts[h]
                # Approximation of forecast standard errors
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(h + 1) / np.sqrt(252)
                
        elif self.model_type == 'egarch':
            omega, alpha, gamma, beta = self.parameters
            last_log_var = np.log(last_var)
            for h in range(horizon):
                forecasts[h] = np.exp(omega + beta * last_log_var)
                last_log_var = np.log(forecasts[h])
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(h + 1) / np.sqrt(252)
                
        elif self.model_type == 'gjr':
            omega, alpha, gamma, beta = self.parameters
            for h in range(horizon):
                forecasts[h] = omega + (alpha + 0.5 * gamma) * last_var + beta * last_var
                last_var = forecasts[h]
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(h + 1) / np.sqrt(252)
                
        else:  # component
            omega, alpha, beta, phi, rho = self.parameters
            last_q = omega  # Long-term component
            for h in range(horizon):
                forecasts[h] = last_q + (alpha + beta) * (last_var - last_q)
                last_var = forecasts[h]
                last_q = omega + rho * (last_q - omega)
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(h + 1) / np.sqrt(252)
        
        dates = pd.date_range(
            start=self.returns.index[-1] + pd.Timedelta('1D'),
            periods=horizon,
            freq='D'
        )
        
        return (
            pd.Series(np.sqrt(forecasts), index=dates),
            pd.Series(std_errs, index=dates)
        )
    
    def get_model_stats(self) -> pd.Series:
        """
        Calculate model statistics and diagnostics.
        
        Returns:
            pd.Series: Model statistics
        """
        if self.parameters is None:
            raise RuntimeError("Model must be fit before calculating statistics")
            
        stats = pd.Series({
            'log_likelihood': -self._log_likelihood(self.parameters),
            'aic': 2 * len(self.parameters) - 2 * -self._log_likelihood(self.parameters),
            'bic': np.log(len(self.returns)) * len(self.parameters) - 2 * -self._log_likelihood(self.parameters),
            'persistence': sum(self.parameters[1:3]),  # alpha + beta
            'unconditional_vol': np.sqrt(self.parameters[0] / (1 - self.persistence)),
            'residual_mean': self.residuals.mean(),
            'residual_std': self.residuals.std(),
            'residual_skew': self.residuals.skew(),
            'residual_kurt': self.residuals.kurtosis(),
            'ljung_box_residuals': self._ljung_box_test(self.residuals),
            'ljung_box_squared': self._ljung_box_test(self.residuals**2)
        })
        
        return stats
    
    @staticmethod
    def _ljung_box_test(series: pd.Series, lags: int = 10) -> float:
        """Calculate Ljung-Box test statistic."""
        acf = [series.autocorr(lag=i) for i in range(1, lags + 1)]
        n = len(series)
        q_stat = n * (n + 2) * sum([(acf[i]**2) / (n - i - 1) for i in range(lags)])
        return q_stat