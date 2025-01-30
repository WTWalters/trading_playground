"""
GARCH model implementation with robust parameter estimation.
"""

from typing import Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class GARCHModel:
    """
    GARCH model with robust parameter estimation.
    
    Uses variance targeting for initialization and
    bounded optimization for numerical stability.
    """
    
    returns: pd.Series
    model_type: str = 'standard'
        
    def __post_init__(self):
        """Initialize model state."""
        self._validate_data()
        self.parameters = None
        self.volatility = None
        self.residuals = None
        
    def _validate_data(self) -> None:
        """Validate input requirements."""
        if not isinstance(self.returns, pd.Series):
            raise TypeError("Returns must be pandas Series")
            
        if len(self.returns) < 252:  # At least one year of data
            raise ValueError("Insufficient data for GARCH estimation")
            
        if not np.isfinite(self.returns).all():
            raise ValueError("Returns contain non-finite values")
            
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize parameters using variance targeting."""
        unconditional_var = self.returns.var()
        
        if self.model_type == 'standard':
            # Target long-run variance
            alpha_init = 0.05
            beta_init = 0.90
            omega_init = unconditional_var * (1 - alpha_init - beta_init)
            return np.array([omega_init, alpha_init, beta_init])
            
        elif self.model_type == 'egarch':
            return np.array([
                np.log(unconditional_var),  # omega
                0.05,  # alpha
                -0.05,  # gamma
                0.90   # beta
            ])
            
        elif self.model_type == 'gjr':
            alpha_init = 0.05
            gamma_init = 0.05
            beta_init = 0.85
            omega_init = unconditional_var * (1 - alpha_init - gamma_init/2 - beta_init)
            return np.array([omega_init, alpha_init, gamma_init, beta_init])
            
        else:  # component
            return np.array([
                unconditional_var * 0.05,  # omega
                0.05,  # alpha
                0.85,  # beta
                0.01,  # phi
                0.97   # rho
            ])
            
    def _constraint_maker(self) -> list:
        """Create parameter constraints with bounds."""
        if self.model_type == 'standard':
            return [{
                'type': 'ineq',
                'fun': lambda x: 1 - x[1] - x[2]  # alpha + beta < 1
            }, {
                'type': 'ineq',
                'fun': lambda x: x[0]  # omega > 0
            }, {
                'type': 'ineq',
                'fun': lambda x: x[1]  # alpha > 0
            }, {
                'type': 'ineq',
                'fun': lambda x: x[2]  # beta > 0
            }]
            
        elif self.model_type == 'egarch':
            return [{
                'type': 'ineq',
                'fun': lambda x: 1 - abs(x[3])  # |beta| < 1
            }]
            
        elif self.model_type == 'gjr':
            return [{
                'type': 'ineq',
                'fun': lambda x: 1 - x[1] - x[2]/2 - x[3]  # alpha + gamma/2 + beta < 1
            }, {
                'type': 'ineq',
                'fun': lambda x: x[0]  # omega > 0
            }]
            
        else:
            raise NotImplementedError(f"Constraints for {self.model_type} not implemented")
            
    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate GARCH variances with numerical safeguards."""
        T = len(returns)
        EPS = 1e-6  # Small constant to prevent numerical issues
        
        variance = np.full(T, returns.var())
        
        if self.model_type == 'standard':
            omega, alpha, beta = params
            
            for t in range(1, T):
                # Add small constant to prevent zero variance
                variance[t] = max(
                    omega + alpha * returns[t-1]**2 + beta * variance[t-1],
                    EPS
                )
                
        elif self.model_type == 'egarch':
            omega, alpha, gamma, beta = params
            log_var = np.full(T, np.log(variance[0]))
            
            for t in range(1, T):
                rt = returns[t-1]
                ht = np.exp(log_var[t-1])
                z = rt / np.sqrt(ht + EPS)
                
                log_var[t] = omega + alpha * (abs(z) - np.sqrt(2/np.pi)) + gamma * z + beta * log_var[t-1]
                # Prevent extreme values
                log_var[t] = np.clip(log_var[t], -20, 20)
                
            variance = np.exp(log_var)
            
        elif self.model_type == 'gjr':
            omega, alpha, gamma, beta = params
            
            for t in range(1, T):
                rt = returns[t-1]
                leverage = rt < 0
                variance[t] = max(
                    omega + (alpha + gamma * leverage) * rt**2 + beta * variance[t-1],
                    EPS
                )
                
        return variance
        
    def _log_likelihood(self, params: np.ndarray) -> float:
        """Calculate log-likelihood with numerical stability."""
        EPS = 1e-10  # Small constant for numerical stability
        
        try:
            variance = self._garch_variance(params, self.returns.values)
            
            # Add small constant to prevent log(0)
            log_likelihood = -0.5 * np.sum(
                np.log(variance + EPS) +
                self.returns.values**2 / (variance + EPS)
            )
            
            # Check for invalid values
            if not np.isfinite(log_likelihood):
                return np.inf
                
            return -log_likelihood  # Minimize negative log-likelihood
            
        except:
            return np.inf
        
    def fit(self, method: str = 'SLSQP') -> Dict[str, float]:
        """
        Fit GARCH model with robust optimization.
        
        Args:
            method: Optimization method
            
        Returns:
            Dict[str, float]: Fitted parameters
        """
        initial_params = self._initialize_parameters()
        constraints = self._constraint_maker()
        
        # Multiple optimization attempts with different initializations
        best_result = None
        best_llh = np.inf
        
        for scale in [1.0, 0.1, 10.0]:
            try:
                result = minimize(
                    self._log_likelihood,
                    initial_params * scale,
                    method=method,
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-8,
                        'disp': False
                    }
                )
                
                if result.success and result.fun < best_llh:
                    best_result = result
                    best_llh = result.fun
                    
            except:
                continue
                
        if best_result is None:
            raise RuntimeError("GARCH fitting failed for all initializations")
            
        self.parameters = best_result.x
        self.volatility = pd.Series(
            np.sqrt(self._garch_variance(self.parameters, self.returns.values)),
            index=self.returns.index
        )
        self.residuals = self.returns / self.volatility
        
        param_names = {
            'standard': ['omega', 'alpha', 'beta'],
            'egarch': ['omega', 'alpha', 'gamma', 'beta'],
            'gjr': ['omega', 'alpha', 'gamma', 'beta']
        }[self.model_type]
        
        return dict(zip(param_names, self.parameters))
        
    def forecast(self, horizon: int = 1) -> Tuple[pd.Series, pd.Series]:
        """
        Forecast volatility with uncertainty.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Tuple[pd.Series, pd.Series]: Forecasts and standard errors
        """
        if self.parameters is None:
            raise RuntimeError("Model must be fit before forecasting")
            
        last_var = self.volatility.iloc[-1]**2
        last_return = self.returns.iloc[-1]
        
        # Initialize forecasts
        forecasts = np.zeros(horizon)
        std_errs = np.zeros(horizon)
        
        if self.model_type == 'standard':
            omega, alpha, beta = self.parameters
            persistence = alpha + beta
            unconditional_var = omega / (1 - persistence)
            
            for h in range(horizon):
                if h == 0:
                    forecasts[h] = omega + alpha * last_return**2 + beta * last_var
                else:
                    forecasts[h] = omega + (alpha + beta) * forecasts[h-1]
                    
                # Include parameter uncertainty in standard errors
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(1 + h * persistence)
                
        elif self.model_type == 'egarch':
            omega, alpha, gamma, beta = self.parameters
            last_log_var = np.log(last_var)
            z_last = last_return / np.sqrt(last_var)
            
            for h in range(horizon):
                if h == 0:
                    log_var = omega + alpha * (abs(z_last) - np.sqrt(2/np.pi)) + gamma * z_last + beta * last_log_var
                else:
                    log_var = omega + beta * np.log(forecasts[h-1])
                    
                forecasts[h] = np.exp(log_var)
                std_errs[h] = forecasts[h] * np.sqrt(1 + h * beta**2)
                
        elif self.model_type == 'gjr':
            omega, alpha, gamma, beta = self.parameters
            leverage_last = last_return < 0
            
            for h in range(horizon):
                if h == 0:
                    forecasts[h] = omega + (alpha + gamma * leverage_last) * last_return**2 + beta * last_var
                else:
                    forecasts[h] = omega + (alpha + gamma/2) * forecasts[h-1] + beta * forecasts[h-1]
                    
                std_errs[h] = np.sqrt(forecasts[h]) * np.sqrt(1 + h * (alpha + gamma/2 + beta))
                
        dates = pd.date_range(
            start=self.returns.index[-1] + pd.Timedelta('1D'),
            periods=horizon,
            freq='D'
        )
        
        return (
            pd.Series(np.sqrt(forecasts), index=dates),
            pd.Series(std_errs, index=dates)
        )