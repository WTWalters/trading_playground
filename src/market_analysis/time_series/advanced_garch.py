"""
Advanced GARCH models implementation.

This module implements specialized GARCH variants:
- IGARCH (Integrated GARCH) for unit-root volatility processes
- TGARCH (Threshold GARCH) for asymmetric volatility spikes
- FIGARCH (Fractionally Integrated GARCH) for long memory
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from dataclasses import dataclass
from .garch import GARCHModel


@dataclass
class IGARCHModel(GARCHModel):
    """
    Integrated GARCH model for persistent volatility.
    
    IGARCH assumes volatility persistence parameter (α + β) = 1,
    implying infinite persistence of volatility shocks.
    Critical for high-frequency trading and risk management.
    """
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize IGARCH parameters."""
        # [omega, alpha] (beta = 1 - alpha)
        return np.array([0.00001, 0.1])
    
    def _constraint_maker(self) -> list:
        """Create parameter constraints for IGARCH."""
        def constraint1(params):
            return params[0]  # omega > 0
            
        def constraint2(params):
            return params[1]  # alpha > 0
            
        def constraint3(params):
            return 1 - params[1]  # beta = (1-alpha) > 0
            
        return [
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3}
        ]
    
    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate IGARCH variances."""
        T = len(returns)
        variance = np.zeros(T)
        variance[0] = np.var(returns)
        
        omega, alpha = params
        beta = 1 - alpha  # IGARCH constraint
        
        for t in range(1, T):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
            
        return variance
    
    def get_persistence(self) -> float:
        """Get volatility persistence (always 1 for IGARCH)."""
        return 1.0


@dataclass
class TGARCHModel(GARCHModel):
    """
    Threshold GARCH model for asymmetric volatility.
    
    TGARCH allows different responses to positive and negative returns,
    capturing leverage effects and volatility feedback.
    Particularly useful for equity markets.
    """
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize TGARCH parameters."""
        # [omega, alpha, gamma, beta]
        return np.array([0.00001, 0.05, 0.05, 0.85])
    
    def _constraint_maker(self) -> list:
        """Create parameter constraints for TGARCH."""
        def constraint1(params):
            return params[0]  # omega > 0
            
        def constraint2(params):
            # Ensure positive variance: alpha + gamma/2 + beta < 1
            return 1 - params[1] - params[2]/2 - params[3]
            
        return [
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}
        ]
    
    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate TGARCH variances."""
        T = len(returns)
        variance = np.zeros(T)
        variance[0] = np.var(returns)
        
        omega, alpha, gamma, beta = params
        
        for t in range(1, T):
            # Threshold term for negative returns
            threshold = returns[t-1] < 0
            variance[t] = (
                omega +
                alpha * returns[t-1]**2 +
                gamma * threshold * returns[t-1]**2 +
                beta * variance[t-1]
            )
            
        return variance
    
    def get_asymmetry(self) -> float:
        """Calculate volatility asymmetry coefficient."""
        if self.parameters is None:
            raise RuntimeError("Model must be fit before calculating asymmetry")
            
        return self.parameters[2] / self.parameters[1]  # gamma/alpha ratio


@dataclass
class FIGARCHModel(GARCHModel):
    """
    Fractionally Integrated GARCH model for long memory.
    
    FIGARCH captures long-range dependence in volatility,
    allowing for hyperbolic decay of volatility shocks.
    Essential for long-term risk assessment.
    """
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize FIGARCH parameters."""
        # [omega, d, phi, beta]
        return np.array([0.00001, 0.4, 0.2, 0.7])
    
    def _constraint_maker(self) -> list:
        """Create parameter constraints for FIGARCH."""
        def constraint1(params):
            return params[0]  # omega > 0
            
        def constraint2(params):
            return params[1]  # d > 0
            
        def constraint3(params):
            return 1 - params[1]  # d < 1
            
        def constraint4(params):
            return params[2]  # phi > 0
            
        def constraint5(params):
            return params[3]  # beta > 0
            
        return [
            {'type': 'ineq', 'fun': f} for f in 
            [constraint1, constraint2, constraint3, constraint4, constraint5]
        ]
    
    def _fractional_differencing(self, d: float, length: int) -> np.ndarray:
        """Calculate fractional differencing coefficients."""
        k = np.arange(length)
        coef = np.zeros(length)
        coef[0] = 1
        
        for i in range(1, length):
            coef[i] = coef[i-1] * (d - i + 1) / i
            
        return coef
    
    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate FIGARCH variances."""
        T = len(returns)
        variance = np.zeros(T)
        variance[0] = np.var(returns)
        
        omega, d, phi, beta = params
        
        # Calculate fractional differencing coefficients
        lag_length = min(100, T)  # Practical truncation
        lambda_coef = self._fractional_differencing(d, lag_length)
        
        # Initialize squared returns
        squared_returns = returns**2
        
        for t in range(1, T):
            # Calculate long memory component
            memory_term = 0
            for i in range(min(t, lag_length)):
                memory_term += lambda_coef[i] * squared_returns[t-i-1]
            
            variance[t] = (
                omega +
                beta * variance[t-1] +
                (1 - beta - phi) * memory_term
            )
            
        return variance
    
    def get_long_memory_parameter(self) -> float:
        """Get long memory parameter (d)."""
        if self.parameters is None:
            raise RuntimeError("Model must be fit before getting memory parameter")
            
        return self.parameters[1]
    
    def calculate_memory_horizon(self, threshold: float = 0.01) -> int:
        """
        Calculate effective memory horizon.
        
        Args:
            threshold: Minimum coefficient value to consider
            
        Returns:
            int: Number of lags with significant memory
        """
        if self.parameters is None:
            raise RuntimeError("Model must be fit before calculating horizon")
            
        d = self.parameters[1]
        coef = self._fractional_differencing(d, 1000)  # Calculate up to 1000 lags
        significant_lags = np.where(np.abs(coef) > threshold)[0]
        
        return len(significant_lags) if len(significant_lags) > 0 else 0