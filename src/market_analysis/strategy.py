# src/market_analysis/strategy.py

class TradingStrategy(ABC):
    """Base class for trading strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, analysis_results: Dict) -> pd.Series:
        """Generate entry/exit signals"""
        pass

    @abstractmethod
    def calculate_stops(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """Calculate stop levels"""
        pass
