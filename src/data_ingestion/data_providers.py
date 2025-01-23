from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    async def fetch_historical_data(self, *args, **kwargs):
        pass

class YahooFinanceProvider(DataProvider):
    async def fetch_historical_data(self, *args, **kwargs):
        pass

class PolygonProvider(DataProvider):
    async def fetch_historical_data(self, *args, **kwargs):
        pass
