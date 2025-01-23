from typing import Dict, Any
import asyncpg
import pandas as pd

class DatabaseManager:
    async def store_market_data(self, data: pd.DataFrame, symbol: str, provider: str, timeframe: str) -> None:
        pass

    async def get_latest_dates(self) -> Dict[str, Any]:
        pass
