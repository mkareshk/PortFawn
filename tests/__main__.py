from datetime import datetime
from portfawn.market_data import MarketData


asset_list = ["SPY", "BND", "GDL"]
start = datetime.strptime("2019-01-01", "%Y-%m-%d").date()
end = datetime.strptime("2019-02-10", "%Y-%m-%d").date()


class TestMarketData:
    def __init__(self, asset_list):
        self.asset_list = asset_list
        self.market_data = MarketData(
            asset_list=self.asset_list, date_start=start, date_end=end
        )


def test_create_market_data_instance():
    market_data = TestMarketData(asset_list)


test_create_market_data_instance()
