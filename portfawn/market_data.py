import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfawn.plot import Plot
from portfawn.utils import (
    get_asset_hash,
    get_assets_signature,
)


class MarketData:
    def __init__(
        self,
        asset_list: list,
        date_start: datetime = "2000-01-01",
        date_end: datetime = "2020-12-31",
        col_price: str = "Close",
        path_data: Path = Path("data_returns"),
    ) -> None:

        # parameters
        self.asset_list = asset_list
        self.date_start = date_start
        self.date_end = date_end
        self.col_price = col_price
        self.path_data = path_data
        self.path_metrics = self.path_data / Path("metrics")

        # make the dates standard
        if type(date_start) != type(date_end):
            raise ValueError(
                f"date_start ({type(date_start)}) and "
                f"date_end ({type(date_end)}) should have the same type"
            )

        elif isinstance(self.date_start, str) and isinstance(self.date_end, str):
            self.date_start_str = self.date_start
            self.date_end_str = self.date_end
            self.date_start = datetime.datetime.strptime(
                self.date_start, "%Y-%m-%d"
            ).date()
            self.date_end = datetime.datetime.strptime(self.date_end, "%Y-%m-%d").date()

        elif isinstance(self.date_start, datetime.date) and isinstance(
            self.date_end, datetime.date
        ):
            self.date_start = self.date_start
            self.date_end = self.date_end
            self.date_start_str = self.date_start.strftime("%Y-%m-%d")
            self.date_end_str = self.date_start.strftime("%Y-%m-%d")

        else:
            raise ValueError(
                "date_start and date_end types should be either datetime.date "
                "or str (e.g. '2014-03-24')"
            )

        self.business_day_num = int(np.busday_count(date_start, date_end))
        self.data_signature = get_assets_signature(
            asset_list=self.asset_list, start=self.date_start_str, end=self.date_end_str
        )

        # retrieve the data
        self.collect()

        self._data_returns = self.price_df.pct_change().dropna()

    @property
    def data_returns(self):
        return self._data_returns

    def collect(self):

        # collect raw data
        self.path_data.mkdir(parents=True, exist_ok=True)

        # read the existing data

        price_files = glob.glob(str(self.path_data / Path("price_*.pkl")))

        for price_file in price_files:
            filename = Path(price_file).stem.replace("price_", "")
            filename_split = filename.split("___")
            if len(filename_split) == 3:
                start = datetime.datetime.strptime(filename_split[0], "%Y-%m-%d").date()
                end = datetime.datetime.strptime(filename_split[1], "%Y-%m-%d").date()
                asset_sig = filename_split[2]
                if (
                    asset_sig == get_asset_hash(self.asset_list)
                    and start <= self.date_start
                    and end >= self.date_end
                ):
                    price_df = pd.read_pickle(price_file)
                    self.price_df = price_df.loc[self.date_start : self.date_end]
                    return

        # data collection using API

        file_price = self.path_data / Path(f"price_{self.data_signature}.pkl")

        raw_df = yf.Tickers(self.asset_list).history(period="max")

        raw_df.dropna(inplace=True)
        col_names = [(self.col_price, ticker) for ticker in self.asset_list]
        price_df = raw_df[col_names]
        price_df.columns = [col[1] for col in price_df.columns.values]
        price_df.dropna(inplace=True)
        price_df.to_pickle(file_price)
        self.price_df = price_df


class MarketDataAnalysis:
    def __init__(self, market_data, path_plot: Path = Path("results") / Path("market")):

        # parameters
        self.returns_data = market_data.data_returns
        self.path_plot = path_plot

        self.path_plot.mkdir(parents=True, exist_ok=True)

    def plot(self):

        self.plot = Plot(path_plot=self.path_plot)

        # distribution of daily returns
        self.plot.plot_box(
            returns=100 * self.returns_data,
            title=f"Distribution of Daily Returns",
            xlabel="Assets",
            ylabel=f"Daily Returns (%)",
            filename=f"asset_daily_returns",
        )

        # heatmap of corr and cov
        self.plot.plot_heatmap(
            self.returns_data,
            "cov",
            "Covariance of Daily returns",
            False,
            "asset_cov_daily",
        )
        self.plot.plot_heatmap(
            self.returns_data,
            "corr",
            "Correlation of Daily returns",
            False,
            "asset_corr_daily",
        )

        # daily returns
        self.returns_data_cum = (self.returns_data + 1).cumprod() - 1
        self.plot.plot_trend(
            returns=self.returns_data_cum,
            title="Cumulative Asset Returns",
            xlabel="Date",
            ylabel="Cumulative Returns",
            filename="asset__daily_cum_returns",
        )
