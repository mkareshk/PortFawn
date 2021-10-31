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
        tickers: list,
        date_start: datetime = "2000-01-01",
        date_end: datetime = "2020-12-31",
        col_price: str = "Close",
        path_data: Path = Path("data_returns"),
    ) -> None:

        # parameters
        self.tickers = tickers
        self.date_start = date_start
        self.date_end = date_end
        self.col_price = col_price
        self.path_data = path_data
        self.path_metrics = self.path_data / Path("metrics")

        self.asset_list = list(self.tickers.values())
        self.tickers_inv = {v: k for k, v in self.tickers.items()}

        self.plot = Plot()

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

        self._data_returns = self._data_prices.pct_change().dropna()
        self._data_returns.columns = [
            self.tickers_inv[i] for i in self._data_returns.columns
        ]

    @property
    def data_returns(self):
        return self._data_returns

    @property
    def data_prices(self):
        return self._data_prices

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
                    self._data_prices = price_df.loc[self.date_start : self.date_end]
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
        self._data_prices = price_df

    def plot_dist_returns(self):
        return self.plot.plot_box(
            df=100 * self.data_returns,
            title=f"Distribution of Daily Returns",
            xlabel="Assets",
            ylabel=f"Daily Returns (%)",
        )

    def plot_corr(self):
        return self.plot.plot_heatmap(
            df=self.data_returns,
            relation_type="corr",
            title="Portfolio Correlation",
            annotate=True,
        )

    def plot_cov(self):
        return self.plot.plot_heatmap(
            df=self.data_returns,
            relation_type="cov",
            title="Portfolio Covariance",
            annotate=True,
        )

    def plot_cum_returns(self):
        portfolio_returns_cum_df = (self.data_returns + 1).cumprod() - 1
        return self.plot.plot_trend(
            df=portfolio_returns_cum_df,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
        )

    def plot_returns(self):
        return self.plot.plot_trend(
            df=self.data_returns,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
        )

    def plot_prices(self):
        return self.plot.plot_trend(
            df=self._data_prices,
            title="Price",
            xlabel="Date",
            ylabel="Price",
        )
