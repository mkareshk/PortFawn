import os
import json
import glob
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfawn.plot import Plot
from portfawn.utils import (
    get_asset_hash,
    get_assets_signature,
    get_freq_list,
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

        # logging
        self.logger = logging.getLogger(__name__)

        # making the dates standard
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

    @property
    def data_returns(self):
        return self.get_metric_by_freq(freq="D", metric="returns")

    def collect(self):

        # collect raw data
        self.price_df = self.retrieve_raw_data()

        self.logger.info(
            f"The price data ({self.data_signature}) is collected from ({self.date_start_str} "
            f"to {self.date_end_str}, "
            f"{self.business_day_num} business days) for {len(self.asset_list)} assets"
        )

        # calculate returns data
        self.freq_dict = get_freq_list()
        self.metrics = ["returns", "mean", "std", "cov", "corr", "cum", "total_return"]
        self.market_metrics = self.calc_market_metrics()

        self.logger.info(
            f"The metrics of returns data are calculated for {self.data_signature}"
        )

    def get_metric_by_freq(self, freq, metric):
        return self.market_metrics[freq][metric]

    def retrieve_raw_data(self):

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
                    price_df = price_df.loc[self.date_start : self.date_end]
                    return price_df

        # data collection using API

        file_price = self.path_data / Path(f"price_{self.data_signature}.pkl")

        raw_df = yf.Tickers(self.asset_list).history(period="max")

        raw_df.dropna(inplace=True)
        col_names = [(self.col_price, ticker) for ticker in self.asset_list]
        price_df = raw_df[col_names]
        price_df.columns = [col[1] for col in price_df.columns.values]
        price_df.dropna(inplace=True)
        price_df.to_pickle(file_price)

        return price_df

    def calc_market_metrics(self):
        market_metrics = {}
        for freq_key in self.freq_dict.keys():
            data = self.calc_market_metrics_by_freq(resample_freq=freq_key)
            if data:
                market_metrics.update({freq_key: data})
        return market_metrics

    def calc_market_metrics_by_freq(self, resample_freq="D"):
        market = {}
        market["returns"] = self.price_df.resample(resample_freq).mean().pct_change()
        market["returns"].dropna(inplace=True)
        if market["returns"].empty:
            return None
        market["mean"] = market["returns"].mean()
        market["std"] = market["returns"].std()
        market["cov"] = market["returns"].cov()
        market["corr"] = market["returns"].corr()
        market["cum"] = (market["returns"] + 1).cumprod() - 1
        market["total_return"] = self.price_df.iloc[[0, -1]].pct_change().dropna()
        return market

    def summary(self):

        path = self.path_data / Path("summary")
        path.mkdir(parents=True, exist_ok=True)

        summary_returns_data = {}
        summary_returns_data.update({"date_start": self.date_start_str})
        summary_returns_data.update({"date_end": self.date_end_str})
        summary_returns_data.update({"data_source": self.data_source})
        summary_returns_data.update({"asset_list": self.asset_list})
        summary_returns_data.update({"col_price": self.col_price})
        summary_returns_data.update({"path_data": str(self.path_data)})

        return summary_returns_data

    def store_metrics(self):

        path = self.path_data / Path("metrics")
        path.mkdir(parents=True, exist_ok=True)

        for freq_key in get_freq_list().keys():

            if freq_key not in self.market_metrics.keys():
                continue

            for metric in self.metrics:
                f_name = self.freq_dict[freq_key]
                filename = path / Path(f"asset_{metric}_{f_name}.csv")
                self.get_metric_by_freq(freq=freq_key, metric=metric).to_csv(filename)


class PlotMarketData:
    def __init__(self, returns_data, path_plot: Path):

        # parameters
        self.returns_data = returns_data
        self.path_plot = path_plot

        self.path_plot.mkdir(parents=True, exist_ok=True)

        # logging
        self.logger = logging.getLogger(__name__)

    def plot(self):

        self.plot = Plot(
            asset_num=len(self.returns_data.asset_list),
            path_plot=self.path_plot,
            plot_type="returns_data",
        )

        for freq_key in self.returns_data.freq_dict.keys():

            # do not plot empty DFs
            if freq_key not in self.returns_data.market_metrics.keys():
                continue

            f_name = self.returns_data.freq_dict[freq_key]
            f_name_cap = f_name.capitalize()

            # box
            self.plot.plot_box(
                returns=self.returns_data.get_metric_by_freq(
                    freq=freq_key, metric="returns"
                ),
                title=f"Distribution of {f_name_cap} Returns",
                xlabel="Assets",
                ylabel=f"{f_name_cap} Returns",
                filename=f"asset_box_{f_name}_returns_assets",
            )

            # heatmap
            self.plot.plot_heatmap(
                self.returns_data.get_metric_by_freq(freq=freq_key, metric="cov"),
                "cov",
                f"Covariance of {f_name_cap} returns",
                f"asset_cov_{f_name}_returns_assets",
            )
            self.plot.plot_heatmap(
                self.returns_data.get_metric_by_freq(freq=freq_key, metric="corr"),
                "corr",
                f"Correlation of {f_name_cap} returns",
                f"asset_corr_{f_name}_returns_assets",
            )

            # trends
            self.plot.plot_trend(
                returns=self.returns_data.get_metric_by_freq(
                    freq=freq_key, metric="returns"
                ),
                title=f"{f_name_cap} Asset Returns",
                xlabel="Date",
                ylabel=f"{f_name_cap} Returns",
                filename=f"asset_trend_{f_name}_returns_assets",
            )
            self.plot.plot_trend(
                returns=self.returns_data.get_metric_by_freq(
                    freq=freq_key, metric="cum"
                ),
                title=f"{f_name_cap} Cumulative Asset Returns",
                xlabel="Date",
                ylabel=f"Cumulative Returns",
                filename=f"asset_cum_{f_name}_returns_assets",
            )

        self.logger.info(
            f"The figures are generated for {self.returns_data.data_signature}"
        )
