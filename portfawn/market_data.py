import os
import json
import glob
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import pandas_datareader.data as web

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
        data_source: str = "yahoo",
        date_start: datetime = datetime.strptime("2000-01-01", "%Y-%m-%d").date(),
        date_end: datetime = datetime.strptime("2019-12-31", "%Y-%m-%d").date(),
        col_price: str = "Adj Close",
        path_data: Path = Path("data"),
    ) -> None:

        # parameters
        self.asset_list = asset_list
        self.data_source = data_source
        self.date_start = date_start
        self.date_end = date_end
        self.business_day_num = int(np.busday_count(date_start, date_end))
        self.col_price = col_price
        self.path_data = path_data

        # logging
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Start collecting market data from ({date_start.strftime('%Y-%m-%d')} "
            f"to {date_end.strftime('%Y-%m-%d')}, "
            f"{self.business_day_num} business days) for {len(asset_list)} assets: {self.asset_list}"
        )

        # raw data
        self.market_data_sig = get_assets_signature(
            asset_list=self.asset_list, start=date_start, end=date_end
        )
        self.price_df = self.retrieve_raw_data()
        self.logger.info(
            f"The price data is collected for {len(self.price_df.columns)} assets"
        )

        # returns
        self.freq_dict = get_freq_list()
        self.metrics = ["returns", "mean", "std", "cov", "corr", "cum", "total_return"]
        self.market = self.calc_asset_returns()
        del self.price_df
        self.logger.info(
            f"The metrics on {self.market.keys()} are calculated for "
            f"{len(self.market['D']['returns'].columns)} assets"
        )

    def retrieve_raw_data(self):

        self.path_data.mkdir(parents=True, exist_ok=True)

        # read the existing data

        price_files = glob.glob(str(self.path_data / Path("price_*.pkl")))

        for price_file in price_files:
            filename = Path(price_file).stem.replace("price_", "")
            filename_split = filename.split("___")
            if len(filename_split) == 3:
                start = datetime.strptime(filename_split[0], "%Y-%m-%d").date()
                end = datetime.strptime(filename_split[1], "%Y-%m-%d").date()
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

        file_price = self.path_data / Path(f"price_{self.market_data_sig}.pkl")

        raw_df = web.DataReader(
            name=self.asset_list,
            data_source=self.data_source,
            start=self.date_start,
            end=self.date_end,
        )
        raw_df.dropna(inplace=True)
        col_names = [(self.col_price, ticker) for ticker in self.asset_list]
        price_df = raw_df[col_names]
        price_df.columns = [col[1] for col in price_df.columns.values]
        price_df.dropna(inplace=True)
        price_df.to_pickle(file_price)

        return price_df

    def calc_asset_returns(self):
        asset_returns = {}
        for freq_key in self.freq_dict.keys():
            data = self.calc_asset_returns_by_freq(resample_freq=freq_key)
            if data:
                asset_returns.update({freq_key: data})
        return asset_returns

    def calc_asset_returns_by_freq(self, resample_freq="M"):
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


class PlotMarketData:
    def __init__(
        self,
        market_data,
        path_data: Path = Path("data"),
        path_results: Path = Path("results"),
    ):

        # parameters
        self.market_data = market_data
        self.path_data = path_data
        self.path_results = path_results
        self.path_data.mkdir(parents=True, exist_ok=True)
        self.path_results.mkdir(parents=True, exist_ok=True)

        # logging
        self.logger = logging.getLogger(__name__)

        # results
        self.plot_figs()
        self.market_data_summary()
        self.store_returns()
        self.logger.info(f"The figures and parameters of the market data are stored")
        parameters = self.market_data_summary().replace("    ", " ").replace("\n", "")
        self.logger.info(f"The parameters of the market data: {parameters}")

    def plot_figs(self):
        self.plot = Plot(
            asset_num=len(self.market_data.asset_list),
            path_results=self.path_results,
            plot_type="market_data",
        )
        for freq_key in self.market_data.freq_dict.keys():

            # do not plot empty DFs
            if freq_key not in self.market_data.market.keys():
                continue

            f_name = self.market_data.freq_dict[freq_key]
            f_name_cap = f_name.capitalize()
            # box
            self.plot.plot_box(
                returns=self.market_data.market[freq_key]["returns"],
                title=f"Distribution of {f_name_cap} Returns",
                xlabel="Assets",
                ylabel=f"{f_name_cap} Returns",
                filename=f"asset_box_{f_name}_returns_assets",
            )
            # heatmap
            self.plot.plot_heatmap(
                self.market_data.market[freq_key]["cov"],
                "cov",
                f"Covariance of {f_name_cap} returns",
                f"asset_cov_{f_name}_returns_assets",
            )
            self.plot.plot_heatmap(
                self.market_data.market[freq_key]["corr"],
                "corr",
                f"Correlation of {f_name_cap} returns",
                f"asset_corr_{f_name}_returns_assets",
            )
            # trends
            self.plot.plot_trend(
                returns=self.market_data.market[freq_key]["returns"],
                title=f"{f_name_cap} Asset Returns",
                xlabel="Date",
                ylabel=f"{f_name_cap} Returns",
                filename=f"asset_trend_{f_name}_returns_assets",
            )
            self.plot.plot_trend(
                returns=self.market_data.market[freq_key]["cum"],
                title=f"{f_name_cap} Cumulative Asset Returns",
                xlabel="Date",
                ylabel=f"Cumulative Returns",
                filename=f"asset_cum_{f_name}_returns_assets",
            )

    def market_data_summary(self):
        summary_market = {}
        summary_market.update(
            {"date_start": self.market_data.date_start.strftime("%Y-%m-%d")}
        )
        summary_market.update(
            {"date_end": self.market_data.date_end.strftime("%Y-%m-%d")}
        )
        summary_market.update({"data_source": self.market_data.data_source})
        summary_market.update({"asset_list": self.market_data.asset_list})
        summary_market.update({"col_price": self.market_data.col_price})
        summary_market.update({"path_data": str(self.path_data)})
        summary_market.update({"path_results": str(self.path_results)})
        summary_str = json.dumps(summary_market, indent=4)
        filename = self.path_results / Path(
            f"market_data_parameters_{self.market_data.market_data_sig}.txt"
        )
        with open(filename, "wt") as fout:
            fout.write(summary_str)
        return summary_str

    def store_returns(self):
        path_returns = self.path_results / "returns"
        path_returns.mkdir(parents=True, exist_ok=True)
        for freq_key in self.market_data.freq_dict.keys():
            if freq_key not in self.market_data.market.keys():
                continue
            for metric in self.market_data.metrics:
                f_name = self.market_data.freq_dict[freq_key]
                file = path_returns / Path(f"asset_{metric}_{f_name}.csv")
                self.market_data.market[freq_key][metric].to_csv(file)
