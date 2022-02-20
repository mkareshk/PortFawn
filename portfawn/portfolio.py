# import hashlib
# import json
# import logging
# from os import error
# import time
# from pathlib import Path

# import dafin
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from dafin import Returns

# from portfawn.plot import Plot

# from portfawn.utils import get_assets_signature, is_jsonable

# from portfawn.models.risk import RiskModel
# from portfawn.models.optimization import QuantumOptModel, ClassicOptModel
# from portfawn.models.economic import EconomicModel

# logger = logging.getLogger(__name__)


# class Portfolio:
#     def __init__(
#         self,
#         name: str,
#         objective: str,
#         risk_type="standard",
#         risk_sample_num=100,
#         risk_sample_size=20,
#         risk_agg_func="median",
#         risk_free_rate=0.0,
#         annualized_days=252,
#         backend="neal",
#         annealing_time=100,
#         scipy_params={"maxiter": 1000, "disp": False, "ftol": 1e-10},
#         target_return=0.1,
#         target_sd=0.1,
#         weight_bound=(0.0, 1.0),
#         init_point=None,
#     ):

#         # args
#         self._name = name
#         self._objective = objective
#         self._risk_type = risk_type
#         self._risk_sample_num = risk_sample_num
#         self._risk_sample_size = risk_sample_size
#         self._risk_agg_func = risk_agg_func
#         self._risk_free_rate = risk_free_rate
#         self._annualized_days = annualized_days
#         self._backend = backend
#         self._annealing_time = annealing_time
#         self._scipy_params = scipy_params
#         self._target_return = target_return
#         self._target_sd = target_sd
#         self._weight_bound = weight_bound
#         self._init_point = init_point

#         self._config = {
#             "name": name,
#             "objective": objective,
#             "risk_type": risk_type,
#             "risk_sample_num": risk_sample_num,
#             "risk_sample_size": risk_sample_size,
#             "risk_agg_func": risk_agg_func,
#             "risk_free_rate": risk_free_rate,
#             "annualized_days": annualized_days,
#             "backend": backend,
#             "annealing_time": annealing_time,
#             "scipy_params": scipy_params,
#             "target_return": target_return,
#             "target_sd": target_sd,
#             "weight_bound": weight_bound,
#             "init_point": init_point,
#         }

#         # risk model
#         self._risk_model = RiskModel(
#             type=self._risk_type,
#             sample_num=self._risk_sample_num,
#             sample_size=self._risk_sample_size,
#             agg_func=self._risk_agg_func,
#         )

#         # economic model
#         self._economic_model = EconomicModel(
#             risk_free_rate=self._risk_free_rate, annualized_days=self._annualized_days
#         )

#         # optimization model
#         if self._objective in ["BMOP"]:
#             self._optimizer = QuantumOptModel(
#                 objective=self._objective,
#                 backend=self._backend,
#                 annealing_time=self._annealing_time,
#             )

#         elif self._objective in ["EWP", "MRP", "MVP", "MSRP"]:
#             self._optimizer = ClassicOptModel(
#                 objective=self._objective,
#                 economic_model=self._economic_model,
#                 scipy_params=self._scipy_params,
#                 target_return=self._target_return,
#                 target_sd=self._target_sd,
#                 weight_bound=self._weight_bound,
#                 init_point=self._init_point,
#             )

#         else:
#             raise NotImplementedError

#     def run(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

#         # returns data
#         returns_data = Returns(
#             asset_list=asset_list,
#             date_start=date_start,
#             date_end=date_end,
#         )

#         asset_returns = returns_data.returns
#         asset_cum_returns = returns_data.cum_returns
#         asset_cov = asset_returns.cov()

#         # risk evaluation
#         expected_return, expected_cov = self._risk_model.evaluate(returns_data)
#         expected_return_np = expected_return.to_numpy()
#         expected_cov_np = expected_cov.to_numpy()

#         # optimization
#         w = self._optimizer.optimize(
#             expected_return=expected_return, expected_cov=expected_cov
#         )
#         asset_weights = {asset_list[ind]: float(w) for ind, w in enumerate(w)}

#         ## performance metrics

#         # portfolio daily returns
#         portfolio_returns = pd.DataFrame(
#             asset_returns.to_numpy().dot(w),
#             index=asset_returns.index,
#             columns=[self._objective],
#         )
#         # portfolio_returns.columns = self._objective
#         portfolio_assets_returns = pd.concat([asset_returns, portfolio_returns], axis=1)
#         # import logging

#         # logging.error(portfolio_assets_returns)
#         # return
#         # portfolio cummulative return
#         portfolio_cum_returns = (portfolio_returns + 1).cumprod() - 1
#         portfolio_assets_cum_returns = pd.concat(
#             [asset_cum_returns, portfolio_cum_returns], axis=1
#         )

#         # total returns

#         portfolio_asset_total_return = portfolio_assets_cum_returns.iloc[-1, :]
#         portfolio_total_return = portfolio_asset_total_return[self._objective]

#         # portfolio expected return and sd
#         portfolio_expected_return = expected_return_np.dot(w)
#         portfolio_expected_sd = np.sqrt(w.T.dot(expected_cov_np).dot(w))

#         # market
#         market_mean_sd = pd.DataFrame(columns=["mean", "sd"])
#         market_mean_sd["mean"] = returns_data.returns.mean()
#         market_mean_sd["sd"] = returns_data.returns.std()

#         # portfolio
#         portfolio_mean_sd = pd.DataFrame(
#             index=[self._objective], columns=["mean", "sd"]
#         )
#         portfolio_mean_sd["mean"] = portfolio_expected_return
#         portfolio_mean_sd["sd"] = portfolio_expected_sd

#         # # random portfolio
#         # n = 1000
#         # returns_np = returns_data.returns.to_numpy()
#         # cov = returns_data.returns.cov().to_numpy()
#         # r_list = []
#         # for i in range(n):
#         #     w_rand = np.random.random((1, cov.shape[0]))
#         #     w_rand = w_rand / w_rand.sum()
#         #     r = returns_np.dot(w_rand.T).mean()
#         #     c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0]
#         #     r_list.append({"mean": r, "sd": c})
#         # random_portfolios = pd.DataFrame(r_list)

#         performance = {}
#         performance.update(
#             {
#                 "asset_weights": asset_weights,
#                 "asset_returns": asset_returns,
#                 "portfolio_returns": portfolio_returns,
#                 "portfolio_assets_returns": portfolio_assets_returns,
#                 "portfolio_cum_returns": portfolio_cum_returns,
#                 "portfolio_assets_cum_returns": portfolio_assets_cum_returns,
#                 "portfolio_asset_total_return": portfolio_asset_total_return,
#                 "portfolio_total_return": portfolio_total_return,
#                 "portfolio_expected_return": portfolio_expected_return,
#                 "portfolio_expected_sd": portfolio_expected_sd,
#                 "market_mean_sd": market_mean_sd,
#                 "portfolio_mean_sd": portfolio_mean_sd,
#                 # "random_portfolios": random_portfolios,
#                 "portfolio_config": self._config,
#             }
#         )

#         return performance

#     def __str__(self):

#         p = self.performance.copy()
#         w_str = json.dumps(p["asset_weights_dict"], sort_keys=True, indent=4)

#         out_str = ""
#         out_str += f"- asset_weights_dict:\n{w_str}\n\n"
#         out_str += f"- daily_return:\n{p['daily_return']}\n\n"
#         out_str += f"- daily_sd:\n{p['daily_sd']}\n\n"
#         out_str += f"- portfolio_returns:\n{p['portfolio_returns']}\n\n"
#         out_str += f"- portfolio_cum_returns:\n{p['portfolio_cum_returns']}\n\n"

#         return out_str


# class PlotPortfolio:
#     def __init__(self, performance) -> None:
#         self.performance = performance
#         self.plot = Plot()

#     def plot_returns(self):
#         fig, ax = self.plot.plot_trend(
#             df=self.performance["portfolio_returns"],
#             title=f"",
#             xlabel="Date",
#             ylabel="Returns",
#             legend=False,
#         )
#         return fig, ax

#     def plot_cum_returns(self):
#         fig, ax = self.plot.plot_trend(
#             df=self.performance["portfolio_assets_cum_returns"],
#             title="",
#             xlabel="Date",
#             ylabel="Returns",
#         )
#         return fig, ax

#     def plot_dist_returns(self):
#         fig, ax = self.plot.plot_box(
#             df=self.performance["portfolio_assets_returns"],
#             title="",
#             xlabel="Portfolio Fitness",
#             ylabel="Daily Returns",
#             yscale="symlog",
#         )
#         return fig, ax

#     def plot_corr(self):

#         fig, ax = self.plot.plot_heatmap(
#             df=self.performance["portfolio_assets_returns"],
#             relation_type="corr",
#             title="",
#             annotate=True,
#         )
#         return fig, ax

#     def plot_cov(self):
#         fig, ax = self.plot.plot_heatmap(
#             df=self.performance["portfolio_assets_returns"],
#             relation_type="cov",
#             title="",
#             annotate=True,
#         )
#         return fig, ax

#     def plot_mean_sd(
#         self,
#         annualized=True,
#         fig=None,
#         ax=None,
#     ):

#         market_mean_sd = self.performance["market_mean_sd"].copy()
#         portfolio_mean_sd = self.performance["portfolio_mean_sd"].copy()
#         random_mean_sd = self.random_portfolio(self.performance["asset_returns"])

#         annualized_days = self.performance["portfolio_config"]["annualized_days"]

#         if annualized:
#             market_mean_sd["mean"] *= annualized_days
#             market_mean_sd["sd"] *= np.sqrt(annualized_days)
#             portfolio_mean_sd["mean"] *= annualized_days
#             portfolio_mean_sd["sd"] *= np.sqrt(annualized_days)
#             random_mean_sd["mean"] *= annualized_days
#             random_mean_sd["sd"] *= np.sqrt(annualized_days)

#         fig, ax = self.plot.plot_scatter_portfolio_random(
#             df_1=market_mean_sd,
#             df_2=portfolio_mean_sd,
#             df_3=random_mean_sd,
#             title="",
#             xlabel="Volatility (SD)",
#             ylabel="Expected Returns",
#         )

#         return fig, ax

#     def random_portfolio(self, asset_returns):
#         n = 1000
#         returns_np = asset_returns.to_numpy()
#         cov = asset_returns.cov().to_numpy()
#         r_list = []
#         for i in range(n):
#             w_rand = np.random.random((1, cov.shape[0]))
#             w_rand = w_rand / w_rand.sum()
#             r = returns_np.dot(w_rand.T).mean()
#             c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0]
#             r_list.append({"mean": r, "sd": c})
#         return pd.DataFrame(r_list)


# class BackTest:
#     def __init__(self, **backtesting_config):
#         """[summary]

#         Args:
#             experiment_name ([type]): [description]
#             portfolio_fitness_list ([type]): [description]
#             tickers ([type]): [description]
#             start_date ([type]): [description]
#             end_date ([type]): [description]
#             optimization_params ([type]): [description]
#             sampling_params ([type]): [description]
#             training_days ([type]): [description]
#             testing_days ([type]): [description]
#             risk_free_rate (float): [description]
#             n_jobs ([type]): [description]
#         """

#         # parameters
#         self._backtesting_config = backtesting_config
#         self.backtesting_name = backtesting_config["backtesting_name"]
#         self.portfolio_fitness = backtesting_config["portfolio_fitness"]
#         self.tickers = backtesting_config["tickers"]
#         self.start_date = backtesting_config["start_date"]
#         self.end_date = backtesting_config["end_date"]
#         self.optimization_params = backtesting_config["optimization_params"]
#         self.sampling_params = backtesting_config["sampling_params"]
#         self.training_days = backtesting_config["training_days"]
#         self.testing_days = backtesting_config["testing_days"]
#         self.risk_free_rate = backtesting_config["risk_free_rate"]
#         self.n_jobs = backtesting_config["n_jobs"]

#         self.asset_list = list(self.tickers.values())
#         self.tickers_inv = {v: k for k, v in self.tickers.items()}
#         self.portfolio_fitness_list = list(self.portfolio_fitness.keys())
#         self.annualized_days = 252

#         # create the time windows
#         self.analysis_range = pd.date_range(
#             start=self.start_date,
#             end=self.end_date,
#             freq=f"{self.testing_days}D",
#         )

#         # each window is a tuple of three elements:
#         # (the first day of training, the reference day, the last day of testing)
#         self.training_delta = pd.Timedelta(self.training_days, unit="d")
#         self.testing_delta = pd.Timedelta(self.testing_days, unit="d")
#         self.analysis_windows = [
#             (i.date() - self.training_delta, i.date(), i.date() + self.testing_delta)
#             for i in self.analysis_range
#         ]

#         # market data
#         self.market_data = MarketData(
#             tickers=self.tickers,
#             date_start=self.start_date - pd.Timedelta(self.training_days, unit="d"),
#             date_end=self.end_date + pd.Timedelta(self.training_days, unit="d"),
#         )

#         self.plot = Plot()

#     @property
#     def backtesting_config(self):
#         return self._backtesting_config

#     def get_portfolio_instances(self):
#         return [
#             dict(
#                 portfolio_fitness=portfolio_fitness,
#                 date_start_training=window[0],
#                 date_end_training=window[1],
#                 date_start_testing=window[1],
#                 date_end_testing=window[2],
#             )
#             for window in self.analysis_windows
#             for portfolio_fitness in self.portfolio_fitness_list
#         ]

#     def run(self):

#         # sequential
#         if self.n_jobs == 1:
#             profile_backtesting = [
#                 self.run_iter(**instance) for instance in self.get_portfolio_instances()
#             ]

#         # parallel
#         elif self.n_jobs > 1:
#             profile_backtesting = Parallel(n_jobs=self.n_jobs)(
#                 delayed(self.run_iter)(**instance)
#                 for instance in self.get_portfolio_instances()
#             )

#         # create profiles
#         profile_backtesting = profile_backtesting
#         data_list = []

#         for profile in profile_backtesting:
#             for mode in ["profile_training", "profile_testing"]:

#                 curr = profile[mode]
#                 d = {
#                     "type": curr["type"],
#                     "portfolio_fitness": curr["portfolio_fitness"],
#                     "date_start": curr["date_start"],
#                     "date_end": curr["date_end"],
#                     "date": curr["date"],
#                     "portfolio_daily_return": curr["daily_return"],
#                     "portfolio_daily_sd": curr["daily_sd"],
#                     "portfolio_total_return": curr["portfolio_total_return"],
#                     "portfolio_asset_total_return": curr[
#                         "portfolio_asset_total_return"
#                     ].to_dict(),
#                     "portfolio_asset_mean_sd": curr[
#                         "portfolio_asset_mean_sd"
#                     ].to_dict(),
#                     "portfolio_mean_sd": curr["portfolio_mean_sd"],
#                     "asset_weights_dict": curr["asset_weights_dict"],
#                     "execution_time": curr["execution_time"],
#                     "optimization_params": curr["optimization_params"],
#                     "sampling_params": curr["sampling_params"],
#                     "mode": mode,
#                 }

#                 data_list.append(d)

#         profile_df = pd.DataFrame(data_list)

#         portfolio_df = profile_df.loc[profile_df["type"] == "testing", :].set_index(
#             "date", inplace=False
#         )
#         portfolio_returns_df = pd.DataFrame()

#         for portfolio_fitness in portfolio_df["portfolio_fitness"].unique():
#             temp = portfolio_df.loc[
#                 portfolio_df["portfolio_fitness"] == portfolio_fitness, :
#             ]
#             portfolio_returns_df[portfolio_fitness] = temp["portfolio_total_return"]

#         portfolio_cum_returns_df = (portfolio_returns_df + 1).cumprod() - 1

#         self.profile_backtesting = profile_backtesting
#         self.profile_df = profile_df
#         self.portfolio_returns_df = portfolio_returns_df
#         self.portfolio_cum_returns_df = portfolio_cum_returns_df
#         self.asset_weights_df = pd.DataFrame(
#             [item for ind, item in profile_df["asset_weights_dict"].items()],
#             index=profile_df["date"],
#         )
#         self.mean_sd = pd.concat([i for i in profile_df["portfolio_mean_sd"]])

#     def run_iter(
#         self,
#         portfolio_fitness,
#         date_start_training,
#         date_end_training,
#         date_start_testing,
#         date_end_testing,
#     ):
#         # training
#         t0 = time.time()

#         portfolio_training = self.train(
#             portfolio_fitness=portfolio_fitness,
#             date_start_training=date_start_training,
#             date_end_training=date_end_training,
#         )

#         training_time = time.time() - t0
#         logger.info(
#             f"Trained {portfolio_fitness} portfolio from {date_start_training}"
#             f"to {date_end_training} in {training_time} seconds"
#         )

#         # testing
#         t0 = time.time()

#         portfolio_testing = self.test(
#             portfolio_fitness=portfolio_fitness,
#             asset_weights=portfolio_training.asset_weights,
#             date_start_testing=date_start_testing,
#             date_end_testing=date_end_testing,
#         )

#         testing_time = time.time() - t0
#         logger.info(
#             f"Tested portfolio from {date_start_testing} to {date_end_testing}"
#             f" in {testing_time} seconds"
#         )

#         # preparing the result
#         profile_training = self.portfolio_profile(portfolio_training)
#         profile_testing = self.portfolio_profile(portfolio_testing)

#         profile_training.update(
#             {
#                 "type": "training",
#                 "date": date_start_training.strftime("%Y/%m/%d"),
#                 "execution_time": training_time,
#             }
#         )
#         profile_testing.update(
#             {
#                 "type": "testing",
#                 "date": date_start_testing.strftime("%Y/%m/%d"),
#                 "execution_time": testing_time,
#             }
#         )

#         return dict(profile_training=profile_training, profile_testing=profile_testing)

#     def train(
#         self,
#         portfolio_fitness,
#         date_start_training,
#         date_end_training,
#     ):
#         data_returns = self.market_data.data_returns.loc[
#             date_start_training:date_end_training, :
#         ]

#         portfolio_training = Portfolio(
#             portfolio_fitness=portfolio_fitness,
#             data_returns=data_returns,
#             risk_free_rate=self.risk_free_rate,
#             optimization_params=self.optimization_params,
#             sampling_params=self.sampling_params,
#         )
#         portfolio_training.optimize()

#         return portfolio_training

#     def test(
#         self,
#         portfolio_fitness,
#         asset_weights,
#         date_start_testing,
#         date_end_testing,
#     ):
#         data_returns = self.market_data.data_returns.loc[
#             date_start_testing:date_end_testing, :
#         ]
#         portfolio_testing = Portfolio(
#             portfolio_fitness=portfolio_fitness,
#             data_returns=data_returns,
#             asset_weights=asset_weights,
#             risk_free_rate=self.risk_free_rate,
#             optimization_params=self.optimization_params,
#             sampling_params=self.sampling_params,
#         )
#         return portfolio_testing

#     @staticmethod
#     def portfolio_profile(portfolio):
#         portfolio.evaluate()

#         result = dict(
#             portfolio_fitness=portfolio.portfolio_fitness,
#             optimization_params=portfolio.optimization_params,
#             sampling_params=portfolio.sampling_params,
#             date_start=portfolio.date_start.strftime("%Y/%m/%d"),
#             date_end=portfolio.date_end.strftime("%Y/%m/%d"),
#             asset_weights=portfolio.asset_weights_dict,
#         )
#         result.update(portfolio.performance)

#         return result

#     def plot_returns(self):
#         fig, ax = self.plot.plot_trend(
#             df=self.portfolio_returns_df,
#             title="",
#             xlabel="Date",
#             ylabel="Returns",
#         )
#         return fig, ax

#     def plot_cum_returns(self):
#         fig, ax = self.plot.plot_trend(
#             df=self.portfolio_cum_returns_df,
#             title="",
#             xlabel="Date",
#             ylabel="Returns",
#         )
#         return fig, ax

#     def plot_dist_returns(self):

#         fig, ax = self.plot.plot_box(
#             df=self.portfolio_returns_df,
#             title="",
#             xlabel="Portfolio Fitness",
#             ylabel="Daily Returns (%)",
#         )
#         return fig, ax

#     def plot_corr(self):
#         fig, ax = self.plot.plot_heatmap(
#             df=self.portfolio_returns_df,
#             relation_type="corr",
#             title="",
#             annotate=True,
#         )
#         return fig, ax

#     def plot_cov(self):
#         fig, ax = self.plot.plot_heatmap(
#             df=self.portfolio_returns_df,
#             relation_type="cov",
#             title="",
#             annotate=True,
#         )
#         return fig, ax

#     def plot_asset_weights(self):
#         fig, ax = self.plot.plot_trend(
#             df=self.asset_weights_df,
#             title="",
#             xlabel="Date",
#             ylabel="Returns",
#         )
#         return fig, ax

#     def plot_asset_weights_dist(self):
#         fig, ax = self.plot.plot_box(
#             df=self.asset_weights_df,
#             title="",
#             xlabel="Date",
#             ylabel="Cumulative Returns",
#             yscale="symlog",
#         )
#         return fig, ax

#     def plot_mean_sd(self, annualized=True):

#         mean_sd = self.mean_sd.copy()

#         if annualized:
#             mean_sd["mean"] *= self.annualized_days
#             mean_sd["sd"] *= np.sqrt(self.annualized_days)

#         fig, ax = self.plot.plot_scatter_seaborn(
#             data=mean_sd,
#             y="mean",
#             x="sd",
#             hue=mean_sd.index,
#             title="",
#             xlabel="Volatility (SD)",
#             ylabel="Expected Returns",
#         )
#         return fig, ax


# class MultiPortoflio:
#     def __init__(
#         self,
#         name: str,
#         objectives_list: list,
#         risk_type="standard",
#         risk_sample_num=100,
#         risk_sample_size=20,
#         risk_agg_func="median",
#         risk_free_rate=0.0,
#         annualized_days=252,
#         backend="neal",
#         annealing_time=100,
#         scipy_params={"maxiter": 1000, "disp": False, "ftol": 1e-10},
#         target_return=0.1,
#         target_sd=0.1,
#         weight_bound=(0.0, 1.0),
#         init_point=None,
#     ):

#         # args
#         self._name = name
#         self._objectives_list = objectives_list
#         self._risk_type = risk_type
#         self._risk_sample_num = risk_sample_num
#         self._risk_sample_size = risk_sample_size
#         self._risk_agg_func = risk_agg_func
#         self._risk_free_rate = risk_free_rate
#         self._annualized_days = annualized_days
#         self._backend = backend
#         self._annealing_time = annealing_time
#         self._scipy_params = scipy_params
#         self._target_return = target_return
#         self._target_sd = target_sd
#         self._weight_bound = weight_bound
#         self._init_point = init_point

#         self.plot = Plot()

#         self.portfolios = {}

#         for objective in self._objectives_list:
#             self.portfolios[objective] = Portfolio(
#                 name=objective,
#                 objective=objective,
#                 risk_type=self._risk_type,
#                 risk_sample_num=self._risk_sample_num,
#                 risk_sample_size=self._risk_sample_size,
#                 risk_agg_func=self._risk_agg_func,
#                 risk_free_rate=self._risk_free_rate,
#                 annualized_days=self._annualized_days,
#                 backend=self._backend,
#                 annealing_time=self._annealing_time,
#                 scipy_params=self._scipy_params,
#                 target_return=self._target_return,
#                 target_sd=self._target_sd,
#                 weight_bound=self._weight_bound,
#                 init_point=self._init_point,
#             )

#     def run(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

#         mean_sd_list = []
#         portfolio_results_list = [
#             self.portfolios[o].run(asset_list, date_start, date_end)
#             for o in self._objectives_list
#         ]
#         annualized_days = portfolio_results_list[0]["portfolio_config"][
#             "annualized_days"
#         ]
#         mean_sd_list = [r["portfolio_mean_sd"] for r in portfolio_results_list]

#         portfolio_mean_sd = pd.concat(mean_sd_list, axis=0)
#         portfolio_mean_sd["mean"] *= annualized_days
#         portfolio_mean_sd["sd"] *= np.sqrt(annualized_days)

#         market_mean_sd = portfolio_results_list[0]["market_mean_sd"]
#         market_mean_sd["mean"] *= annualized_days
#         market_mean_sd["sd"] *= np.sqrt(annualized_days)

#         # random portfolios
#         n = 1000
#         returns_np = portfolio_results_list[0]["asset_returns"].to_numpy()
#         cov = portfolio_results_list[0]["asset_returns"].cov().to_numpy()

#         r_list = []
#         for i in range(n):
#             w_rand = np.random.random((1, cov.shape[0]))
#             w_rand = w_rand / w_rand.sum()
#             r = returns_np.dot(w_rand.T).mean() * annualized_days
#             c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0] * np.sqrt(annualized_days)
#             r_list.append({"mean": r, "sd": c})
#         mean_sd_random = pd.DataFrame(r_list)

#         return {
#             "market_mean_sd": market_mean_sd,
#             "portfolio_mean_sd": portfolio_mean_sd,
#             "mean_sd_random": mean_sd_random,
#         }


# class PlotMultiPortfolio:
#     def __init__(self, performance) -> None:
#         self.performance = performance
#         self.plot = Plot()

#     def plot_mean_sd(self):
#         fig, ax = self.plot.plot_scatter_portfolio_random(
#             df_1=self.performance["market_mean_sd"],
#             df_2=self.performance["portfolio_mean_sd"],
#             df_3=self.performance["mean_sd_random"],
#             title="Expected Returns vs. Volatility",
#             xlabel="Volatility (SD)",
#             ylabel="Expected Returns",
#         )
#         return fig, ax
