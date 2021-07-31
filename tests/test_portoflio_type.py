
import shutil
import hashlib
import unittest
from pathlib import Path

from portfawn.portfolio import BackTesting, BackTestAnalysis
from tests.utils import get_normal_param

kwargs = get_normal_param()
kwargs["portfolio_types"] = ['SA']
kwargs["n_jobs"] = 1

# backtesting
portfolio_backtesting = BackTesting(**kwargs)
portfolio_backtesting.run()


class TestPortfolioType(unittest.TestCase):

    def run_portfolio(self, kwargs):
        kwargs = get_normal_param()
        kwargs["portfolio_types"] = ['SA']

        # backtesting
        portfolio_backtesting = BackTesting(**kwargs)
        portfolio_backtesting.run()

        # analysis
        hash = hashlib.md5("".join([str(i) for i in kwargs.values()]).encode(
            "utf-8")).hexdigest()[0:6]
        dirname = f"results_{hash}"
        analysis = BackTestAnalysis(portfolio_backtesting,
                                    result_path=Path(dirname))
        analysis.plot()

        shutil.rmtree(dirname, ignore_errors=True)

    def test_sampling_standard(self):
        kwargs = get_normal_param()
        kwargs["sampling_params"] = {"type": "standard"}
        self.run_portfolio(kwargs)

    def test_sampling_mean_cov(self):
        kwargs = get_normal_param()
        kwargs["sampling_params"] = {"type": "bootstrapping", "sample_size": 10,
                                     "sample_num": 49, 'agg_func': 'mean', 'risk_func': 'cov'}
        self.run_portfolio(kwargs)

    def test_sampling_mean_corr(self):
        kwargs = get_normal_param()
        kwargs["sampling_params"] = {"type": "bootstrapping", "sample_size": 10,
                                     "sample_num": 49, 'agg_func': 'mean', 'risk_func': 'corr'}
        self.run_portfolio(kwargs)

    def test_sampling_median_cov(self):
        kwargs = get_normal_param()
        kwargs["sampling_params"] = {"type": "bootstrapping", "sample_size": 10,
                                     "sample_num": 49, 'agg_func': 'median', 'risk_func': 'cov'}
        self.run_portfolio(kwargs)

    def test_sampling_median_corr(self):
        kwargs = get_normal_param()
        kwargs["sampling_params"] = {"type": "bootstrapping", "sample_size": 10,
                                     "sample_num": 49, 'agg_func': 'median', 'risk_func': 'corr'}
        self.run_portfolio(kwargs)
