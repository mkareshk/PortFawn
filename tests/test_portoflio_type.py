
import shutil
import hashlib
import unittest
from pathlib import Path

from portfawn.portfolio import BackTesting, BackTestAnalysis
from tests.utils import get_normal_param

kwargs = get_normal_param()
# kwargs["portfolio_types"] = ['MSR', 'SA']
kwargs["n_jobs"] = 1
kwargs["asset_list"] = ['BND', 'SPY', 'GLD']
# backtesting
portfolio_backtesting = BackTesting(**kwargs)
portfolio_backtesting.run()
hash = hashlib.md5("".join([str(i) for i in kwargs.values()]).encode(
    "utf-8")).hexdigest()[0:6]
dirname = f"results_{hash}"
analysis = BackTestAnalysis(portfolio_backtesting,
                            result_path=Path(dirname))
analysis.plot()
# class TestPortfolioType(unittest.TestCase):

#     def run_portfolio(self, kwargs):
#         kwargs = get_normal_param()
#         kwargs["portfolio_types"] = [
#             # "Equal",
#             # "MV",
#             # "MR",
#             # "MSR",
#             "SA",
#         ]
#         kwargs["n_jobs"] = 1

#         # backtesting
#         portfolio_backtesting = BackTesting(**kwargs)
#         portfolio_backtesting.run()

#         # analysis
#         hash = hashlib.md5("".join([str(i) for i in kwargs.values()]).encode(
#             "utf-8")).hexdigest()[0:6]
#         dirname = f"results_{hash}"
#         analysis = BackTestAnalysis(portfolio_backtesting,
#                                     result_path=Path(dirname))
#         analysis.plot()

#         # shutil.rmtree(dirname, ignore_errors=True)

#     def test_portfolio_type_sa(self):
#         kwargs = get_normal_param()
#         kwargs["sampling_params"] = {"type": "standard"}
#         self.run_portfolio(kwargs)
