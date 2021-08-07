import shutil
import hashlib
import unittest
from pathlib import Path

from portfawn.portfolio import BackTesting, BackTestAnalysis
from tests.utils import get_normal_param


class TestPortfolioType(unittest.TestCase):
    def run_portfolio(self, kwargs):

        # backtesting
        portfolio_backtesting = BackTesting(**kwargs)
        portfolio_backtesting.run()

        # analysis
        hash = hashlib.md5(
            "".join([str(i) for i in kwargs.values()]).encode("utf-8")
        ).hexdigest()[0:6]
        dirname = f"results_{hash}"
        analysis = BackTestAnalysis(portfolio_backtesting, result_path=Path(dirname))
        analysis.plot()

        shutil.rmtree(dirname, ignore_errors=True)
