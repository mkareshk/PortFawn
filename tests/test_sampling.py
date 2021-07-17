from portfawn.portfolio import BackTesting, BackTestAnalysis
from tests.utils import get_normal_param


def test_pipeline():

    param = get_normal_param()
    param["sampling_params"] = {
        "name": "bootstrapping",
        "sample_size": 10,
        "sample_num": 100,
    }

    # backtesting
    portfolio_backtesting = BackTesting(**param)
    portfolio_backtesting.run()

    # analysis
    analysis = BackTestAnalysis(portfolio_backtesting)
    analysis.plot()
