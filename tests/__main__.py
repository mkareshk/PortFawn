from pathlib import Path

from portfawn.portfolio import BackTesting, BackTestAnalysis
from tests.utils import get_normal_param

kwargs = get_normal_param()

# backtesting
portfolio_backtesting = BackTesting(**kwargs)
portfolio_backtesting.run()

# analysis
analysis = BackTestAnalysis(
    portfolio_backtesting,
    result_path=Path(f"results_test"),
)
analysis.store_params(kwargs)
analysis.plot()
