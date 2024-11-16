import dafin

from portfawn import (
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    OptimizationModel,
    RandomPortfolio,
)

# params
assets_list = ["SPY", "GLD", "BND"]
date_start = "2010-01-01"
date_end = "2022-12-30"

data_instance = dafin.ReturnsData(assets_list)
returns_data = data_instance.get_returns(date_start, date_end)

# portfolio optimization
mean_vafiance_portfolio = [
    MeanVariancePortfolio(optimization_model=OptimizationModel(objective=o))
    for o in ["MVP", "MSRP", "BMOP"]
]
portfolio_list = [
    RandomPortfolio(),
    EquallyWeightedPortfolio(),
]
portfolio_list.extend(mean_vafiance_portfolio)

for portfolio in portfolio_list:
    portfolio.fit(returns_data)
    performance = portfolio.evaluate(returns_data)
    print(portfolio)
