# get data
import dafin

data_instance = dafin.ReturnsData(["AAPL", "GOOGL"])
returns_data = data_instance.get_returns("2022-01-01", "2022-12-30")
print(returns_data)


# portfolio
import portfawn

# random portfolio
random_portfolio = portfawn.portfolio.RandomPortfolio()
random_portfolio.fit(returns_data)
returns_portfolio = random_portfolio.evaluate(returns_data)
print(returns_portfolio)

# equally weighted
equally_weighted_portfolio = portfawn.portfolio.EquallyWeightedPortfolio()
equally_weighted_portfolio.fit(returns_data)
returns_portfolio = equally_weighted_portfolio.evaluate(returns_data)
print(returns_portfolio)

# mean variance
mean_var_portfolio = portfawn.portfolio.MeanVariancePortfolio()
mean_var_portfolio.fit(returns_data)
returns_portfolio = mean_var_portfolio.evaluate(returns_data)
print(returns_portfolio)
