# Portfawn: Optimizing Financial Portfolios with Classical and Quantum Techniques

Portfawn is an advanced library for constructing and optimizing financial portfolios using state-of-the-art classical and quantum optimization methods. Designed for flexibility, it allows you to fine-tune portfolio construction with customizable objectives, adaptable risk models, and cutting-edge optimization techniques—all through a single, streamlined package.


## Why Choose Portfawn?

- **Optimize Your Way**: Achieve your portfolio goals with customizable optimization objectives:
  - **Classical Objectives**: Minimum variance, maximum Sharpe ratio, target returns, and more.
  - **Quantum Objectives**: Innovative quantum optimization for complex problems.
- **Flexibility**: Configure every aspect—risk models, constraints, backend systems, and more.
- **Single-Command Installation**: Install the package with tailored dependencies for quantum or classical setups using optional tags.
- **Seamless Backtesting**: Validate your strategies with automated workflows.
- **User-Friendly API**: Designed to simplify complex financial engineering tasks.


## Key Features

### Advanced Portfolio Optimization

1. **Classical Optimization Objectives**:
   - **Minimum Variance Portfolio (MVP)**: Minimize portfolio risk.
   - **Maximum Return Portfolio (MRP)**: Maximize portfolio return.
   - **Maximum Sharpe Ratio Portfolio (MSRP)**: Maximize returns per unit risk.
   - **Customizable Parameters**: Define constraints, target returns, risk-free rates, and optimization precision.

2. **Quantum Optimization Objectives**:
   - **Binary Mean-Optimal Portfolio (BMOP)**: Solve portfolio problems using quantum annealing.
   - Backend flexibility:
     - Simulated annealing via `neal`.
     - Quantum processing units (QPUs) via D-Wave.

### Unified Interface
Portfawn provides a unified API for both classical and quantum optimization, making it easy to switch between methods with minimal code changes.


## Installation

You can install Portfawn and its dependencies in a single command:

- **Basic installation** (classical optimization only):
  ```bash
  pip install portfawn
  ```

- **With quantum optimization support**:
  ```bash
  pip install portfawn"[quantum]"
  ```

This modular installation ensures you only install what you need!


## Getting Started

This example demonstrates how to use Portfawn to optimize portfolios with classical and quantum objectives, run backtests, and visualize results.

#### 1. Import Libraries
```python
import dafin
import matplotlib.pyplot as plt

from portfawn import (
    BackTest,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    RandomPortfolio,
    OptimizationModel,
)
```

#### 2. Define Assets and Load Data
```python
# Define assets and date range
asset_list = ["SPY", "GLD", "BND"]
date_start = "2010-01-01"
date_end = "2022-12-30"

# Load returns data
returns_data = dafin.ReturnsData(asset_list).get_returns(date_start, date_end)
```

#### 3. Create and Configure Portfolios
```python
portfolio_list = [
    RandomPortfolio(),
    EquallyWeightedPortfolio(),
    MeanVariancePortfolio(name="Minimum Variance Portfolio", optimization_model=OptimizationModel(objective="MVP")),
    MeanVariancePortfolio(name="Maximum Return Portfolio", optimization_model=OptimizationModel(objective="MRP")),
    MeanVariancePortfolio(name="Maximum Sharpe Ratio Portfolio", optimization_model=OptimizationModel(objective="MSRP")),
    MeanVariancePortfolio(name="Binary Multi-Objective Portfolio", optimization_model=OptimizationModel(objective="BMOP")),
]
```

#### 4. Run Backtesting
```python
backtest = BackTest(
    portfolio_list=portfolio_list,
    asset_list=asset_list,
    date_start=date_start,
    date_end=date_end,
    fitting_days=22,
    evaluation_days=5,
    n_jobs=2,
)
backtest.run()
```

#### 5. Visualize and Save Results

```python
fig, ax = backtest.plot_cum_returns()
plt.savefig("plot_cum_returns.png")
```
<img src="assets/plot_cum_returns.png" alt="Description" width="700" />


```python
fig, ax = backtest.plot_dist_returns()
plt.savefig("plot_dist_returns.png")
```
<img src="assets/plot_dist_returns.png" alt="Description" width="700" />


```python
fig, ax = backtest.plot_corr()
plt.savefig("plot_corr.png")
```
<img src="assets/plot_corr.png" alt="Description" width="700" />


```python
fig, ax = backtest.plot_cov()
plt.savefig("plot_cov.png")
```
<img src="assets/plot_cov.png" alt="Description" width="700" />


## Demo

Check out the demos in `demo/`.


## License

This project is licensed under the [MIT License](LICENSE).


## Support

Need help or have feedback? Get in touch:
- **Email**: mkareshk@outlook.com
- **Issues**: [GitHub Issues](https://github.com/mkareshk/portfawn/issues)
