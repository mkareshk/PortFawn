import numpy as np
import pandas as pd
from dafin import annualize_returns, annualize_sd


def random_portfolio(returns, days_per_year, annualized=True, portfolio_num=1000):

    # returns
    returns_mat = returns.to_numpy()

    # weights
    weights_mat = np.random.random((portfolio_num, returns_mat.shape[1]))
    weights_mat = weights_mat / weights_mat.sum(axis=1)[:, np.newaxis]

    # random portfolio
    random_portoflio = np.matmul(returns_mat, weights_mat.T)
    random_portoflio_df = pd.DataFrame(data=random_portoflio, index=returns.index)
    mean_sd = pd.DataFrame()

    if annualized:
        mean_sd["mean"] = annualize_returns(
            returns=random_portoflio_df,
            days_per_year=days_per_year,
        )
        mean_sd["sd"] = annualize_sd(
            returns=random_portoflio_df,
            days_per_year=days_per_year,
        )
    else:
        mean_sd["mean"] = random_portoflio_df.mean()
        mean_sd["sd"] = random_portoflio_df.std()

    return mean_sd
