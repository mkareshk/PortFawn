import numpy as np
import pandas as pd


class Sampling:
    def __init__(self, data_returns, sampling_params):

        if sampling_params["type"] == "standard":  # simple, but unstable
            self._expected_return = data_returns.mean()
            self._expected_risk = data_returns.cov()

        elif sampling_params["type"] == "bootstrapping":  # for robust stats

            sample_size = sampling_params["sample_size"]
            agg_func = sampling_params["agg_func"]
            risk_func = sampling_params["risk_func"]

            # # check for params
            if not all(
                i in sampling_params
                for i in ["sample_size", "sample_num", "agg_func", "risk_func"]
            ):
                raise Exception(
                    "sample_size, sample_num, and agg_func should be passed when using bootstrapping for sampling"
                )

            # draw samples
            return_list = []
            risk_list = []
            for i in range(sampling_params["sample_num"]):

                sample = data_returns.sample(n=sample_size)

                if agg_func == "mean":
                    return_list.append(sample.mean())
                elif agg_func == "median":
                    return_list.append(sample.median())
                else:
                    raise NotImplemented

                if risk_func == "cov":
                    risk_list.append(sample.cov())
                elif risk_func == "corr":
                    risk_list.append(sample.corr())
                else:
                    raise NotImplemented

            return_df = pd.DataFrame(return_list)
            risk_matrix = np.array([i.to_numpy() for i in risk_list])

            if agg_func == "mean":
                self._expected_return = return_df.mean()
                risk_matrix = np.mean(risk_matrix, axis=0)
            elif agg_func == "median":
                risk_matrix = np.median(risk_matrix, axis=0)
            else:
                raise NotImplemented

            self._expected_risk = pd.DataFrame(risk_matrix)
            self._expected_risk.columns = return_df.columns
            self._expected_risk.index = return_df.columns

        else:
            raise NotImplemented(
                "The sampling param should be standard or bootstrapping"
            )

    @property
    def expected_return(self):
        return self._expected_return

    @property
    def expected_risk(self):
        return self._expected_risk
