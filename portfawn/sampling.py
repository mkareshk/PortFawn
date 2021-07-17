import numpy as np
import pandas as pd


class Sampling:
    def __init__(self, data_returns, sampling_params):

        if sampling_params["name"] == "simple":
            self._expected_return = data_returns.mean()
            self._expected_risk = data_returns.cov()

        elif sampling_params["name"] == "bootstrapping":

            if (
                "sample_size" not in sampling_params.keys()
                or "sample_num" not in sampling_params.keys()
            ):
                raise Exception("sample_size and sample_num should pass")

            samples = data_returns.sample(n=sampling_params["sample_size"])
            self._expected_return = np.zeros(samples.mean().shape)
            self._expected_risk = np.zeros(samples.cov().shape)

            for i in range(sampling_params["sample_num"]):

                samples = data_returns.sample(n=sampling_params["sample_size"])

                self._expected_return += samples.mean()
                self._expected_risk += samples.cov()

            self._expected_return /= sampling_params["sample_num"]
            self._expected_risk /= sampling_params["sample_num"]

        else:
            raise NotImplemented

    @property
    def expected_return(self):
        return self._expected_return

    @property
    def expected_risk(self):
        return self._expected_risk
