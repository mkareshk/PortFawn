import time

import numpy as np


class ExpectedStats:
    def __init__(self, returns_data, optimization_type):

        self.returns_data = returns_data
        self.optimization_type = optimization_type

    def _get_sample_set(self, sample_type, random_selection_ratio):

        if sample_type == "full":
            return self.returns_data.values

        elif sample_type == "random":
            selected_rows = np.random.randint(
                low=0,
                high=len(self.returns_data),
                size=int(len(self.returns_data * random_selection_ratio)),
            )
            return self.returns_data.values[selected_rows, :]

        elif sample_type == "weighted_tail":
            p = np.linspace(start=0.01, stop=1.0, num=len(self.returns_data))
            p = p / sum(p)
            selected_rows = np.random.choice(
                range(len(self.returns_data)), size=len(self.returns_data), p=p
            )
            return self.returns_data.values[selected_rows, :]

    def expected_mean_cov(
        self, sample_type, instance_num=10, random_selection_ratio=0.2
    ):

        if sample_type == "full":
            sample_set = self._get_sample_set(sample_type, random_selection_ratio)

        elif sample_type in ["random", "weighted_tail"]:
            instance_num = np.max(5, instance_num)
            sample_set = []
            for i in range(instance_num):
                np.random.seed(int(np.sqrt(time.time()) * i))
                sample = self._get_sample_set(sample_type, random_selection_ratio)
                sample_set.append(sample)
            sample_set = np.array(sample_set[0])

        mean, cov = np.mean(sample_set, axis=0), np.cov(sample_set.T)

        if self.optimization_type in [
            "binary_sa",
            "binary_qpu",
        ]:  # convert the stats to QUBO-ready ones
            mi = mean.min()
            ma = mean.max()
            mean = (mean - mi) / (ma - mi)  # scale normalization
            cov = (cov - cov.mean()) / cov.std()  # z-score normalization
            mean, cov = np.rint(mean), np.rint(cov)

        return mean, cov
