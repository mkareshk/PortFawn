import numpy as np


class ExpectedStats:
    def __init__(self, returns, optimization_type):

        self.returns = returns
        self.optimization_type = optimization_type

    def _get_sample_set(self, sample_type):

        if sample_type == "full":
            return self.returns.values

        elif sample_type == "random":
            selected_rows = np.random.randint(
                low=0, high=len(self.returns), size=int(len(self.returns) / 5)
            )
            return self.returns.values[selected_rows, :]

        elif sample_type == "weighted_tail":
            p = np.linspace(start=0.01, stop=1.0, num=len(self.returns))
            p = p / sum(p)
            selected_rows = np.random.choice(
                range(len(self.returns)), size=len(self.returns), p=p
            )
            return self.returns.values[selected_rows, :]

    def expected_mean_cov(self, sample_type, instance_num=10):

        if sample_type == "full":
            sample = self._get_sample_set(sample_type)
            mean, cov = np.mean(sample, axis=0), np.cov(sample.T)

        elif sample_type in ["random", "weighted_tail"]:
            instance_num = np.max(5, instance_num)
            sample_set = []
            for i in range(instance_num):
                np.random.seed(int(np.sqrt(time.time()) * i))
                sample_set.append(self._get_sample_set(sample_type))
            sample_set = np.array(sample_set[0])
            mean, cov = np.mean(sample_set, axis=0), np.cov(sample_set.T)

        if self.optimization_type in ["binary_sa", "binary_qpu"]:
            mean = (mean - mean.min()) / (
                mean.max() - mean.min()
            )  # scale normalization
            cov = (cov - cov.mean()) / cov.std()  # z-score normalization
            mean, cov = np.rint(mean), np.rint(cov)

        return mean, cov
