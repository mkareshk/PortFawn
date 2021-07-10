class Sampling:
    def __init__(self, data_returns, sampling_params):

        self.data_returns = data_returns
        self.sampling_params = sampling_params
        self.sampling_method = self.sampling_params["name"]

    @property
    def expected_return(self):

        if self.sampling_method == "simple":
            return self.data_returns.mean()

    @property
    def expected_risk(self):

        if self.sampling_method == "simple":
            return self.data_returns.cov()
