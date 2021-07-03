class ExpectedStats:
    def __init__(self, data_returns, sampling_method):

        self.data_returns = data_returns
        self.sampling_method = sampling_method

    @property
    def expected_return(self):

        if self.sampling_method == "simple":
            return self.data_returns.mean()

    @property
    def expected_risk(self):

        if self.sampling_method == "simple":
            return self.data_returns.std()
