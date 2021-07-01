class ExpectedStats:
    def __init__(self, data_returns, state_type):

        self.data_returns = data_returns
        self.state_type = state_type

    @property
    def expected_return(self):

        if self.state_type == 'simple':
            return self.data_returns.mean()

    @property
    def expected_risk(self):

        if self.state_type == 'simple':
            return self.data_returns.std()
