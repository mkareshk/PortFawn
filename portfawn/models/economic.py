class EconomicModel:
    def __init__(self, risk_free_rate: float = 0.0, annualized_days: int = 252) -> None:
        self._risk_free_rate = risk_free_rate
        self._annualized_days = annualized_days

    @property
    def risk_free_rate(self):
        return self._risk_free_rate

    @property
    def annualized_days(self):
        return self._annualized_days
