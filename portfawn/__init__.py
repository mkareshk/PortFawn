import logging

from portfawn.backtest import BackTest
from portfawn.models import ClassicOptModel, QuantumOptModel, RiskModel
from portfawn.portfolio import (
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    RandomPortfolio,
)

from .models import OptimizationModel, RiskModel

logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] (%(name)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.WARNING,
)

logger = logging.getLogger(__name__)
