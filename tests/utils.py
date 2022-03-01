import matplotlib

OBJECTIVES_QUANTUM = ["BMOP"]
OBJECTIVES_CLASSIC = ["MRP", "MVP", "MSRP"]
OBJECTIVES_MISC = ["EWP"]
OBJECTIVES = OBJECTIVES_QUANTUM + OBJECTIVES_CLASSIC + OBJECTIVES_MISC

ASSET_LIST = ["SPY", "GLD", "BND"]

DATE_START = "2020-11-01"
DATE_END = "2020-12-31"
FITTING_DATES = 3 * 5
EVALUATION_DAYS = 1 * 5

RISK_FREE_RATE = 0.01

TARGET_RETURN = 0.2
TARGET_SD = 0.05

WEIGHT_BOUND = (0.02, 0.98)

N_JOBS = 10


def check_figure(fig, ax):
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
