# Data related
MINIMUM_NUMBER_OF_TICKETS = 1500
N_TRAIN_QUARTERS = 36
TOP_N = 5

# Model hyperparameters
N_TREES = 10
OBJECTIVE = "binary"
METRIC = "binary_logloss"
# COLS_TO_FILTER = [
#     "sp500_change__minus_730",
#     "sp500_change__minus_365",
#     "sp500_change__minus_120",
#     "stock_change__minus_730",
#     "stock_change__minus_365",
#     "stock_change__minus_120",
#     "std__minus_730",
#     "std__minus_365",
#     "std__minus_120"
# ]
COLS_TO_FILTER = [
    "sp500_change__minus_730",
    "sp500_change__minus_365",
    "sp500_change__minus_120",
    "stock_change__minus_730",
    "stock_change__minus_365",
    "stock_change__minus_120",
    "std__minus_730",
    "std__minus_365",
    "std__minus_120",
    "EBITDAEV",
    "EBITEV",
    "RevenueEV",
    "CashOnHandEV",
    "PFCF",
    "PE",
    "PB",
    "ROC",
    "DividendYieldLastYear",
    "EPS_minus_EarningsPerShare_change_1_years",
    "EPS_minus_EarningsPerShare_change_2_years",
    "FreeCashFlowPerShare_change_1_years",
    "FreeCashFlowPerShare_change_2_years",
    "OperatingCashFlowPerShare_change_1_years",
    "OperatingCashFlowPerShare_change_2_years",
    "EBITDA_change_1_years",
    "EBITDA_change_2_years",
    "EBIT_change_1_years",
    "EBIT_change_2_years",
    "Revenue_change_1_years",
    "Revenue_change_2_years",
    "NetCashFlow_change_1_years",
    "NetCashFlow_change_2_years",
    "CurrentRatio_change_1_years",
    "CurrentRatio_change_2_years",
]
