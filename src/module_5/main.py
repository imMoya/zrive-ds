import pandas as pd
import numpy as np
from pathlib import Path
from utils import (
    run_model_for_execution_date,
    report_images
)
from config import *

if __name__ == "__main__":
    # Read dataset
    ROOT = Path(__file__).resolve().parent
    DATASET = ROOT / "data" / "financials_against_return.feather"
    data_set = pd.read_feather(DATASET)

    # Filter dataset by min. number of tickets
    df_quarter_lengths = data_set.groupby(["execution_date"]).size().reset_index().rename(columns = {0:"count"})
    data_set = pd.merge(data_set, df_quarter_lengths, on = ["execution_date"])
    data_set = data_set[data_set["count"]>=MINIMUM_NUMBER_OF_TICKETS]

    # Create target
    data_set["diff_ch_sp500"] = data_set["stock_change_div_365"] - data_set["sp500_change_365"]
    data_set.loc[data_set["diff_ch_sp500"]>0,"target"] = 1
    data_set.loc[data_set["diff_ch_sp500"]<0,"target"] = 0
    data_set["target"].value_counts()

    # Train models
    hyperparams = {
        "n_estimators": N_TREES,
        "objective": OBJECTIVE,
        "metric": METRIC,
        "cols_to_filter": COLS_TO_FILTER
    }

    execution_dates = np.sort(data_set['execution_date'].unique())
    all_results = {}
    all_predicted_tickers_list = []
    all_models = {}

    for execution_date in execution_dates:
        print(execution_date)
        run_model_for_execution_date(execution_date, data_set, all_results, all_predicted_tickers_list, all_models, hyperparams, N_TRAIN_QUARTERS ,include_nulls_in_test=False)

    # Report figures
    FIG_FOLDER = ROOT / "figs" / f"n_trees_{N_TREES}_cols_by_price_and_financial_features_top{TOP_N}"
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)
    report_images(all_results, all_models, FIG_FOLDER)
    
