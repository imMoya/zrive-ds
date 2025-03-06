import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from plotnine import ggplot, geom_histogram, aes, geom_col, coord_flip,geom_bar,scale_x_discrete, geom_point, theme,element_text, geom_hline, annotate, geom_abline
from config import TOP_N

def get_weighted_performance_of_stocks(df, metric):
    df["norm_prob"] = 1 / len(df)
    return np.sum(df["norm_prob"] * df[metric])


def get_top_tickers_per_prob(preds, train_set, test_set, top_n):
    if len(preds) == len(train_set):
        data_set = train_set
    elif len(preds) == len(test_set):
        data_set = test_set
    else:
        raise ValueError("Not matching train/test")
    
    data_set = data_set.assign(prob=preds)
    return data_set.nlargest(top_n, 'prob')

def top_wt_performance(preds, train_set, test_set, top_n):
    top_dataset = get_top_tickers_per_prob(preds, train_set, test_set, top_n)
    return [("weighted-return", get_weighted_performance_of_stocks(top_dataset, "diff_ch_sp500"), True)]


def split_train_test_by_period(data_set, test_execution_date, n_train_quarters=None, include_nulls_in_test=False):
    train_set = data_set.loc[data_set["execution_date"] <= pd.to_datetime(test_execution_date) - pd.Timedelta(350, unit="day")]
    train_set = train_set[~pd.isna(train_set["diff_ch_sp500"])]
    
    execution_dates = train_set.sort_values("execution_date")["execution_date"].unique()
    if n_train_quarters is not None:
        train_set = train_set[train_set["execution_date"].isin(execution_dates[-n_train_quarters:])]
    
    test_set = data_set.loc[data_set["execution_date"] == test_execution_date]
    if not include_nulls_in_test:
        test_set = test_set[~pd.isna(test_set["diff_ch_sp500"])]
    test_set = test_set.sort_values("date", ascending=False).drop_duplicates("Ticker", keep="first")
    
    return train_set, test_set



def get_columns_to_remove():
    columns_to_remove = [
                         "date",
                         "improve_sp500",
                         "Ticker",
                         "freq",
                         "set",
                         "close_sp500_365",
                         "close_365",
                         "stock_change_365",
                         "sp500_change_365",
                         "stock_change_div_365",
                         "stock_change_730",
                         "sp500_change_365",
                         "stock_change_div_730",
                         "diff_ch_sp500",
                         "diff_ch_avg_500",
                         "execution_date","target","index","quarter","std_730","count","close_0", 
                         "close_sp500_0", 
                         "sp500_change_730",]
        
    return columns_to_remove

def train_model(train_set, test_set, hyperparams):
    if hyperparams["cols_to_filter"] is not None:
        X_train = train_set[hyperparams["cols_to_filter"]]
        X_test = test_set[hyperparams["cols_to_filter"]]
    else:
        columns_to_remove = get_columns_to_remove()
        X_train = train_set.drop(columns=columns_to_remove, errors="ignore")
        X_test = test_set.drop(columns=columns_to_remove, errors="ignore")
    y_train = train_set["target"]
    y_test = test_set["target"]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
        "random_state": 1, "verbosity": -1, "n_jobs": 10
    }
    params.update(hyperparams)
    
    eval_result = {}
    model = lgb.train(
        params=params, train_set=lgb_train, valid_sets=[lgb_test, lgb_train],
        feval=lambda preds, train_data: top_wt_performance(preds, train_set, test_set, top_n=TOP_N),
        callbacks=[lgb.record_evaluation(eval_result)]
    )
    
    return model, eval_result, X_train, X_test

def run_model_for_execution_date(execution_date, data_set, all_results, all_predicted_tickers_list, all_models, hyperparams, n_train_quarters=36, include_nulls_in_test=False):
    train_set, test_set = split_train_test_by_period(data_set, execution_date, n_train_quarters=n_train_quarters,  include_nulls_in_test=include_nulls_in_test)
    if train_set.empty or test_set.empty:
        return all_results, all_predicted_tickers_list, all_models, None, None, None
    
    model, evals_result, X_train, X_test = train_model(train_set, test_set, hyperparams)
    test_set['prob'] = model.predict(X_test)
    predicted_tickers = test_set.sort_values('prob', ascending=False)
    predicted_tickers["execution_date"] = execution_date
    
    all_results[execution_date] = evals_result
    all_models[execution_date] = model
    all_predicted_tickers_list.append(predicted_tickers)
    
    return all_results, all_predicted_tickers_list, all_models, model, X_train, X_test

def draw_feature_importance(model, top = 15):
    fi = model.feature_importance()
    fn = model.feature_name()
    feature_importance = pd.DataFrame([{"feature":fn[i],"imp":fi[i]} for i in range(len(fi))])
    feature_importance = feature_importance.sort_values("imp",ascending = False).head(top)
    feature_importance = feature_importance.sort_values("imp",ascending = True)
    plot = ggplot(feature_importance,aes(x = "feature",y  = "imp")) + geom_col(fill = "lightblue") + coord_flip() +  scale_x_discrete(limits = feature_importance["feature"])
    return plot

def parse_results_into_df(all_results, set_):
    df = pd.DataFrame()
    for date in all_results:
        df_tmp = pd.DataFrame(all_results[(date)][set_])
        df_tmp["n_trees"] = list(range(len(df_tmp)))
        df_tmp["execution_date"] = date
        df= pd.concat([df,df_tmp])
    
    df["execution_date"] = df["execution_date"].astype(str)
    
    return df

def report_images(all_results, all_models, fig_folder: str):
    test_results = parse_results_into_df(all_results, "valid_0")
    train_results = parse_results_into_df(all_results, "training")

    # Get final tree results for each execution date
    test_results_final_tree = test_results.sort_values(["execution_date", "n_trees"]).drop_duplicates("execution_date", keep="last")
    train_results_final_tree = train_results.sort_values(["execution_date", "n_trees"]).drop_duplicates("execution_date", keep="last")

    # Compute mean weighted return
    mean_test_return = test_results_final_tree["weighted-return"].mean()
    mean_train_return = train_results_final_tree["weighted-return"].mean()

    fig_folder = Path(fig_folder)

    # Offset for text placement (so it doesn’t overlap the line)
    offset = 0.05 * (test_results_final_tree["weighted-return"].max() - test_results_final_tree["weighted-return"].min())

    fig_folder = Path(fig_folder)

    # Create test plot
    fig_test = (ggplot(test_results_final_tree) +
                geom_point(aes(x="execution_date", y="weighted-return")) +
                geom_abline(intercept=mean_test_return, slope=0, linetype="dashed", color="red", size=1) +
                annotate("text", 
                         x=test_results_final_tree["execution_date"].iloc[-1], 
                         y=mean_test_return + offset,  # Moves the text up
                         label=f"Mean: {mean_test_return:.4f}", 
                         ha="right", size=8, color="red") +
                theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1)))

    fig_test.save(fig_folder / "test.png")

    # Create train plot
    offset = 0.05 * (train_results_final_tree["weighted-return"].max() - train_results_final_tree["weighted-return"].min())

    fig_train = (ggplot(train_results_final_tree) +
                 geom_point(aes(x="execution_date", y="weighted-return")) +
                 geom_abline(intercept=mean_train_return, slope=0, linetype="dashed", color="red", size=1) +
                 annotate("text", 
                          x=train_results_final_tree["execution_date"].iloc[-1], 
                          y=mean_train_return + offset,  # Moves the text up
                          label=f"Mean: {mean_train_return:.4f}", 
                          ha="right", size=8, color="red") +
                 theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1)))

    fig_train.save(fig_folder / "train.png")

    # Save feature importance plots
    for model in all_models:
        plot = draw_feature_importance(all_models[model])
        feature_imp_folder = fig_folder / "feat_imp"
        feature_imp_folder.mkdir(parents=True, exist_ok=True)
        plot.save(feature_imp_folder / f"{str(model)}.png", width=10, height=10)


        