import pandas as pd


def map_ordered_items_to_prices(df: pd.DataFrame, inventory_df: pd.DataFrame, list_col_name: str):
    price_dict = inventory_df.set_index('variant_id')['compare_at_price'].to_dict()
    return df[list_col_name].apply(lambda x: [price_dict.get(item, None) for item in x])


def sum_prices(price_list: list) -> float:
    return sum(float(price) for price in price_list if price is not None)


def has_none_value(lst: list) -> bool:
    return None in lst


def summary_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    user_orders_summary = df.groupby('user_id').agg(
        num_orders=pd.NamedAgg(column='id', aggfunc='count'),
        total_spent=pd.NamedAgg(column='items_total_price', aggfunc='sum')
    ).reset_index()
    return user_orders_summary


def summary_abandoned_df(df: pd.DataFrame) -> pd.DataFrame:
    user_abandoned_summary = df.groupby('user_id').agg(
        num_orders=pd.NamedAgg(column='id', aggfunc='count'),
        total_abandoned=pd.NamedAgg(column='abandoned_total_price', aggfunc='sum')
    ).reset_index()
    return user_abandoned_summary

def extract_hour(df: pd.DataFrame, datetime_col: str, hour_col: str) -> pd.DataFrame:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[hour_col] = df[datetime_col].dt.hour
    return df


