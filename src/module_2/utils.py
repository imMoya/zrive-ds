def map_ordered_items_to_prices(df, inventory_df, list_col_name):
    price_dict = inventory_df.set_index('variant_id')['compare_at_price'].to_dict()
    return df[list_col_name].apply(lambda x: [price_dict.get(item, None) for item in x])


def sum_prices(price_list: list) -> float:
    return sum(float(price) for price in price_list if price is not None)


def has_none_value(lst: list) -> bool:
    return None in lst

