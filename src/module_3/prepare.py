"prepare.py"

from pathlib import Path

import pandas as pd


def filter_five_items_inside_order(df: pd.DataFrame) -> pd.DataFrame:
    df_filter = df.groupby('order_id').filter(lambda x: x.shape[0] >= 5)
    return df_filter


if __name__ == '__main__':
    DATA_DIR = Path('datasets/module_3/')
    # Load the data and filter the orders with at least 5 items
    df = pd.read_csv(Path(DATA_DIR, 'feature_frame.csv'))
    df = filter_five_items_inside_order(df)
    df.to_csv(Path(DATA_DIR, 'feature_frame_filtered.csv'), index=False)
    # Split the data into train, validation and test
