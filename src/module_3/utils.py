import pandas as pd


def extract_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
	df[datetime_col] = pd.to_datetime(df[datetime_col])
	df['year'] = df[datetime_col].dt.year
	df['month'] = df[datetime_col].dt.month
	df['day'] = df[datetime_col].dt.day
	df['hour'] = df[datetime_col].dt.hour
	df['minute'] = df[datetime_col].dt.minute
	return df
