"""This is a dummy example"""

import os
import warnings
import requests
from requests.exceptions import HTTPError, Timeout, RequestException
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import openmeteo_requests  # type: ignore
import pandas as pd  # type: ignore
import requests_cache
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse  # type: ignore
from retry_requests import retry  # type: ignore

from src.module_1.references import (
	API_URL,
	COORDINATES,
	END_DATE,
	INI_DATE,
	VARIABLES,
)


@dataclass
class City:
	name: str
	latitude: float
	longitude: float


@dataclass
class Date:
	start_date: str
	end_date: str


class FetchDataError(Exception):
    pass


class Meteo:
	def __init__(self, cities: list[str] | None):
		self.setup()
		self.cities = cities

	@property
	def date(self) -> Date:
		return Date(INI_DATE, END_DATE)

	def setup(self) -> None:
		cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
		retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
		self.openmeteo = openmeteo_requests.Client(session=retry_session)

	def get_data_meteo_api(self, city_name: str) -> WeatherApiResponse:
		city = City(
			name=city_name,
			latitude=COORDINATES[city_name]['latitude'],
			longitude=COORDINATES[city_name]['longitude'],
		)
		date = self.date
		params = self.define_params(city, date)
		self.responses = self.call_api(API_URL, params)
		return self.responses

	def call_api(self, api_url: str, params: dict[str, Any]) -> WeatherApiResponse:
		response = self.openmeteo.weather_api(api_url, params=params)
		return response
	
	@staticmethod
	def call_api_requests(api_url: str, params: dict[str, Any]):
		try:
			response = requests.get(url=api_url, params=params)
			response.raise_for_status()
			return response.json()
		except HTTPError as http_err:
			raise FetchDataError(f"HTTP error occurred: {http_err}") from http_err
		except Timeout as timeout_err:
			raise FetchDataError(f"Request timed out: {timeout_err}") from timeout_err
		except RequestException as req_err:
			raise FetchDataError(f"An error occurred during the request: {req_err}") from req_err
		except ValueError as json_err:
			raise FetchDataError(f"Error decoding JSON: {json_err}") from json_err

	@staticmethod
	def define_params(city: City, date: Date) -> dict[str, Any]:
		return {
			'latitude': city.latitude,
			'longitude': city.longitude,
			'start_date': date.start_date,
			'end_date': date.end_date,
			'hourly': VARIABLES,
		}

	@property
	def data(self) -> pd.DataFrame:
		df = pd.DataFrame({})
		if self.cities is None:
			raise ValueError('`cities` is None, but should be a list of str')
		for city in self.cities:
			df_ = self.get_city_data(city)
			df = pd.concat([df, df_])
		return df

	def get_city_data(self, city: str) -> pd.DataFrame:
		response = self.get_data_meteo_api(city)[0]
		df = self.get_hourly_response(response)
		df = self.compute_additional_params(df)
		df['city'] = city
		return df

	@staticmethod
	def get_hourly_response(response: WeatherApiResponse) -> pd.DataFrame:
		hourly = response.Hourly()
		hourly_data = {
			'date': pd.date_range(
				start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
				end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
				freq=pd.Timedelta(seconds=hourly.Interval()),
				inclusive='left',
			)
		}
		for i, val in enumerate(VARIABLES):
			hourly_data[val] = hourly.Variables(i).ValuesAsNumpy()
		return pd.DataFrame(hourly_data)

	@staticmethod
	def compute_additional_params(df: pd.DataFrame) -> pd.DataFrame:
		warnings.filterwarnings(
			'ignore',
			category=UserWarning,
			message='Converting to PeriodArray/Index representation '
			'will drop timezone information',
		)
		df['date'] = pd.to_datetime(df['date'])
		df['month'] = df['date'].dt.to_period('M')
		df_monthly = (
			df.groupby('month')
			.agg(
				temperature_2m_mean=('temperature_2m', 'mean'),
				precipitation_sum=('precipitation', 'sum'),
				wind_speed_10m_max=('wind_speed_10m', 'max'),
			)
			.reset_index()
		)
		df_monthly['month'] = df_monthly['month'].dt.to_timestamp()
		return df_monthly

	@staticmethod
	def plot_weather_parameter(
		df: pd.DataFrame,
		parameter: str = 'temperature_2m_mean',
		title: str | None = '',
		output_filename: Path | str | None = None,
	) -> None:
		if parameter not in df.columns:
			raise ValueError(f"Parameter '{parameter}' not found in DataFrame.")
		fig = plt.figure(figsize=(16, 8))
		for city, city_data in df.groupby('city'):
			plt.plot(
				city_data['month'],
				city_data[parameter],
				label=city,
				marker='o',
				linestyle='-',
			)
		plt.title(
			title if title else f'{parameter.capitalize()} Across Cities', fontsize=16
		)
		plt.xlabel('Month', fontsize=14)
		plt.ylabel(parameter.replace('_', ' ').capitalize(), fontsize=14)
		plt.legend(title='Cities', fontsize=12)
		plt.grid()
		plt.xticks(rotation=45)
		plt.tight_layout()
		if output_filename:
			fig.savefig(output_filename)


if __name__ == '__main__':
	script_dir = os.path.dirname(os.path.abspath(__file__))
	FIG_FOLDER = os.path.join(script_dir, 'figs')
	os.makedirs(FIG_FOLDER, exist_ok=True)
	data = Meteo(['Madrid', 'London', 'Rio']).data
	print(data)
	parameters = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']
	[
		Meteo.plot_weather_parameter(
			data,
			parameter=parameter,
			output_filename=os.path.join(FIG_FOLDER, f'{parameter}.png'),
		)
		for parameter in parameters
	]
	# Additional example of API call via requests... Put 400º of latitude to handle error...
	city = City(name="Madrid", latitude= 400.416775, longitude= -3.703790)
	date = Date(INI_DATE, END_DATE)
	Meteo.call_api_requests(API_URL, Meteo.define_params(city, date))
