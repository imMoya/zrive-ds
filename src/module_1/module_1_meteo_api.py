""" This is a dummy example """
import openmeteo_requests
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse

import matplotlib.pyplot as plt
import requests_cache
import pandas as pd
from retry_requests import retry
from typing import Dict
from references import API_URL, COORDINATES, VARIABLES, INI_DATE, END_DATE
from dataclasses import dataclass

@dataclass
class City:
    name: str
    latitude: float
    longitude: float

@dataclass
class Date:
    start_date: str
    end_date: str

class Meteo:
    def __init__(self):
        self.setup()

    @property
    def date(self):
        return Date(INI_DATE, END_DATE) 
    
    def setup(self) -> None:
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)

    def get_data_meteo_api(self, city_name: str):
        city = City(name=city_name, latitude=COORDINATES[city_name]['latitude'], longitude=COORDINATES[city_name]['longitude'])
        date = self.date
        params = self.define_params(city, date)
        print(params)
        self.responses = self.call_api(API_URL, params)
        return self.responses
    
    def call_api(self, api_url, params):
        # TODO: validate response and handle possible errors
        return self.openmeteo.weather_api(api_url, params=params)
    
    @staticmethod
    def define_params(
        city: City,
        date: Date
    ) -> Dict: 
        return {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "start_date": date.start_date,
            "end_date": date.end_date,
            "hourly": VARIABLES
        }
    
    @property
    def data(self):
        return self.get_hourly_response(response)

    @staticmethod
    def get_hourly_response(response: WeatherApiResponse) -> pd.DataFrame:
        hourly = response.Hourly()
        print(hourly.Interval())
        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        for i, val in enumerate(VARIABLES):
            hourly_data[val] = hourly.Variables(i).ValuesAsNumpy()
        return pd.DataFrame(hourly_data)
    
    @staticmethod
    def compute_additional_params(df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        df_monthly = df.groupby('month').agg(
            temperature_2m_mean=('temperature_2m', 'mean'),
            precipitation_sum=('precipitation', 'sum'),
            wind_speed_10m_max=('wind_speed_10m', 'max')
        ).reset_index()
        df_monthly['month'] = df_monthly['month'].dt.to_timestamp()
        return df_monthly

    # TODO: add plotting capabilities
    # TODO: compute variables from dataframe
    # TODO: reformat code


if __name__ == "__main__":
    meteo = Meteo()
    
    response = meteo.get_data_meteo_api("Madrid")[0]
    df = meteo.get_hourly_response(response)
    df = meteo.compute_additional_params(df)
    plt.plot(df["month"], df["temperature_2m_mean"])
    plt.show()
