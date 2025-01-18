"""This is a dummy example to show how to import code from src/ for testing"""

from datetime import datetime

import numpy as np

from src.module_1.module_1_meteo_api import Meteo


def test_madrid():
	madrid_data = Meteo(cities=['Madrid']).data
	assert madrid_data['month'].iloc[0] == datetime(2010, 1, 1)
	assert np.isclose(madrid_data['temperature_2m_mean'].iloc[0], 4.398334, rtol=0.0001)
	assert np.isclose(madrid_data['precipitation_sum'].iloc[0], 52.000000, rtol=0.0001)
	assert np.isclose(madrid_data['wind_speed_10m_max'].iloc[0], 30.190672, rtol=0.0001)


def test_london():
	london_data = Meteo(cities=['London']).data
	assert london_data['month'].iloc[0] == datetime(2010, 1, 1)
	assert np.isclose(london_data['temperature_2m_mean'].iloc[0], 1.358538, rtol=0.0001)
	assert np.isclose(london_data['precipitation_sum'].iloc[0], 45.900002, rtol=0.0001)
	assert np.isclose(london_data['wind_speed_10m_max'].iloc[0], 29.627365, rtol=0.0001)


def test_rio():
	rio_data = Meteo(cities=['Rio']).data
	assert rio_data['month'].iloc[0] == datetime(2010, 1, 1)
	assert np.isclose(rio_data['temperature_2m_mean'].iloc[0], 27.916586, rtol=0.0001)
	assert np.isclose(rio_data['precipitation_sum'].iloc[0], 109.699997, rtol=0.0001)
	assert np.isclose(rio_data['wind_speed_10m_max'].iloc[0], 20.192118, rtol=0.0001)
