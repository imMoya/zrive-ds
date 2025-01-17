API_URL: str = 'https://archive-api.open-meteo.com/v1/archive'
COORDINATES: dict[str, dict[str, float]] = {
	'Madrid': {'latitude': 40.416775, 'longitude': -3.703790},
	'London': {'latitude': 51.507351, 'longitude': -0.127758},
	'Rio': {'latitude': -22.906847, 'longitude': -43.172896},
}
VARIABLES: list[str] = ['temperature_2m', 'precipitation', 'wind_speed_10m']
INI_DATE: str = '2010-01-01'
END_DATE: str = '2020-12-31'
