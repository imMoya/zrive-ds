""" This is a dummy example to show how to import code from src/ for testing"""
from datetime import datetime

from src.module_1 import Meteo


def test_madrid():
    madrid_data = Meteo(cities=['Madrid']).data
    assert madrid_data["month"].iloc[0] == datetime(2010, 1, 1)
    assert madrid_data["month"].iloc[0] == datetime(2010, 1, 1)