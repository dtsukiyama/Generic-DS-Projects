import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from utils.pipeline import FeatureSelector, FeatureOneHot, dummySelector, LogTransformer

class Processing(object):

    @classmethod
    def clean_payload(cls, payload):
        data = pd.DataFrame({'demand_supply_ratio':payload['demand_supply_ratio'],
                             'day':datetime.strptime(payload['date'], '%Y-%m-%d').day,
                             'week':datetime.strptime(payload['date'], '%Y-%m-%d').isocalendar()[1],
                             'month':datetime.strptime(payload['date'], '%Y-%m-%d').month,
                             'tmv':payload['tmv'],
                             'category_grouped':payload['category_grouped']}, index = [0])
        demand =  payload['demand_supply_ratio']
        supply = payload['supply']
        return demand, supply, data

    @classmethod
    def clean_expected_payload(cls, payload):
        data = pd.DataFrame({'demand_supply_ratio':payload['demand_supply_ratio'],
                             'day':datetime.strptime(payload['date'], '%Y-%m-%d').day,
                             'week':datetime.strptime(payload['date'], '%Y-%m-%d').isocalendar()[1],
                             'month':datetime.strptime(payload['date'], '%Y-%m-%d').month,
                             'tmv':payload['tmv'],
                             'price':payload['price'],
                             'category_grouped':payload['category_grouped']}, index = [0])
        demand =  payload['demand_supply_ratio']
        supply = payload['supply']
        return demand, supply, data
