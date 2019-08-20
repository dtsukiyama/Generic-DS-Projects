import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from utils.pipeline import FeatureSelector, FeatureOneHot, dummySelector, LogTransformer

class Processing(object):

    @classmethod
    def clean(cls, data):
        """
        Args: data
        Returns:
            - sorts data
            - convert date string to datetime
            - creates month and day features
            - log price features
            - create host revenue feature
        """
        data = data.sort_values(['vehicle_id','date'])
        data['date'] = pd.to_datetime(data['date'])
        data['day'] = data['date'].dt.dayofweek
        data['week'] = data['date'].dt.week
        data['month'] = data['date'].dt.month
        data['log_price'] = np.log(data['price'])
        data['log_tmv'] = np.log(data['tmv'])
        data['host_revenue'] = data['is_booked']*data['price']

        data = pd.merge(data,
                        data.groupby('date').sum()['is_booked']\
                        .reset_index()\
                        .rename(columns={'is_booked':'day_bookings'}),
                         how = 'left',
                         on = 'date')
        return data

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
    def additional_features(cls, data):
        # daily aggreate demand
        mean_demand = data.groupby('date').mean()['demand_supply_ratio']\
        .reset_index()\
        .rename(columns={'demand_supply_ratio':'mean_demand'})

        median_demand = data.groupby('date').median()['demand_supply_ratio']\
        .reset_index()\
        .rename(columns={'demand_supply_ratio':'median_demand'})

        var_demand = data.groupby('date').var()['demand_supply_ratio']\
        .reset_index()\
        .rename(columns={'demand_supply_ratio':'var_demand'})

        total_demand = data.groupby('date').sum()['demand_supply_ratio']\
        .reset_index()\
        .rename(columns={'demand_supply_ratio':'sum_demand'})

        dfs = [mean_demand, median_demand, var_demand, total_demand]
        for df in dfs:
            data = pd.merge(data, df, how = 'left', on = 'date')
        return data

    @classmethod
    def time_demand(cls, data):
        data = data.set_index('date').resample('D')\
        .agg({'demand_supply_ratio':'sum','price':'mean'})\
        .reset_index().rename(columns={'demand_supply_ratio':'sum_demand'})
        data['day'] = data['date'].dt.dayofweek
        data['week'] = data['date'].dt.week
        data['month'] = data['date'].dt.month
        data.rename(columns={'date':'ds','price':'y'}, inplace=True)
        return data

    @classmethod
    def original_constructor(cls, feature):
        return Pipeline([('selector', FeatureSelector(column=feature))
                         ])

    @classmethod
    def categorical_constructor(cls, feature):
        return Pipeline([('selector', FeatureOneHot(column=feature)),
                         ('standard', OneHotEncoder()),
                        ])

    @classmethod
    def scale_constructor(cls, feature):
        return Pipeline([('selector', FeatureSelector(column=feature)),
                         ('standard', StandardScaler())
                         ])

    @classmethod
    def log_constructor(cls, feature):
        return Pipeline([('selector', LogTransformer(column=feature))
                         ])
