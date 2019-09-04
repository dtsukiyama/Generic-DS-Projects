import pandas as pd
import numpy as np
import gc
import csv
import json
from datetime import datetime

class IO:

    @classmethod
    def read(cls, path):
        """
        Args: path to file
        Returns: json data loaded into memory
        """
        data = []
        with open('data/transactions.txt') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    @classmethod
    def yield_keys(cls, generator):
        return next(generator).keys()

    @classmethod
    def get_keys(cls, path):
        """
        Args: path to file
        Returns: generator of json data

            - Here I want to only know what the keys are so I don't want to load all the data
            - use yield_keys function to return keys

        """
        data = []
        with open('data/transactions.txt') as f:
            for line in f:
                yield json.loads(line)

    @classmethod
    def gen_values(cls, keys, generator):
        lookup = dict()
        for key in keys:
            lookup[key] = generator[key]
        return lookup.values()

    @classmethod
    def write_csv(cls, input_path, output_path):
        """
            - write json to csv
            - it is simpler to use the convert function, the tradeoff is memory and speed
        """
        csv_data = open(output_path, 'w')
        csvwriter = csv.writer(csv_data)
        generator = cls.get_keys(input_path)
        keys = cls.yield_keys(generator)
        rows = cls.get_keys(input_path)
        header = sorted(list(keys))
        count = 0
        for row in rows:
            if count == 0:
                csvwriter.writerow(header)
                count += 1
            csvwriter.writerow(cls.gen_values(header, row))
        csv_data.close()

    def find_null(data):
        """
        Args: data
            - Pandas dataframe
        Returns: list of columns which are entirely null
        """
        N = len(data)
        records = data.isnull().sum()
        lookup = dict()
        for col, value in zip(records.index, records.values):
            lookup[col] = value/N

        # which columns have all null?
        lookup = {k:v for k,v in lookup.items() if v == 1.0}
        return list(lookup.keys())

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
