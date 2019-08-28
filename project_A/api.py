import numpy as np

from flask import Flask, request
from utils.models import Scoring, Pricing
from utils.utils import Processing

application = Flask(__name__)

@application.route('/predict', methods=['POST'])
def predict():
    """
    Args: json
        - For example if this was an API

        payload = {'date':'2018-11-14',
                   'tmv':19525,
                   'category_grouped':'compact',
                   'demand_supply_ratio':4.70207,
                   'supply':30}

    """
    if request.method == 'POST':
        payload = request.get_json()
        demand, supply, features = Processing.clean_payload(payload)
        pricing_model = Pricing(1)
        return {'price': pricing_model.predict(demand, supply)}

if __name__=='__main__':
    application.run(host='0.0.0.0', port=4000)
