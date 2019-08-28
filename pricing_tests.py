from utils.models import Pricing
from utils.utils import Processing

payload = {'date':'2018-11-14',
           'tmv':19525,
           'category_grouped':'compact',
           'demand_supply_ratio':4.70207,
           'supply':30}

def pricing(payload):
    demand, supply, features = Processing.clean_payload(payload)
    pricing_model = Pricing(1)
    return pricing_model.predict(demand, supply)

def test_pricing():
    assert pricing(payload) == 22.0
