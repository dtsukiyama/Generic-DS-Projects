import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

from utils.utils import Processing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from scipy.stats import poisson
import scipy.optimize as optimize


class Pricing(object):
    def __init__(self, score, elasticity = -2.0, nominal_price = 55):
        self.elasticity = elasticity
        self.nominal_price = nominal_price
        self.score = score

    def price_elasticity(self, price, nominal_demand):
        return nominal_demand * ( price / self.nominal_price ) ** (self.elasticity)

    def objective(self, p_t, nominal_demand=np.array([50,40,30,20])):
        return ((-1.0*self.score) * np.sum( p_t * self.price_elasticity(p_t,
                                                            nominal_demand = nominal_demand) )) / 100

    def constraint_1(self, p_t):
        return p_t

    def constraint_2(self, p_t, supply = 20, forecasted_demand = 35.0):
        return supply - self.price_elasticity(p_t, nominal_demand = forecasted_demand)

    def predict(self, demand, supply):

        demand = [demand]
        starting_values = 10 * np.ones(len(demand))
        bounds = tuple((10.0, 150) for b in starting_values)

        constraints = ({'type': 'ineq', 'fun':  lambda x:  self.constraint_1(x)},
                        {'type': 'ineq', 'fun':  lambda x,
                        supply = supply,
                        forecasted_demand = demand: self.constraint_2(x,
                                                                      supply = supply,
                                                                      forecasted_demand = demand)})

        results = optimize.minimize(self.objective,
                                    starting_values,
                                    args=(demand),
                                    method = 'SLSQP',
                                    bounds = bounds,
                                    constraints = constraints)
        return np.round(results['x'][0])

class Scoring(object):

    tree_path = 'utils/models/classifier.pkl'
    pipe_path = 'utils/models/tree_pipe.pkl'

    @classmethod
    def load_model(cls):
        with open(cls.tree_path, 'rb') as file:
            model = pickle.load(file)
        return model

    @classmethod
    def load_pipe(cls):
        with open(cls.pipe_path, 'rb') as file:
            model = pickle.load(file)
        return model

    @classmethod
    def poisson_predictions(cls, x):
        prediction_poisson = poisson.rvs(mu=x, size=5, random_state=5)
        return np.mean(prediction_poisson)
