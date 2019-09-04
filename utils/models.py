import pandas as pd
import numpy as np
import xgboost as xgb

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from utils.pipeline import FeatureSelector, LogTransformer
from utils.utils import IO
import scipy.optimize as optimize

class Metrics:

    def probabilityOp(prediction_data):
        """
        Find number of cases correctly classified at different thresholds
        """
        count = None
        probs = []
        for decile in [np.round(x * 0.1, 1) for x in range(1, 10)]:
            count = Counter(prediction_data.query('prob > @decile')['test'])
            if len(count) == 2:
                probs.append(count[1]*1.0/(count[0]+count[1]))
            else:
                probs.append(1)

        del count
        return list(zip([np.round(x * 0.1, 1) for x in range(1, 10)], probs))

class Processing(object):
    """
    Contains:
        - classmethods for constructing different feature types (One Hot Encode, Log, Scale)
    """


    dollar_features = ['Amount']
    time_features = ['Time']


    remove_cols = ['accountNumber','customerId','cardCVV','enteredCVV','cardLast4Digits',
                   'transactionDateTime','currentExpDate','accountOpenDate','dateOfLastAddressChange',
                   'merchantName']

    dollar_features_b = ['creditLimit','transactionAmount','currentBalance']
    continuous_features = ['availableMoney']
    base_features = ['credit_share']
    null_value_features = ['acqCountry','merchantCountryCode','transactionType']

    @classmethod
    def original_constructor(cls, feature):
        return Pipeline([('selector', FeatureSelector(column=feature))
                         ])

    @classmethod
    def categorical_constructor(cls, feature):
        return Pipeline([('selector', FeatureSelector(column=feature)),
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

    @classmethod
    def clean(cls, data):
        """
        Args: data
        Returns: cleaned data
        """

        for col in cls.dollar_features:
            data.loc[data[col] == 0, col] = 0.01

        return data

    @classmethod
    def features(cls, data):
        """
        Args: data
        Returns: Scikit-Learn pipeline of baseline feature transformations
        """
        base_features = data.columns
        base_features = [b for b in base_features if b not in ['Time','Amount','Class']]

        # build categorical feature dicionary
        feature_pipe = dict()

        # build continuous log features
        for col in cls.dollar_features:
            feature_pipe[col] = cls.log_constructor(col)

        # build continuous scale features
        for col in cls.time_features:
            feature_pipe[col] = cls.scale_constructor(col)

        # row stats
        for col in base_features:
            feature_pipe[col] = cls.original_constructor(col)

        feature_list = []
        for key in feature_pipe.keys():
            feature_list.append((key, feature_pipe[key]))

        features = FeatureUnion(feature_list)
        pipe = Pipeline([('features', features)])
        return pipe

    @classmethod
    def clean_b(cls, data):
        """
        Args: data
        Returns: cleaned data
        """

        null_lookup = IO.find_null(data)
        data.drop(null_lookup, axis=1, inplace=True)

        # convert to datetime, create time features
        data['transactionDateTime'] = pd.to_datetime(data['transactionDateTime'])
        data['hour'] = data['transactionDateTime'].dt.hour
        data['day'] = data['transactionDateTime'].dt.dayofweek

        # boolean for if cardCVV and enteredCVV match
        data['cvv_match'] = data['cardCVV'] == data['enteredCVV']
        data.drop(cls.remove_cols, axis = 1, inplace=True)

        # percent available money vs credit limit
        data['credit_share'] = 	data['availableMoney']/data['creditLimit']

        # boolean for acqCountry/merchantCountryCode match
        data['country_match'] = data['acqCountry'] == data['merchantCountryCode']

        # columns that perhaps should not be numeric
        convert_cols = ['posEntryMode','posConditionCode','hour','day']
        for col in convert_cols:
            data[col] = data[col].fillna(0)

        for col in convert_cols:
            data[col] = data[col].astype(str)

        # categorical columns with null values
        for col in cls.null_value_features:
            data[col] = data[col].fillna('None')

        for col in cls.dollar_features_b:
            data.loc[data[col] == 0, col] = 0.01

        return data

    @classmethod
    def features_b(cls, data):
        """
        Args: data
        Returns: Scikit-Learn pipeline of baseline feature transformations
        """

        categorical_features = data.select_dtypes(include=['object']).columns
        boolean_features = data.select_dtypes(include=['bool']).columns

        # build categorical feature dicionary
        feature_pipe = dict()
        for col in categorical_features:
            feature_pipe[col] = cls.categorical_constructor(col)

        # boolean are categorical
        for col in boolean_features:
            feature_pipe[col] = cls.categorical_constructor(col)

        # build continuous log features
        for col in cls.dollar_features_b:
            feature_pipe[col] = cls.log_constructor(col)

        # build continuous scale features
        for col in cls.continuous_features:
            feature_pipe[col] = cls.scale_constructor(col)

        # row stats
        for col in cls.base_features:
            feature_pipe[col] = cls.original_constructor(col)

        feature_list = []
        for key in feature_pipe.keys():
            feature_list.append((key, feature_pipe[key]))

        features = FeatureUnion(feature_list)
        pipe = Pipeline([('features', features)])
        return pipe


class Models(object):

    """
    contains:
        - baseline model method
            used to build an intial model to measure performance against
    """

    baseline_parameters = {'min_child_weight': 50,
                           'eta': 0.1,
                           'colsample_bytree': 0.3,
                           'max_depth': 8,
                           'subsample': 0.8,
                           'lambda': 1.,
                           'nthread': -1,
                           'booster' : 'gbtree',
                           'silent': 1,
                           'eval_metric': 'auc',
                           'objective': 'binary:logistic'}
    baseline_rounds = 1000

    @classmethod
    def baseline(cls, X, y):
        X_train, X_holdout, y_train, y_holdout = train_test_split(X,
                                                                  y,
                                                                  test_size=0.1,
                                                                  random_state=4321)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=1234)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(cls.baseline_parameters,
                          dtrain,
                          cls.baseline_rounds,
                          watchlist,
                          early_stopping_rounds=10,
                          maximize=False,
                          verbose_eval=1)

        # holdout
        dholdout = xgb.DMatrix(X_holdout)
        preds = model.predict(dholdout)
        classes = np.array(preds) > 0.5
        classes = classes.astype(int)
        print('ROC score: {}'.format(roc_auc_score(y_holdout, preds)))
        print('Classification report: {}'.format(print(classification_report(y_holdout, classes))))
        prediction_data = pd.DataFrame({'prob':preds,
                                        'test':y_holdout})
        print(Metrics.probabilityOp(prediction_data))

        return model

    @classmethod
    def resample_baseline(cls, X, y):
        X_train, X_holdout, y_train, y_holdout = train_test_split(X,
                                                                  y,
                                                                  test_size=0.1,
                                                                  random_state=4321)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=1234)

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(cls.baseline_parameters,
                          dtrain,
                          cls.baseline_rounds,
                          watchlist,
                          early_stopping_rounds=10,
                          maximize=False,
                          verbose_eval=1)

        # holdout
        dholdout = xgb.DMatrix(X_holdout)
        preds = model.predict(dholdout)
        classes = np.array(preds) > 0.5
        classes = classes.astype(int)
        print('ROC score: {}'.format(roc_auc_score(y_holdout, preds)))
        print('Classification report: {}'.format(print(classification_report(y_holdout, classes))))
        prediction_data = pd.DataFrame({'prob':preds,
                                        'test':y_holdout})
        print(Metrics.probabilityOp(prediction_data))

        return model


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
