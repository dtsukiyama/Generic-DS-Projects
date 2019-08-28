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

class Metrics(object):

    @classmethod
    def logit_to_prob(cls, logit):
        odds = np.exp(logit)
        return odds/(1+odds)

class Model(object):

    """
    class to explore different models
    """

    naive_features = ['category_grouped','month','price']
    kmeans_feature_columns = ['is_booked','demand_supply_ratio',
                              'day','week','month','log_price','log_tmv',
                              'category_compact','category_midsize',
                              'category_suv','category_upscale','category_van']

    final_features = ['is_booked','demand_supply_ratio',
                      'day','week','month','log_tmv',
                      'category_compact','category_midsize',
                      'category_suv','category_upscale','category_van',
                      'clusters','mean_demand','median_demand','var_demand']

    kmeans_path = 'models/kmeans.pkl'
    tree_path = 'models/tree.pkl'
    classifier_path = 'models/classifier.pkl'

    @classmethod
    def process_kmeans(cls, data):
        data = Processing.clean(data)
        dummies = pd.get_dummies(data['category_grouped'])
        data = pd.concat([data.drop('category_grouped', axis = 1), dummies], axis = 1)
        return data[cls.kmeans_feature_columns]

    @classmethod
    def elbow(cls, data):
        """
        https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch11/ch11.ipynb

            - measure distortion to get number of clusters
        """
        distortions = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        plt.show()

    @classmethod
    def kmeans_features(cls, data, N):
        model = KMeans(n_clusters = N)
        model.fit(data[cls.kmeans_feature_columns])
        return model.labels_.tolist()

    @classmethod
    def fit_kmeans(cls, data, N):
        model = KMeans(n_clusters = N)
        model.fit(data[cls.kmeans_features])
        cls.save_model()

    @classmethod
    def statistical_model(cls, data):
        """
        Regression model to get coefficients of independent variables
            - dependent variable: is_booked
        """
        data = Processing.clean(data)
        dummy_categories = pd.get_dummies(data['category_grouped'], prefix='category')
        train = pd.concat([data.drop(['category_grouped','price','date','tmv'], axis = 1), dummy_categories], axis = 1)
        train['intercept'] = 1.0
        train_features = [b for b in train.columns if b not in ['is_booked','category_compact',
                                                                'vehicle_id','host_revenue',
                                                                'day_bookings']]
        logit = sm.Logit(train['is_booked'], train[train_features])
        model = logit.fit()
        print(model.summary())
        return model

    @classmethod
    def baseline_features(cls, data):
        dummy_categories = pd.get_dummies(data['category_grouped'], prefix='category')
        data = pd.concat([data.drop(['category_grouped','price','date','tmv'], axis = 1), dummy_categories], axis = 1)
        train_features = [b for b in data.columns if b not in ['is_booked','vehicle_id','host_revenue','log_price','day_bookings']]
        return data, train_features

    @classmethod
    def classifier_features(cls, data):
        dummy_categories = pd.get_dummies(data['category_grouped'], prefix='category')
        data = pd.concat([data.drop(['category_grouped','price','date','tmv'], axis = 1), dummy_categories], axis = 1)
        train_features = [b for b in data.columns if b not in ['is_booked','vehicle_id','host_revenue','day_bookings']]
        return data, train_features

    @classmethod
    def tree_features(cls, data):
        dummy_categories = pd.get_dummies(data['category_grouped'], prefix='category')
        data = pd.concat([data.drop(['category_grouped','price','tmv'], axis = 1), dummy_categories], axis = 1)
        train_features = [b for b in data.columns if b not in ['date','is_booked','vehicle_id','host_revenue','log_price','day_bookings']]
        return data, train_features

    @classmethod
    def naive_baseline(cls, data):
        """
        Naive baseline price prediction
            - group by category_grouped and month, get median price
        """
        data = Processing.clean(data)
        X_train, X_test, y_train, y_test = train_test_split(data[cls.naive_features],
                                                            data.price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        train_predictions = X_train.groupby(['category_grouped','month'])\
        .median()['price']\
        .reset_index()\
        .rename(columns={'price':'predictions'})

        predictions = pd.merge(X_train, train_predictions, how = 'left', on = ['category_grouped','month'])

        # training error
        training_error = mean_absolute_error(predictions.predictions, y_train)
        print("Training Mean Absolute Error: {}".format(training_error))

        # test
        test_predictions = pd.merge(X_test, train_predictions, how = 'left', on = ['category_grouped','month'])
        testing_error = mean_absolute_error(test_predictions.predictions, y_test)
        print("Testing Mean Absolute Error: {}".format(testing_error))

    @classmethod
    def baseline_linear_model(cls, data):
        """
        Linear regression model
        """
        data = Processing.clean(data)
        data, train_features = cls.baseline_features(data)
        reg = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(data[train_features],
                                                            data.log_price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        test_error = mean_absolute_error(np.exp(predictions), np.exp(y_test))
        print('Mean Absolute Error: {}'.format(test_error))

    @classmethod
    def baseline_tree_model(cls, data):
        """
        Regression tree model
        """
        data = Processing.clean(data)
        data, train_features = cls.baseline_features(data)
        X_train, X_test, y_train, y_test = train_test_split(data[train_features],
                                                            data.log_price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_error = mean_absolute_error(np.exp(predictions), np.exp(y_test))
        print('Mean Absolute Error: {}'.format(test_error))

    @classmethod
    def baseline_tree_model_time(cls, data):
        data = Processing.clean(data)
        data, train_features = cls.tree_features(data)
        # time split

        train = data.query('date < "2019-01-01"')
        test = data.query('date >= "2019-01-01"')

        model = DecisionTreeRegressor(max_depth=5)
        model.fit(train[train_features], train.log_price)
        predictions = model.predict(test[train_features])
        test_error = mean_absolute_error(np.exp(predictions), np.exp(test.log_price))
        print('Mean Absolute Error: {}'.format(test_error))

    @classmethod
    def time_demand_model(cls, data):
        data = Processing.clean(data)
        data = Processing.time_demand(data)
        features = ['sum_demand','day','week','month']
        X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                            data.price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)
        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_error = mean_absolute_error(np.exp(predictions), np.exp(y_test))
        print('Mean Absolute Error: {}'.format(test_error))

    @classmethod
    def tree_model(cls, data):
        data = Processing.clean(data)
        data = cls.tree_features(data)
        data = Processing.additional_features(data)
        data['clusters'] = cls.kmeans_features(data[cls.kmeans_feature_columns], 3)
        data.fillna(0, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(data[cls.final_features],
                                                            data.log_price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_error = mean_absolute_error(np.exp(predictions), np.exp(y_test))
        print('Mean Absolute Error: {}'.format(test_error))

    @classmethod
    def baseline_classifier(cls, data):
        data = Processing.clean(data)
        train, train_features = cls.classifier_features(data)
        X_train, X_test, y_train, y_test = train_test_split(train[train_features],
                                                            train.is_booked,
                                                            test_size=0.10,
                                                            random_state=1234)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        preds = clf.predict(X_test)
        print('Accuracy Score:{}'.format(accuracy_score(y_test, preds)))
        print('ROC score: {}'.format(roc_auc_score(y_test, probs)))
        print('Classification report: {}'.format(print(classification_report(y_test, preds))))

class Train(object):

    tree_path = 'utils/models/tree.pkl'
    classifier_path = 'utils/models/classifier.pkl'
    pipe_path = 'utils/models/pipe.pkl'
    tree_pipe_path = 'utils/models/tree_pipe.pkl'

    categorical_features = ['category_grouped']
    log_features = ['tmv']
    continuous_features = ['demand_supply_ratio','week','day','month']


    @classmethod
    def save_model(cls, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    @classmethod
    def features(cls):
        """
        build pipeline for regression tree model
        """

        # build categorical feature dicionary
        feature_pipe = dict()
        for col in cls.categorical_features:
            feature_pipe[col] = Processing.categorical_constructor(col)

        # build continuous log features
        for col in cls.log_features:
            feature_pipe[col] = Processing.log_constructor(col)

        for col in cls.continuous_features:
            feature_pipe[col] = Processing.original_constructor(col)

        feature_list = []
        for key in feature_pipe.keys():
            feature_list.append((key, feature_pipe[key]))

        features = FeatureUnion(feature_list)
        pipe = Pipeline([('features', features)])
        return pipe

    @classmethod
    def booked_features(cls):
        """
        build pipeline for booked classfier model
        """

        # build categorical feature dicionary
        feature_pipe = dict()
        for col in cls.categorical_features:
            feature_pipe[col] = Processing.categorical_constructor(col)

        # build continuous log features
        for col in cls.log_features:
            feature_pipe[col] = Processing.log_constructor(col)

        for col in cls.continuous_features:
            feature_pipe[col] = Processing.original_constructor(col)

        feature_list = []
        for key in feature_pipe.keys():
            feature_list.append((key, feature_pipe[key]))

        features = FeatureUnion(feature_list)
        pipe = Pipeline([('features', features)])
        return pipe

    @classmethod
    def train_tree_model(cls, data):
        """
        Train regression tree model
        """
        data = Processing.clean(data)
        pipe = cls.features()
        pipe.fit(data)
        cls.save_model(pipe, cls.pipe_path)
        X = pipe.transform(data)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            data.log_price,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_error = mean_absolute_error(np.exp(predictions), np.exp(y_test))
        print('Mean Absolute Error: {}'.format(test_error))
        cls.save_model(model, cls.tree_path)

    @classmethod
    def train_expected_value(cls, data):
        data = Processing.clean(data)
        pipe = cls.booked_features()
        pipe.fit(data)
        cls.save_model(pipe, cls.tree_pipe_path)
        X = pipe.transform(data)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            data.is_booked,
                                                            stratify=data.month,
                                                            test_size=0.20,
                                                            random_state=1234)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        preds = clf.predict(X_test)
        print('Accuracy Score:{}'.format(accuracy_score(y_test, preds)))
        print('ROC score: {}'.format(roc_auc_score(y_test, probs)))
        print('Classification report: {}'.format(print(classification_report(y_test, preds))))
        cls.save_model(clf, cls.classifier_path)

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
