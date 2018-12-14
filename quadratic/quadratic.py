import logging
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.linear_model
import sklearn.preprocessing
import typing


class MetaRandomForestQuadratic(sklearn.base.RegressorMixin):

    def __init__(self, n_estimators: int, random_seed: int,
                 meta_columns: typing.List, base_columns: typing.List, poly_degree: int):
        if poly_degree != 2:
            logging.warning('Polynomial degree of 2 assumed. ')
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.meta_columns = meta_columns
        self.base_columns = base_columns
        self.poly_degree = 2
        self.feat_trans = sklearn.preprocessing.PolynomialFeatures(self.poly_degree)
        self.meta_model = sklearn.ensemble.RandomForestRegressor(n_estimators=self.n_estimators,
                                                                 random_state=self.random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.meta_model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a 1D numpy array
        """
        predictions = []
        for idx, row in X.iterrows():
            base_model = sklearn.linear_model.LinearRegression(fit_intercept=False)
            base_model.intercept_ = 0
            base_model.coef_ = self.meta_model.predict([row[self.meta_columns]])[0]
            input = self.feat_trans.fit_transform([row[self.base_columns]])
            prediction = base_model.predict(input)[0]
            predictions.append(prediction)
        res = np.array(predictions)
        return res


def get_coefficient_names() -> typing.List:
    return [
        'intercept',
        'coef_gamma',
        'coef_C',
        'coef_gamma_sq',
        'coef_gamma_C',
        'coef_C_sq',
    ]


def generate_coefficients_data(poly_degree: int, performance_data: pd.DataFrame, param_columns: typing.List) -> pd.DataFrame:
    """
    Pre-processess the coefficients for all datasets at once (for speed)
    """
    if poly_degree != 2:
        logging.warning('Not Implemented: polynomial degree of > 2. Will use degree 2 for meta-model')
    coef_names = get_coefficient_names()
    results = []
    for idx, task_id in enumerate(performance_data['task_id'].unique()):
        frame_task = performance_data.loc[performance_data['task_id'] == task_id]
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        poly_feat = sklearn.preprocessing.PolynomialFeatures(2)
        X = poly_feat.fit_transform(frame_task[param_columns])
        y = frame_task['predictive_accuracy']
        model.fit(X, y)
        result = {
            'task_id': task_id,
            coef_names[0]: model.coef_[0],
            coef_names[1]: model.coef_[1],
            coef_names[2]: model.coef_[2],
            coef_names[3]: model.coef_[3],
            coef_names[4]: model.coef_[4],
            coef_names[5]: model.coef_[5],
        }
        results.append(result)
    return pd.DataFrame(results).set_index('task_id')
