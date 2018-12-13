import arff
import argparse
import logging
import matplotlib.pyplot as plt
import openmlcontrib
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.ensemble
import os

import evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--performances_path', type=str,
                        default=os.path.expanduser('~') + '/projects/sklearn-bot/data/svc.arff')
    parser.add_argument('--metafeatures_path', type=str,
                        default=os.path.expanduser('~') + '/projects/sklearn-bot/data/metafeatures.arff')
    parser.add_argument('--output_directory', type=str,
                        default=os.path.expanduser('~') + '/experiments/meta-models')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    with open(args.performances_path, 'r') as fp:
        arff_performances = arff.load(fp)
    performances = openmlcontrib.meta.arff_to_dataframe(arff_performances, None)
    with open(args.metafeatures_path, 'r') as fp:
        arff_metafeatures = arff.load(fp)
    metafeatures = openmlcontrib.meta.arff_to_dataframe(arff_metafeatures, None)

    results = []
    precision_at_n = 20
    cv_iterations = 5
    for idx, task_id in enumerate(performances['task_id'].unique()):
        logging.info('Processing task %d (%d/%d)' % (task_id, idx+1, len(performances['task_id'].unique())))
        frame_task = performances.loc[performances['task_id'] == task_id]
        frame_other = performances.loc[performances['task_id'] != task_id]

        X = frame_task[['svc__gamma', 'svc__C']].values
        y = frame_task['predictive_accuracy'].values

        poly_transform = sklearn.preprocessing.PolynomialFeatures(2)
        gamma_complexity_poly = poly_transform.fit_transform(X)[1:]

        quadratic_model = sklearn.linear_model.LinearRegression()
        score_te, score_tr = evaluation.cross_validate_surrogate(quadratic_model, gamma_complexity_poly, y, cv_iterations, precision_at_n)
        results.append({'task_id': task_id, 'strategy': 'quadratic_surrogate', 'set': 'test', precision_at_n: score_te})
        results.append({'task_id': task_id, 'strategy': 'quadratic_surrogate', 'set': 'train', precision_at_n: score_tr})

        random_forest_model = sklearn.ensemble.RandomForestRegressor(n_estimators=16)
        score_te, score_tr = evaluation.cross_validate_surrogate(random_forest_model, gamma_complexity_poly, y, cv_iterations, precision_at_n)
        results.append({'task_id': task_id, 'strategy': 'rf_surrogate', 'set': 'test', precision_at_n: score_te})
        results.append({'task_id': task_id, 'strategy': 'rf_surrogate', 'set': 'train', precision_at_n: score_tr})
    result_frame = pd.DataFrame(results)

    os.makedirs(args.output_directory, exist_ok=True)
    fig, ax = plt.subplots()
    sns.boxplot(x="strategy", y=precision_at_n, hue="set", data=result_frame, ax=ax)
    plt.savefig(os.path.join(args.output_directory, 'metamodels.png'))


if __name__ == '__main__':
    run(parse_args())
