import arff
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument('--poly_degree', type=int, default=2)
    parser.add_argument('--precision_at_n', type=int, default=20)
    parser.add_argument('--precision_out_of_k', type=int, default=100)
    parser.add_argument('--cv_iterations', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_limit', type=int, default=None, help='For fast testing')
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
    # impute missing meta-features with -1 value
    metafeatures = openmlcontrib.meta.arff_to_dataframe(arff_metafeatures, None).set_index('task_id').fillna(-1)
    # remove all non-rbf rows
    performances = performances.loc[performances['svc__kernel'] == 'rbf']
    # join with meta-features frame, and remove tasks without meta-features
    performances = performances.join(metafeatures, on='task_id', how='inner')

    results = []
    precision_name = 'precision_at_%d_out_%d' % (args.precision_at_n, args.precision_out_of_k)
    spearman_name = 'spearmanr_%d' % args.precision_out_of_k

    # sklearn objects
    quadratic_model = sklearn.linear_model.LinearRegression()
    random_forest_model = sklearn.ensemble.RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_seed)
    poly_transform = sklearn.preprocessing.PolynomialFeatures(args.poly_degree)

    all_tasks = performances['task_id'].unique()
    if args.task_limit is not None:
        all_tasks = all_tasks[:args.task_limit]
    for idx, task_id in enumerate(all_tasks):
        logging.info('Processing task %d (%d/%d)' % (task_id, idx+1, len(all_tasks)))
        frame_task = performances.loc[performances['task_id'] == task_id]
        frame_others = performances.loc[performances['task_id'] != task_id]
        assert(frame_task.shape[0] > 100)

        # some convenience datasets
        param_columns = ['svc__gamma', 'svc__C']
        X_poly_train = poly_transform.fit_transform(frame_others[param_columns].values)[:, 1:]
        X_poly_test = poly_transform.fit_transform(frame_task[param_columns].values)[:, 1:]
        X_poly_meta_train = np.concatenate((X_poly_train, frame_others[metafeatures.columns.values]), axis=1)
        X_poly_meta_test = np.concatenate((X_poly_test, frame_task[metafeatures.columns.values]), axis=1)

        # surrogates
        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.cross_validate_surrogate(quadratic_model,
                                                                                     X_poly_test,
                                                                                     frame_task['predictive_accuracy'].values,
                                                                                     args.cv_iterations,
                                                                                     args.precision_at_n,
                                                                                     args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'quadratic_surrogate', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'quadratic_surrogate', 'set': 'train-obs', precision_name: prec_tr, spearman_name: spearm_tr})

        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.cross_validate_surrogate(random_forest_model,
                                                                                     frame_task[param_columns].values,
                                                                                     frame_task['predictive_accuracy'].values,
                                                                                     args.cv_iterations,
                                                                                     args.precision_at_n,
                                                                                     args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'RF_surrogate', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'RF_surrogate', 'set': 'train-obs', precision_name: prec_tr, spearman_name: spearm_tr})

        # aggregates
        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.evaluate_fold(quadratic_model,
                                                                          X_poly_train,
                                                                          frame_others['predictive_accuracy'].values,
                                                                          X_poly_test,
                                                                          frame_task['predictive_accuracy'].values,
                                                                          args.precision_at_n,
                                                                          args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'quadratic_aggregate', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'quadratic_aggregate', 'set': 'train-tasks', precision_name: prec_tr, spearman_name: spearm_tr})

        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.evaluate_fold(random_forest_model,
                                                                          frame_others[param_columns],
                                                                          frame_others['predictive_accuracy'].values,
                                                                          frame_task[param_columns].values,
                                                                          frame_task['predictive_accuracy'].values,
                                                                          args.precision_at_n,
                                                                          args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'RF_aggregate', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'RF_aggregate', 'set': 'train-tasks', precision_name: prec_tr, spearman_name: spearm_tr})

        # meta-models
        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.evaluate_fold(quadratic_model,
                                                                          X_poly_meta_train,
                                                                          frame_others['predictive_accuracy'].values,
                                                                          X_poly_meta_test,
                                                                          frame_task['predictive_accuracy'].values,
                                                                          args.precision_at_n,
                                                                          args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'quadratic_meta', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'quadratic_meta', 'set': 'train-tasks', precision_name: prec_tr, spearman_name: spearm_tr})

        columns = list(param_columns) + list(metafeatures.columns.values)
        prec_te, prec_tr, spearm_te, spearm_tr = evaluation.evaluate_fold(random_forest_model,
                                                                          frame_others[columns],
                                                                          frame_others['predictive_accuracy'].values,
                                                                          frame_task[columns].values,
                                                                          frame_task['predictive_accuracy'].values,
                                                                          args.precision_at_n,
                                                                          args.precision_out_of_k)
        results.append({'task_id': task_id, 'strategy': 'RF_meta', 'set': 'test', precision_name: prec_te, spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'RF_meta', 'set': 'train-tasks', precision_name: prec_tr, spearman_name: spearm_tr})

    result_frame = pd.DataFrame(results)

    os.makedirs(args.output_directory, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(x="strategy", y=precision_name, hue="set", data=result_frame, ax=ax)
    plt.savefig(os.path.join(args.output_directory, '%s.png' % precision_name))

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(x="strategy", y=spearman_name, hue="set", data=result_frame, ax=ax)
    plt.savefig(os.path.join(args.output_directory, '%s.png' % spearman_name))


if __name__ == '__main__':
    run(parse_args())
