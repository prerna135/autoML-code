import logging
import numpy as np
import scipy.stats
import sklearn


def precision_at_n(y_real, y_hat, top_n):
    y_hat_ranks = scipy.stats.rankdata(y_hat, method='average')
    test_y_ranks = scipy.stats.rankdata(y_real, method='average')
    y_hat_maxargs = y_hat_ranks.argsort()
    test_y_maxargs = test_y_ranks.argsort()
    cnt = 0
    for entry in y_hat_maxargs[:top_n]:
        if entry in test_y_maxargs[:top_n]:
            cnt += 1
    return cnt / top_n


def cross_validate_surrogate(model, X, y, n_folds, top_n):
    kf = sklearn.model_selection.KFold(n_splits=n_folds, random_state=42, shuffle=True)
    splits = kf.split(X)

    precision_scores_te = []
    precision_scores_tr = []
    for train_idx, test_idx in splits:
        train_x, train_y = X[train_idx], y[train_idx]
        test_x, test_y = X[test_idx], y[test_idx]
        new_model = sklearn.base.clone(model)
        new_model.fit(train_x, train_y)
        y_hat_te = new_model.predict(test_x)
        y_hat_tr = new_model.predict(train_x)
        precision_scores_te.append(precision_at_n(test_y, y_hat_te, top_n))
        precision_scores_tr.append(precision_at_n(train_y, y_hat_tr, top_n))
    return np.mean(precision_scores_te), np.mean(precision_scores_tr)
