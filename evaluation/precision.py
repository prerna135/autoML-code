import numpy as np
import scipy.stats
import sklearn


def precision_at_n(y_real: np.ndarray, y_hat: np.ndarray, top_n: int):
    y_hat_ranks = scipy.stats.rankdata(y_hat, method='average')
    test_y_ranks = scipy.stats.rankdata(y_real, method='average')
    y_hat_maxargs = y_hat_ranks.argsort()
    test_y_maxargs = test_y_ranks.argsort()
    cnt = 0
    for entry in y_hat_maxargs[:top_n]:
        if entry in test_y_maxargs[:top_n]:
            cnt += 1
    return cnt / top_n


def evaluate_fold(model: sklearn.base.RegressorMixin, X_tr: np.ndarray,
                  y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                  top_n: int, use_k: int):
    new_model = sklearn.base.clone(model)
    new_model.fit(X_tr, y_tr)
    experiments = {
        'tr': (X_tr, y_tr),
        'te': (X_te, y_te),
    }

    precision_score = dict()
    spearman_score = dict()
    for exp_type, (X, y) in experiments.items():
        y_hat = new_model.predict(X)
        rand_indices = np.random.randint(len(X), size=use_k)
        precision_score[exp_type] = precision_at_n(y[rand_indices], y_hat[rand_indices], top_n)
        spearman_score[exp_type] = scipy.stats.pearsonr(y[rand_indices], y_hat[rand_indices])[0]
    return precision_score['te'], precision_score['tr'], spearman_score['te'], spearman_score['tr']


def cross_validate_surrogate(model: sklearn.base.RegressorMixin, data: np.ndarray,
                             targets: np.ndarray, n_folds: int, top_n: int, use_k: int):
    kf = sklearn.model_selection.KFold(n_splits=n_folds, random_state=42, shuffle=True)
    splits = kf.split(data)

    precision_scores_te = list()
    precision_scores_tr = list()
    spearman_scores_te = list()
    spearman_scores_tr = list()
    for tr_idx, te_idx in splits:
        X_tr, y_tr = data[tr_idx], targets[tr_idx]
        X_te, y_te = data[te_idx], targets[te_idx]
        prec_te, prec_tr, spearm_te, spearm_tr = evaluate_fold(model, X_tr, y_tr, X_te, y_te, top_n, use_k)
        precision_scores_te.append(prec_te)
        precision_scores_tr.append(prec_tr)
        spearman_scores_te.append(spearm_te)
        spearman_scores_tr.append(spearm_tr)

    return np.mean(precision_scores_te), \
           np.mean(precision_scores_tr), \
           np.mean(spearman_scores_te), \
           np.mean(spearman_scores_tr)
