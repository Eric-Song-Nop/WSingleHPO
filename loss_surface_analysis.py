import argparse
from glob import glob
import os
import pickle
from pprint import pprint
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate, KFold, StratifiedKFold,
    StratifiedShuffleSplit, ShuffleSplit,
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    RandomForestRegressor as RF,
    HistGradientBoostingClassifier as HGB,
)

from dscan.dataset_label_cols import get_full_data

from optimizers import PySOTOptimizer

np.set_printoptions(precision=4)


# search space
def get_search_space(method='HGB'):
    lst_hps = {
        'HGB': ['max_iter', 'learning_rate', 'min_samples_leaf', 'l2_regularization'],
        'SVM': ['C', 'gamma', 'tol'],
        'MLP-adam': [
            'hidden_layer_sizes', 'alpha', 'learning_rate_init',
            # PARAMS NOT OPTIMIZED BY ASKL
            # 'batch_size', 'tol', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon',
        ],
        'MLP-adam-v2': [
            'hidden_layer_sizes', 'alpha', 'learning_rate_init', 'nlayers',
            # PARAMS NOT OPTIMIZED BY ASKL
            # 'batch_size', 'tol', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon',
        ],
    }
    api_config = {
        'HGB': {
            lst_hps['HGB'][0]: {'type': 'int', 'space': 'linear', 'range': (10, 200)},
            lst_hps['HGB'][1]: {'type': 'real', 'space': 'log', 'range': (1e-3, 1.0)},
            lst_hps['HGB'][2]: {'type': 'int', 'space': 'linear', 'range': (1, 40)},
            lst_hps['HGB'][3]: {'type': 'real', 'space': 'log', 'range': (1e-4, 1.0)},
        },
        'SVM': {
            lst_hps['SVM'][0]: {'type': 'real', 'space': 'log', 'range': (0.01, 1000.0)},
            lst_hps['SVM'][1]: {'type': 'real', 'space': 'log', 'range': (1e-5, 10.0)},
            lst_hps['SVM'][2]: {'type': 'real', 'space': 'log', 'range': (1e-5, 1e-1)},
        },
        'MLP-adam': {
            lst_hps['MLP-adam'][0]: {"type": "int", "space": "linear", "range": (50, 200)},
            lst_hps['MLP-adam'][1]: {"type": "real", "space": "log", "range": (1e-5, 1e1)},
            lst_hps['MLP-adam'][2]: {
                "type": "real", "space": "log", "range": (1e-5, 1e-1)
            },
            ## PARAMS NOT OPTIMIZER BY ASKL
            # lst_hps['MLP-adam'][3]: {
            #     "type": "int", "space": "linear", "range": (10, 250)
            # },
            # lst_hps['MLP-adam'][4]: {
            #     "type": "real", "space": "log", "range": (1e-5, 1e-1)
            # },
            # lst_hps['MLP-adam'][5]: {
            #     "type": "real", "space": "logit", "range": (0.1, 0.9)
            # },
            # lst_hps['MLP-adam'][6]: {
            #     "type": "real", "space": "logit", "range": (0.5, 0.99)
            # },
            # lst_hps['MLP-adam'][7]: {
            #     "type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)
            # },
            # lst_hps['MLP-adam'][8]: {
            #     "type": "real", "space": "log", "range": (1e-9, 1e-6)
            # },
        },
        'MLP-adam-v2': {
            lst_hps['MLP-adam-v2'][0]: {
                "type": "int", "space": "log", "range": (16, 264)
            },
            lst_hps['MLP-adam-v2'][1]: {
                "type": "real", "space": "log", "range": (1e-5, 1e1)
            },
            lst_hps['MLP-adam-v2'][2]: {
                "type": "real", "space": "log", "range": (1e-5, 1e-1)
            },
            lst_hps['MLP-adam-v2'][3]: {
                "type": "int", "space": "linear", "range": (1, 3)
            },
        },
    }

    default_config = {
        'HGB': {
            lst_hps['HGB'][0]: 100,
            lst_hps['HGB'][1]: 0.1,
            lst_hps['HGB'][2]: 20,
            lst_hps['HGB'][3]: 0. ## <=== THIS IS NOT PART OF THE SEARCH SPACE
        },
        'SVM': {
            lst_hps['SVM'][0]: 1.0,
            lst_hps['SVM'][1]: 0.1,
            lst_hps['SVM'][2]: 1e-3,
        },
        'MLP-adam': {
            lst_hps['MLP-adam'][0]: 100,
            lst_hps['MLP-adam'][1]: 1e-4,
            lst_hps['MLP-adam'][2]: 1e-3,
            ## PARAMS NOT OPTIMIZED BY ASKL
            # lst_hps['MLP-adam'][3]: 'auto', ## <=== NOT PART OF SEARCH SPACE
            # lst_hps['MLP-adam'][4]: 1e-4,
            # lst_hps['MLP-adam'][5]: 0.1,
            # lst_hps['MLP-adam'][6]: 0.9,
            # lst_hps['MLP-adam'][7]: 0.999,
            # lst_hps['MLP-adam'][8]: 1e-8,
        },
        'MLP-adam-v2': {
            lst_hps['MLP-adam-v2'][0]: 32,
            lst_hps['MLP-adam-v2'][1]: 1e-4,
            lst_hps['MLP-adam-v2'][2]: 1e-3,
            lst_hps['MLP-adam-v2'][3]: 1,
        },
    }
    assert method in lst_hps
    assert method in api_config
    assert method in default_config
    return lst_hps[method], api_config[method], default_config[method]


def get_est_from_hp(hp, method='HGB'):
    if method == 'HGB':
        return HGB(
            **hp,
            random_state=5489,
        )
    elif method == 'SVM':
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            **hp,
        )
    elif method == 'MLP-adam':
        return MLPC(
            **hp, random_state=5489,
            ## PARAMETERS NOT OPTIMIZED BY ASKL
            solver='adam', activation='relu', early_stopping=True, shuffle=True,
            batch_size='auto', tol=1e-4, validation_fraction=0.1,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        )
    elif method == 'MLP-adam-v2':
        hp_mod = {k: v for k, v in hp.items() if k != 'nlayers'}
        if hp['nlayers'] > 1:
            hsize = hp_mod['hidden_layer_sizes']
            hp_mod['hidden_layer_sizes'] = (hsize, ) * hp['nlayers']
        return MLPC(
            **hp_mod, random_state=5489,
            ## PARAMETERS NOT OPTIMIZED BY ASKL
            solver='adam', activation='relu', early_stopping=True, shuffle=True,
            batch_size='auto', tol=1e-4, validation_fraction=0.1,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        )
    else:
        raise Exception(f"Unknown method '{method}' encountered")



#Default performance
def evaluate_hp(
        hp, X, y, score_metric, n_splits=5,
        random_state=5489, method='HGB', vfrac=0.2
):
    ret = {}
    # est = HGB(**hp)
    est = get_est_from_hp(hp, method=method)
    cv = ShuffleSplit(
        n_splits=1, test_size=vfrac, random_state=random_state
    ) if n_splits == 1 else KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    cv_results = cross_validate(est, X, y, scoring=score_metric, cv=cv)
    cv_scores = cv_results['test_score']
    print(
        f'- Plain CV scores     : {np.mean(cv_scores):.4f} +- {np.std(cv_scores):.4f}'
        f' {cv_scores}'
    )
    ret['kfold'] = cv_scores.copy()
    cv = StratifiedShuffleSplit(
        n_splits=1, test_size=vfrac, random_state=random_state,
    ) if n_splits == 1 else StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )
    cv_results = cross_validate(est, X, y, scoring=score_metric, cv=cv)
    cv_scores = cv_results['test_score']
    print(
        f'- Stratified CV scores: {np.mean(cv_scores):.4f} +- {np.std(cv_scores):.4f}'
        f' {cv_scores}'
    )
    ret['skfold'] = cv_scores.copy()
    return ret


# create global/local loss surfaces
def create_global_local_loss_surfaces(
        run_csvs, lst_hps, no_header=True, num_pairs=-1, send_top=False
):
    dlist = []
    lsurfs = [RF(n_estimators=100, criterion='mse') for r in run_csvs]
    samples_nums = []
    for f, rf in zip(run_csvs, lsurfs):
        assert f.find('_full.csv') == -1 or f.find('_full_pysot.csv') == -1
        print(f'Processing {f} ...')
        df = pd.read_csv(f, header=None) if no_header else pd.read_csv(f)
        sample_num = int(df.iloc[0, -1])
        samples_nums.append(sample_num)
        print(f"sample num: {sample_num}")
        df = df.iloc[:, :-1]
        assert df.shape[1] == len(lst_hps) + 1
        print(f'- Size: {df.shape}')
        df = df.values
        print(f'- Best obj: {np.min(df[:, -1]):.4f}')
        X = df[:, :len(lst_hps)]
        y = 1. + df[:, -1]
        if num_pairs != -1 and num_pairs < X.shape[0]:
            # send only a subset of (HP, loss) pairs
            if send_top:
                top_idxs = np.argsort(y)[:num_pairs]
                X = X[top_idxs, :]
                y = y[top_idxs]
            else:
                X = X[:num_pairs, :]
                y = y[:num_pairs]
            print(f'- Best obj in selected list: {np.min(y) - 1.:.4f}')
        print(f'- fitting local surface with {X.shape} ...')
        rf.fit(X, y)
        dlist += [df]

    gdf = np.vstack(dlist)
    X = gdf[:, :len(lst_hps)]
    y = 1. + gdf[:, -1]
    print(f'Fitting global surface with {X.shape} ...')
    gsurf = RF(n_estimators=100, criterion='mse').fit(X, y)
    return lsurfs, gsurf, samples_nums


def MPLM(x, lsurfs):
    preds = [rf.predict([x])[0] for rf in lsurfs]
    return np.max(preds)


def APLM(x, lsurfs):
    preds = [rf.predict([x])[0] for rf in lsurfs]
    return np.mean(preds)


def WAPLM(x, lsurfs, samples_nums):
    """
    Weighted Average Loss Surface, weighted according to data sample amount
    """
    preds = [rf.predict([x])[0] for rf in lsurfs]
    # print(f'pred shape: {len(preds)}, sample_nums_len: {len(samples_nums)}')
    return np.average(preds, weights=samples_nums)


def SGM_U(x, gsurf):
    preds = [est.predict([x])[0] for est in gsurf.estimators_]
    mean = gsurf.predict([x])[0]
    v_mean = np.mean(preds)
    v_std = np.std(preds)
    assert np.isclose(mean, v_mean), (f'Ex mean: {v_mean}, Im mean: {mean}')
    return v_mean + v_std


def SGM(x, gsurf):
    return gsurf.predict([x])[0]


def test_objectives(lsurfs, gsurf, api_config, nsamples=3, default_hp=None):
    samples = []
    for k, v in api_config.items():
        vtype = v['type']
        assert vtype in ['real', 'int']
        vmin, vmax = v['range']
        if vtype == 'real':
            samples += [np.random.uniform(vmin, vmax, size=nsamples)]
        else:
            assert vtype == 'int'
            samples += [np.random.randint(vmin, vmax, size=nsamples).astype(np.int)]

    rnd_hps = np.transpose(np.array(samples))
    print(f'Random samples: {rnd_hps.shape}')

    for x in rnd_hps:
        print(f'Processing HP: {x} ...')
        print(f'- Max-of-locals : {MPLM(x, lsurfs):.4f}')
        print(f'- Mean-of-locals: {APLM(x, lsurfs):.4f}')
        print(f'- UCB-of-global : {SGM_U(x, gsurf):.4f}')

    x = [100, 0.1, 20, 0] if default_hp is None else default_hp
    ret = np.array([
        SGM(x, gsurf),
        SGM_U(x, gsurf),
        MPLM(x, lsurfs),
        APLM(x, lsurfs),
    ])
    print(f'Processing Default HP: {x} ...')
    print(f'- Mean-of-global: {1. - SGM(x, gsurf):.4f}')
    print(f'- UCB-of-global : {1. - SGM_U(x, gsurf):.4f}')
    print(f'- Max-of-locals : {1. - MPLM(x, lsurfs):.4f}')
    print(f'- Mean-of-locals: {1. - APLM(x, lsurfs):.4f}')
    return 1. - ret


def get_hp_suggestions(
        lst_hps,
        api_config,
        lsurfs,
        gsurf,
        samples_nums,
        batch_size,
        n_iters,
        n_restarts,
        check_every,
        print_every=30,
        tol=1e-3
):
    best_hps_scores = []
    for obj in [
            SGM,
            SGM_U,
            MPLM,
            WAPLM,
            APLM,
    ]:
        print(f"obj: {obj}, name: {obj.__name__}")
        surf = lsurfs if obj.__name__.find('locals') != -1 else gsurf
        if obj is WAPLM:
            surf = lsurfs
        opt = PySOTOptimizer(api_config)
        i = 0
        current_best = 1.
        last_check_best = 1.
        all_X, all_y = [], []
        restarts = 0
        while True:
            try:
                X = opt.suggest(n_suggestions=batch_size)
                if obj is WAPLM:
                    y = [obj(np.array([xx[k] for k in lst_hps]), surf, samples_nums) for xx in X]
                else:
                    y = [obj(np.array([xx[k] for k in lst_hps]), surf) for xx in X]
                all_X += X
                all_y += y
                opt.observe(X, y)
            except (BaseException) as e:
                print(f'Caught following exception:\n{e}')
                raise e
                i = 0
                opt = PySOTOptimizer(api_config)
                restarts += 1
                continue
            if np.min(all_y) < current_best:
                current_best = np.min(all_y)
            if i >= n_iters:
                if (i - n_iters) % check_every == 0:
                    improv = (last_check_best - current_best)/current_best
                    print(
                        f'[R{restarts+1}] Current iter: {i + 1}/{n_iters}, '
                        f'improvement: {improv:.4f}/{tol:.4f}'
                    )
                    if improv < tol:
                        restarts += 1
                        if restarts < n_restarts:
                            i = 0
                        else:
                            break
                    last_check_best = current_best
            i += 1
            if i % print_every == 0:
                print(f'[R{restarts+1}] Iter: {i} --> current best: {current_best:.4f}')
            # break
        print(f'Optimization with {obj.__name__} with best obj: {current_best:.4f}')
        best_hp = all_X[all_y.index(current_best)]
        print(f'Best HP: {best_hp}')
        best_hps_scores += [(obj.__name__, best_hp, current_best)]
    return best_hps_scores


# Evaluate the suggested HPs
def eval_suggested_hps(
        best_hps_scores, X, y, score_metric,
        n_splits=5, method='HGB', vfrac=0.2
):
    metrics_ret = {}
    for o, hp, cb in best_hps_scores:
        print(f'Objective             : {o} ....')
        print(f'- HP                  : {hp}')
        print(f'- Predicted objective : {(1. - cb):.4f}')
        metrics_ret[o] = evaluate_hp(
            hp, X, y, score_metric, n_splits=n_splits, method=method, vfrac=vfrac
        )
    return metrics_ret


if __name__ == '__main__':
    # Experiment config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path', help='Path to the directory with the csv from HPO',
        type=str, required=True,
    )
    parser.add_argument(
        '-d', '--data_path', help='Path to corresponding data',
        type=str, required=True,
    )
    parser.add_argument(
        '-f', '--full_hpo_file',
        help='Path to the file containing full HPO data (if available)',
        type=str, default='',
    )
    parser.add_argument(
        '-l', '--clabel', help='Label column',
        type=str, required=True,
    )
    scorers = ['balanced_accuracy', 'f1', 'roc_auc']
    parser.add_argument(
        '-s', '--score_metric', help='Scoring metric',
        required=True, choices=scorers,
    )
    ## short = False
    ## n_restarts = 5 if short else 10
    ## n_iters = 100 if short else 200
    ## check_every = 10 if short else 20
    parser.add_argument(
        '-r', '--n_restarts', help='Number of restarts for final loss surface optimization',
        type=int, default=5,
    )
    parser.add_argument(
        '-i', '--n_iters', help='Number of iterations for final loss surface optimization',
        type=int, default=100,
    )
    parser.add_argument(
        '-c', '--check_every', help='Check for convergence every this iteration',
        type=int, default=10,
    )
    ## batch_size = 4
    ## print_every = 30
    ## tol = 1e-3
    parser.add_argument(
        '-b', '--batch_size', help='Batch size for final loss surface optimization',
        type=int, default=4,
    )
    parser.add_argument(
        '-P', '--print_every', help='Print every this iterations',
        type=int, default=30,
    )
    parser.add_argument(
        '-t', '--tol', help='Tolerance for final optimization', type=float, default=1e-3,
    )
    parser.add_argument(
        '-o', '--output_path',
        help='Path to the directory where we save the final analysis results',
        type=str, default='',
    )
    parser.add_argument(
        '-H', '--header', help='Do HPO csv files have headers?', action='store_true'
    )
    parser.add_argument(
        '-F', '--nfolds',
        help='Number of folds for score computation',
        type=int, default=5,
    )
    parser.add_argument(
        '-V', '--vfrac',
        help='Validation fraction for train/validation split',
        type=float, default=0.2,
    )
    methods = ['HGB', 'SVM', 'MLP-adam', 'MLP-adam-v2']
    parser.add_argument(
        '-M', '--method', help='ML method to evaluate',
        choices=methods, default=methods[0],
    )
    parser.add_argument(
        '-X', '--prescale_x', help='Whether to pre-scale X.',
        action='store_true'
    )
    parser.add_argument(
        '-T', '--num_pairs_per_party',
        help='Number of (HP, loss) pairs to send to aggregator from each party',
        type=int, default=-1,
    )
    parser.add_argument(
        '-R', '--send_top', help='Whether to send the best-T (HP, loss) pairs',
        action='store_true'
    )


    args = parser.parse_args()
    assert os.path.isdir(args.path)
    assert os.path.isfile(args.data_path)
    assert args.n_restarts > 0
    assert args.n_iters >  10
    assert args.check_every > 1
    assert args.batch_size > 0
    assert args.print_every > 1
    assert args.tol > 1e-4
    assert args.output_path == '' or os.path.isdir(args.output_path)
    assert args.nfolds > 2 or args.nfolds == 1
    assert args.vfrac > 0.0 and args.vfrac < 0.5

    # local HPO run results
    run_csvs = glob(f'{args.path}/*.csv')
    run_csvs = [r for r in run_csvs if (r.find('_full.csv') == -1) and (r.find('_full_pysot.csv') == -1)]
    print(f'Found following HPO runs:\n{run_csvs}')

    # search space
    hp_names, search_space, default_hp = get_search_space(method=args.method)
    X, y = get_full_data(args.data_path, args.clabel, prescale=args.prescale_x)

    # Generate local/global loss surfaces
    lsurfs, gsurf, samples_nums = create_global_local_loss_surfaces(
        run_csvs, hp_names, no_header=(not args.header),
        num_pairs=args.num_pairs_per_party, send_top=args.send_top,
    )

    # Evaluate default HP
    print(f'Default HP: {default_hp}')
    dhp_array = [default_hp[n] for n in hp_names]
    default_pred_perf = test_objectives(
            lsurfs, gsurf, search_space, default_hp=dhp_array)
    print('Evaluating default HP on full data: ...')
    default_perf = evaluate_hp(
        default_hp, X, y, args.score_metric,
        n_splits=args.nfolds, method=args.method, vfrac=args.vfrac,
    )

    # Generate HP suggestions using the loss surfaces
    best_hps_scores = get_hp_suggestions(
        hp_names,
        search_space,
        lsurfs,
        gsurf,
        samples_nums,
        args.batch_size,
        args.n_iters,
        args.n_restarts,
        args.check_every,
        args.print_every,
        args.tol,
    )

    # Evaluate HP suggestions
    suggests_perf = eval_suggested_hps(
        best_hps_scores, X, y, args.score_metric,
        n_splits=args.nfolds, method=args.method, vfrac=args.vfrac,
    )

    # Final stats:
    print('='*80)
    final_stats = {}
    best_possible = None
    if args.full_hpo_file != '':
        best_possible = -np.min(pd.read_csv(args.full_hpo_file)['label'].values)
        print(f'Best possible: {best_possible:.4f}')
        final_stats['best_possible'] = best_possible
    print('Performance summary for the default HP:')
    print(f'- Predicted objectives: {default_pred_perf}')
    for k, v in default_perf.items():
        print(f'- {k:6}: {np.mean(v):.4f} -+ {np.std(v):.4f} {v}')
    default_metric = np.mean(default_perf['skfold'])
    final_stats['default'] = {
        'HP': default_hp,
        'predicted': default_pred_perf,
        **default_perf,
        'metric': default_metric,
    }
    for o, hp, p in best_hps_scores:
        print(f'Performance summary for HP selected with {o}:')
        print(f'- Predicted objective: {1 - p}')
        for k, v in suggests_perf[o].items():
            print(f'- {k:6}: {np.mean(v):.4f} -+ {np.std(v):.4f} {v}')
        final_stats[o] = {
            'HP': hp,
            'predicted': 1. - p,
            **suggests_perf[o],
        }
        sugg_metric = np.mean(suggests_perf[o]['skfold'])
        if best_possible is not None:
            regret = (best_possible - sugg_metric) / (best_possible - default_metric)
            print(f'- Regret: {regret:.4f}')
            final_stats[o]['regret'] = regret
    print('='*80)
    pprint(final_stats)
    print('='*80)

    if args.output_path != '':
        pkf_name = 'final_stats.pkl'
        if args.num_pairs_per_party != -1:
            if args.send_top:
                pkf_name = f'final_stats_R_T{args.num_pairs_per_party}.pkl'
            else:
                pkf_name = f'final_stats_T{args.num_pairs_per_party}.pkl'
        fpath = os.path.join(args.output_path, pkf_name)
        print(f'Saving final stats in {fpath}')
        with open(fpath, 'wb') as pkf:
            pickle.dump(final_stats, pkf)
