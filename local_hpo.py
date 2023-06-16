import argparse
import csv
import os
import sys
from glob import glob

import numpy as np

np.set_printoptions(precision=4)
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     cross_validate)

from dscan.dataset_label_cols import get_full_data
from loss_surface_analysis import (evaluate_hp, get_est_from_hp,
                                   get_search_space)
from optimizers import HyperoptOptimizer as HPO


def execute_hpo(
    X,
    y,
    api_config,
    score_metric,
    col_names,
    party_prefix,
    pname,
    save_path,
    hpo_niters,
    n_restarts,
    nsuggests,
    nfolds_per_party=5,
    random_state=5489,
    method="HGB",
    vfrac=0.2,
):
    print(f"Starting HPO for {pname} with data {X.shape}, {y.shape} ...")
    skf_in = (
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=vfrac,
            random_state=5489,
        )
        if nfolds_per_party == 1
        else StratifiedKFold(
            n_splits=nfolds_per_party,
            shuffle=True,
            random_state=5489,
        )
    )
    party_csv_path = os.path.join(save_path, f"{party_prefix}_{pname}.csv")
    print(f" ... saving results in {party_csv_path}")
    with open(party_csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col_names)
        writer.writeheader()
    best_score = 1.0
    for i in range(n_restarts):
        opt = HPO(api_config)
        for j in range(hpo_niters):
            # run one suggest/observe loop
            hps = opt.suggest(n_suggestions=nsuggests)
            objs = []
            for hp in hps:
                # est = HGB(**hp)
                est = get_est_from_hp(hp, method=method)
                cv_res = cross_validate(est, X, y, scoring=score_metric, cv=skf_in)
                objs += [-np.mean(cv_res["test_score"])]
            assert len(hps) == len(objs)
            opt.observe(hps, objs)
            # save results
            for hp, obj in zip(hps, objs):
                if obj < best_score:
                    best_score = obj
                hp_score_pair = hp
                hp_score_pair["label"] = obj
                hp_score_pair["samples_num"] = y.shape[0]
                with open(party_csv_path, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=col_names)
                    writer.writerow(hp_score_pair)
            if (j + 1) % 20 == 0:
                print(
                    f"[{pname}]: [R{i+1}] HPO iters {j+1}/{hpo_niters} completed, "
                    f"best score: {best_score:.4f}"
                )
    best_score = -np.min(pd.read_csv(party_csv_path).values[:, -2])
    print(f"Completed HPO for {pname} with best {score_metric}: {best_score:.4f}")
    return party_csv_path, best_score


def process_full_hpo_file(full_hpo_file, save_file, save_path):
    rdf = pd.read_csv(full_hpo_file)
    csv_path = os.path.join(save_path, save_file)
    print(f"Copying {full_hpo_file} to {csv_path}")
    rdf.to_csv(csv_path, header=True, index=False)
    best_score = -np.min(rdf.values[:, -1])
    print(f"Copied HPO run from {full_hpo_file} with best score: {best_score:.4f}")
    return csv_path, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        help="Path to corresponding data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-l",
        "--clabel",
        help="Label column",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the directory where the csv from HPO is to be saved",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--full_hpo_file",
        help="Path to the file containing full HPO data (if available)",
        type=str,
        default="",
    )
    parser.add_argument(
        "-P",
        "--nparties",
        help="Number of parties",
        type=int,
        required=True,
    )
    scorers = ["balanced_accuracy", "f1", "roc_auc"]
    parser.add_argument(
        "-s",
        "--score_metric",
        help="Scoring metric",
        required=True,
        choices=scorers,
    )
    # Experiment config
    parser.add_argument(
        "-F",
        "--nfolds_per_party",
        help="Number of folds for score computation in each party",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-V",
        "--vfrac",
        help="Validation fraction for train/validation split",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-i",
        "--hpo_niters",
        help="Number of iterations in the HPO",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-r",
        "--n_restarts",
        help="Number of restarts for the HPO runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-S",
        "--nsuggests",
        help="Number of suggests per HPO iteration",
        type=int,
        default=1,
    )
    methods = ["HGB", "SVM", "MLP-adam", "MLP-adam-v2"]
    parser.add_argument(
        "-M",
        "--method",
        help="ML method to evaluate",
        choices=methods,
        default=methods[0],
    )
    parser.add_argument(
        "-X", "--prescale_x", help="Whether to pre-scale X.", action="store_true"
    )

    args = parser.parse_args()
    assert os.path.isdir(args.path)
    assert os.path.isfile(args.data_path)
    assert args.nparties > 1
    party_prefix = f"hp_{args.score_metric}_pairs_data_"
    assert args.nfolds_per_party >= 3 or args.nfolds_per_party == 1
    assert args.vfrac > 0.0 and args.vfrac < 0.5
    assert args.hpo_niters > 10
    assert args.n_restarts > 0
    assert args.nsuggests > 0
    assert args.full_hpo_file == "" or os.path.isfile(args.full_hpo_file)

    found_files = glob(f"{args.path}/*.csv")
    if len(found_files) > 0:
        print(
            f"Found existing {len(found_files)} CSV files in {args.path}:\n{found_files}"
        )
        print(f"Please provide a different path or save the above files")
        print(f"Exiting now ...")
        sys.exit(0)

    # search space
    hp_names, search_space, default_hp = get_search_space(method=args.method)
    X, y = get_full_data(args.data_path, args.clabel, prescale=args.prescale_x)

    col_names = hp_names + ["label"] + ["samples_num"]

    # Evaluate default HP
    print(f"Default HP: {default_hp}")
    print("Evaluating default HP on full data: ...")
    default_perf = evaluate_hp(
        default_hp,
        X,
        y,
        args.score_metric,
        n_splits=args.nfolds_per_party,
        method=args.method,
        vfrac=args.vfrac,
    )

    # Run HPO on full data
    full_hpo_csv, best_full_score = (
        execute_hpo(
            X,
            y,
            search_space,
            args.score_metric,
            col_names,
            party_prefix,
            "full",
            args.path,
            args.hpo_niters,
            args.n_restarts,
            args.nsuggests,
            nfolds_per_party=args.nfolds_per_party,
            method=args.method,
            vfrac=args.vfrac,
        )
        if args.full_hpo_file == ""
        else process_full_hpo_file(
            args.full_hpo_file,
            f"{party_prefix}_full.csv",
            args.path,
        )
    )
    party_stats = [("full", best_full_score, full_hpo_csv)]

    # Run per-party HPOs
    skf = StratifiedKFold(n_splits=args.nparties, shuffle=True, random_state=5489)
    party_idx = 0
    for _, tindex in skf.split(X, y):
        pX, py = X[tindex], y[tindex]
        pname = f"party_{party_idx}"
        hpo_csv, best_score = execute_hpo(
            pX,
            py,
            search_space,
            args.score_metric,
            col_names,
            party_prefix,
            pname,
            args.path,
            args.hpo_niters,
            args.n_restarts,
            args.nsuggests,
            nfolds_per_party=args.nfolds_per_party,
            method=args.method,
            vfrac=args.vfrac,
        )
        party_stats += [(pname, best_score, hpo_csv)]
        party_idx += 1

    print("=" * 80)
    final_stats = {}
    print("Performance summary for the default HP:")
    for k, v in default_perf.items():
        print(f"- {k:6}: {np.mean(v):.4f} -+ {np.std(v):.4f} {v}")
    print("Performance summary for HPO:")
    for p, s, f in party_stats:
        print(f"- {p:10} best score: {s:.4f} ({f})")
    print("=" * 80)
