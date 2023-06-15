import argparse
import csv
import os
import sys
from glob import glob

import numpy as np

np.set_printoptions(precision=4)
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting

from dscan.dataset_label_cols import data_files, get_full_data, label_cols
from local_hpo import execute_hpo
from loss_surface_analysis import evaluate_hp, get_search_space


def mainFun():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data_path",
        help="Path to data files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path where to output HPO runs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-F",
        "--nfolds",
        help="Number of folds in the CV",
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
        "-I",
        "--niters",
        help="Number of HPO iterations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-r",
        "--nrestarts",
        help="Number of restarts for the HPO runs",
        type=int,
        default=1,
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
        "-D",
        "--subset_of_datasets",
        help="Use only subset of data sets",
        action="store_true",
    )
    parser.add_argument(
        "-X", "--prescale_x", help="Whether to pre-scale X.", action="store_true"
    )

    args = parser.parse_args()
    assert os.path.isdir(args.data_path), f"Data path: {args.data_path} ..."
    assert os.path.isdir(args.output_path)
    assert args.nfolds > 2 or args.nfolds == 1
    assert args.vfrac > 0.0 and args.vfrac < 0.5
    assert args.niters >= 10
    assert args.nrestarts >= 1
    assert args.nsuggests >= 1
    dset_subset = [
        "dataset_40_sonar",
        "dataset_53_heart-statlog",
        "oil_spill",
        "pc3",
        "pollen",
        "eeg_eye_state",
    ]
    if args.method != "SVM":
        dset_subset += ["electricity-normalized"]

    hpnames, search_space, default_config = get_search_space(method=args.method)
    print(f"Considering the following search space:\n{search_space}")
    metrics = [
        "balanced_accuracy",
    ]
    dvh_col_names = (
        [
            "Dataset",
            "Label",
            "metric",
            "nfolds",
            "vfrac",
            "config",
            "k-fold-avg",
        ]
        if args.nfolds == 1
        else [
            "Dataset",
            "Label",
            "metric",
            "nfolds",
            "config",
            "k-fold-avg",
        ]
    )
    print(dvh_col_names)
    fpath = f"perf_all_data.csv"
    dvh_file_name = os.path.join(args.output_path, fpath)
    print(f"Saving default performance for all data in {dvh_file_name} ...")
    if os.path.isfile(dvh_file_name):
        print(f" ... summary result already available so continuing to add to this ...")
    else:
        print(f" ... adding header to summary result file ...")
        with open(dvh_file_name, "a", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=dvh_col_names)
            writer.writeheader()

    for d, l in zip(data_files, label_cols):
        if args.subset_of_datasets:
            if d not in dset_subset:
                continue
        print(f"{d} --> {l}")
        dpath = os.path.join(args.data_path, f"{d}.csv")
        assert os.path.isfile(dpath)
        X, y = get_full_data(dpath, l, prescale=args.prescale_x)
        for m in metrics:
            front = (
                [d, l, m, args.nfolds, args.vfrac]
                if args.nfolds == 1
                else [d, l, m, args.nfolds]
            )
            hpo_fprefix = (
                f"{d}_{l}_{m}_k{args.nfolds}"
                + (f"_V{args.vfrac:.1f}" if args.nfolds == 1 else "")
                + f"_I{args.niters}_R{args.nrestarts}_S{args.nsuggests}"
            )
            efile = glob(os.path.join(args.output_path, hpo_fprefix + "*.csv"))
            if len(efile) > 0:
                print(
                    f"Skipping HPO vs default comparison for {hpo_fprefix}; "
                    f"following files found:\n - {efile[0]}"
                )
                continue
            print(f"Initiating Default vs. HPO for {hpo_fprefix} ...")
            # evaluating default performance
            def_perf = evaluate_hp(
                default_config,
                X,
                y,
                m,
                n_splits=args.nfolds,
                method=args.method,
                vfrac=args.vfrac,
            )
            for k, v in def_perf.items():
                if k == "kfold":
                    continue
                assert k == "skfold"
                row = front + ["default", np.mean(v)]
                assert len(row) == len(dvh_col_names)
                # print(row)
                with open(dvh_file_name, "a", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=dvh_col_names)
                    writer.writerow({k: v for k, v in zip(dvh_col_names, row)})

            # executing HPO
            cpath, best_score = execute_hpo(
                X,
                y,
                search_space,
                m,
                hpnames + ["label"],
                hpo_fprefix,
                "full",
                args.output_path,
                args.niters,
                n_restarts=args.nrestarts,
                nsuggests=args.nsuggests,
                nfolds_per_party=args.nfolds,
                random_state=5489,
                method=args.method,
                vfrac=args.vfrac,
            )
            row = front + ["hpo", best_score]
            assert len(row) == len(dvh_col_names)
            with open(dvh_file_name, "a", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=dvh_col_names)
                writer.writerow({k: v for k, v in zip(dvh_col_names, row)})


if __name__ == "__main__":
    mainFun()
