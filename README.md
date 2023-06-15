# Installation

We need the following version of Python and the `requirements.txt` contain all the required libraries for our experiments.

- Python 3.6.8
- `pip install -r requirements.txt`

# General

## Computing the performance upper and lower bound

```
python hpo_v_default.py --help
usage: hpo_v_default.py [-h] -d DATA_PATH -o OUTPUT_PATH [-F NFOLDS] [-V VFRAC] [-I NITERS] [-r NRESTARTS]
                        [-S NSUGGESTS] [-M {HGB,SVM,MLP-adam,MLP-adam-v2}] [-D] [-X]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Path to data files
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path where to output HPO runs
  -F NFOLDS, --nfolds NFOLDS
                        Number of folds in the CV
  -V VFRAC, --vfrac VFRAC
                        Validation fraction for train/validation split
  -I NITERS, --niters NITERS
                        Number of HPO iterations
  -r NRESTARTS, --nrestarts NRESTARTS
                        Number of restarts for the HPO runs
  -S NSUGGESTS, --nsuggests NSUGGESTS
                        Number of suggests per HPO iteration
  -M {HGB,SVM,MLP-adam,MLP-adam-v2}, --method {HGB,SVM,MLP-adam,MLP-adam-v2}
                        ML method to evaluate
  -D, --subset_of_datasets
                        Use only subset of data sets
  -X, --prescale_x      Whether to pre-scale X.

```

### For histogram-based gradient boosted decision trees (HGB)

```
> mkdir -p HGB/hpo_v_default
> python hpo_v_default.py -d datasets/ -o HGB/hpo_v_default/ -F 10 -I 30 -r 5 -S 4 -M HGB -D
```

This results in the following files:

```
> tree HGB/hpo_v_default
HGB/hpo_v_default/
 |- dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv
```

### For support vector machines with radial basis function kernels (SVM)

```
> mkdir -p SVM-prescale/hpo_v_default
> python hpo_v_default.py -d datasets/ -o SVM-prescale/hpo_v_default/ -F 10 -I 30 -r 5 -S 4 -M SVM -D -X
```

This results in the following files:

```
> tree SVM-prescale/hpo_v_default
SVM-prescale/hpo_v_default/
 |- dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv
```

### For multi-layered perceptrons with Adam optimizer (MLP-adam)

```
> mkdir -p MLP-adam-prescale/hpo_v_default
> python hpo_v_default.py -d datasets/ -o MLP-adam-prescale/hpo_v_default/ -F 10 -I 30 -r 5 -S 4 -M MLP-adam -D -X
```

This results in the following files:

```
> tree MLP-adam-prescale/hpo_v_default
MLP-adam-prescale/hpo_v_default/
 |- dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv
 |- pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv
```

# Generating results for Table 1: Comparison to Single-shot Baseline

Table 1 in the paper is an aggregation of the results in Table 4, 5 and 6 in Appendix B.3. The following subsections detail how the results in Tables 4, 5 and 6 are generated.

## Performing per-party local HPOs

```
> python local_hpo.py --help
usage: local_hpo.py [-h] -d DATA_PATH -l CLABEL -p PATH [-f FULL_HPO_FILE] -P NPARTIES -s
                    {balanced_accuracy,f1,roc_auc} [-F NFOLDS_PER_PARTY] [-V VFRAC] [-i HPO_NITERS]
                    [-r N_RESTARTS] [-S NSUGGESTS] [-M {HGB,SVM,MLP-adam,MLP-adam-v2}] [-X]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Path to corresponding data
  -l CLABEL, --clabel CLABEL
                        Label column
  -p PATH, --path PATH  Path to the directory where the csv from HPO is to be saved
  -f FULL_HPO_FILE, --full_hpo_file FULL_HPO_FILE
                        Path to the file containing full HPO data (if available)
  -P NPARTIES, --nparties NPARTIES
                        Number of parties
  -s {balanced_accuracy,f1,roc_auc}, --score_metric {balanced_accuracy,f1,roc_auc}
                        Scoring metric
  -F NFOLDS_PER_PARTY, --nfolds_per_party NFOLDS_PER_PARTY
                        Number of folds for score computation in each party
  -V VFRAC, --vfrac VFRAC
                        Validation fraction for train/validation split
  -i HPO_NITERS, --hpo_niters HPO_NITERS
                        Number of iterations in the HPO
  -r N_RESTARTS, --n_restarts N_RESTARTS
                        Number of restarts for the HPO runs
  -S NSUGGESTS, --nsuggests NSUGGESTS
                        Number of suggests per HPO iteration
  -M {HGB,SVM,MLP-adam,MLP-adam-v2}, --method {HGB,SVM,MLP-adam,MLP-adam-v2}
                        ML method to evaluate
  -X, --prescale_x      Whether to pre-scale X.
```

## Executing FLoRA  and computing relative regret

```
> python loss_surface_analysis.py --help
usage: loss_surface_analysis.py [-h] -p PATH -d DATA_PATH [-f FULL_HPO_FILE] -l CLABEL -s
                                {balanced_accuracy,f1,roc_auc} [-r N_RESTARTS] [-i N_ITERS] [-c CHECK_EVERY]
                                [-b BATCH_SIZE] [-P PRINT_EVERY] [-t TOL] [-o OUTPUT_PATH] [-H] [-F NFOLDS]
                                [-V VFRAC] [-M {HGB,SVM,MLP-adam,MLP-adam-v2}] [-X] [-T NUM_PAIRS_PER_PARTY]
                                [-R]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the directory with the csv from HPO
  -d DATA_PATH, --data_path DATA_PATH
                        Path to corresponding data
  -f FULL_HPO_FILE, --full_hpo_file FULL_HPO_FILE
                        Path to the file containing full HPO data (if available)
  -l CLABEL, --clabel CLABEL
                        Label column
  -s {balanced_accuracy,f1,roc_auc}, --score_metric {balanced_accuracy,f1,roc_auc}
                        Scoring metric
  -r N_RESTARTS, --n_restarts N_RESTARTS
                        Number of restarts for final loss surface optimization
  -i N_ITERS, --n_iters N_ITERS
                        Number of iterations for final loss surface optimization
  -c CHECK_EVERY, --check_every CHECK_EVERY
                        Check for convergence every this iteration
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for final loss surface optimization
  -P PRINT_EVERY, --print_every PRINT_EVERY
                        Print every this iterations
  -t TOL, --tol TOL     Tolerance for final optimization
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the directory where we save the final analysis results
  -H, --header          Do HPO csv files have headers?
  -F NFOLDS, --nfolds NFOLDS
                        Number of folds for score computation
  -V VFRAC, --vfrac VFRAC
                        Validation fraction for train/validation split
  -M {HGB,SVM,MLP-adam,MLP-adam-v2}, --method {HGB,SVM,MLP-adam,MLP-adam-v2}
                        ML method to evaluate
  -X, --prescale_x      Whether to pre-scale X.
  -T NUM_PAIRS_PER_PARTY, --num_pairs_per_party NUM_PAIRS_PER_PARTY
                        Number of (HP, loss) pairs to send to aggregator from each party
  -R, --send_top        Whether to send the best-T (HP, loss) pairs
```

In this set of experiments, we consider FL-HPO with 3 parties (`-P 3`) with the balanced accuracy metric (`-s balanced_accuracy`). The evaluation of any hyperparameter is performed via 10-fold cross-validation (`-F 10`). For each party's local HPO for each method, we consider 100 HPO iterations, resulting in 100 (HP, loss) pairs per party (`-i 100 -r 1 -S 1`).

### Table 4: Histogram-based gradient boosted decision trees (HGB)

We begin by detailing the commands for `HGB` for the `Sonar` dataset. After that, we specify the appropriate values for the options for the other datasets.

#### `Sonar` dataset

- Dataset `-d`: `datasets/dataset_40_sonar.csv`
- Label column `-l`: `Class`
- Results directory `-p/-o`: `HGB/sonar_hp_bacc_iid3p_F10`
- Full HPO file `-f`: `HGB/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`

```
> mkdir HGB/sonar_hp_bacc_iid3p_F10
> python local_hpo.py -f HGB/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/dataset_40_sonar.csv -l Class -p HGB/sonar_hp_bacc_iid3p_F10/ -P 3 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB

```

This create the per-party loss pairs

```
> tree HGB/sonar_hp_bacc_iid3p_F10/
 |- hp_balanced_accuracy_pairs_data__full.csv
 |- hp_balanced_accuracy_pairs_data__party_0.csv
 |- hp_balanced_accuracy_pairs_data__party_1.csv
 |- hp_balanced_accuracy_pairs_data__party_2.csv
```

Now we create the different loss surfaces and perform single-shot FLoRA:

```
> python loss_surface_analysis.py -p HGB/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f HGB/sonar_hp_bacc_iid3p_F10/hp_balanced_accuracy_pairs_data__full.csv -l Class -s balanced_accuracy -o HGB/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M HGB

```

This command prints the regret for the different loss surfaces in the logs and saves the results in `final_stats.pkl` in the output directory specified via `-o`.

#### Remaining datasets

We need to create the dataset specific results directories and run the above pair of commands (`local_hpo.py` and `loss_surface_analysis.py`) for the remaining datasets with the following corresponding options:

- Sonar: (used in above example)
  - Dataset `-d`: `datasets/dataset_40_sonar.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `HGB/sonar_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Electricity:
  - Dataset `-d`: `datasets/electricity-normalized.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `HGB/elec_norm_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- EEG eye state:
  - Dataset `-d`: `datasets/eeg_eye_state.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `HGB/eeg_eye_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Heart Statlog:
  - Dataset `-d`: `datasets/dataset_53_heart-statlog.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `HGB/heart_statlog_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Oil Spill:
  - Dataset `-d`: `datasets/oil_spill.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `HGB/oil_spill_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- PC3:
  - Dataset `-d`: `datasets/pc3.csv`
  - Label column `-l`: `c`
  - Results directory `-p/-o`: `HGB/pc3_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Pollen:
  - Dataset `-d`: `datasets/pollen.csv`
  - Label column `-l`: `binaryClass`
  - Results directory `-p/-o`: `HGB/pollen_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `HGB/hpo_v_default/pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv`

### Table 5: Support vector machines with radial basis function kernels (SVM)

We begin by detailing the commands for `SVM` for the `Sonar` dataset. After that, we specify the appropriate values for the options for the other datasets.

#### `Sonar` dataset

- Dataset `-d`: `datasets/dataset_40_sonar.csv`
- Label column `-l`: `Class`
- Results directory `-p/-o`: `SVM-prescale/sonar_hp_bacc_iid3p_F10`
- Full HPO file `-f`: `SVM-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`

```
> mkdir SVM-prescale/sonar_hp_bacc_iid3p_F10
> python local_hpo.py -f SVM-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/dataset_40_sonar.csv -l Class -p SVM-prescale/sonar_hp_bacc_iid3p_F10/ -P 3 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M SVM -X

```

This create the per-party loss pairs

```
> tree SVM-prescale/sonar_hp_bacc_iid3p_F10/
 |- hp_balanced_accuracy_pairs_data__full.csv
 |- hp_balanced_accuracy_pairs_data__party_0.csv
 |- hp_balanced_accuracy_pairs_data__party_1.csv
 |- hp_balanced_accuracy_pairs_data__party_2.csv
```

Now we create the different loss surfaces and perform single-shot FLoRA:

```
> python loss_surface_analysis.py -p SVM-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f SVM-prescale/sonar_hp_bacc_iid3p_F10/hp_balanced_accuracy_pairs_data__full.csv -l Class -s balanced_accuracy -o SVM-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M SVM -X 

```

This command prints the regret for the different loss surfaces in the logs and saves the results in `final_stats.pkl` in the output directory specified via `-o`.

#### Remaining datasets

We need to create the dataset specific results directories and run the above pair of commands (`local_hpo.py` and `loss_surface_analysis.py`) for the remaining datasets with the following corresponding options:

- Sonar: (used in above example)
  - Dataset `-d`: `datasets/dataset_40_sonar.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `SVM-prescale/sonar_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Electricity:
  - Dataset `-d`: `datasets/electricity-normalized.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `SVM-prescale/elec_norm_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- EEG eye state:
  - Dataset `-d`: `datasets/eeg_eye_state.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `SVM-prescale/eeg_eye_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Heart Statlog:
  - Dataset `-d`: `datasets/dataset_53_heart-statlog.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `SVM-prescale/heart_statlog_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Oil Spill:
  - Dataset `-d`: `datasets/oil_spill.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `SVM-prescale/oil_spill_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- PC3:
  - Dataset `-d`: `datasets/pc3.csv`
  - Label column `-l`: `c`
  - Results directory `-p/-o`: `SVM-prescale/pc3_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Pollen:
  - Dataset `-d`: `datasets/pollen.csv`
  - Label column `-l`: `binaryClass`
  - Results directory `-p/-o`: `SVM-prescale/pollen_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `SVM-prescale/hpo_v_default/pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv`

### Table 6: Multi-layered perceptrons with Adam optimizer (MLP-adam)

We begin by detailing the commands for `MLP-adam` for the `Sonar` dataset. After that, we specify the appropriate values for the options for the other datasets.

#### `Sonar` dataset

- Dataset `-d`: `datasets/dataset_40_sonar.csv`
- Label column `-l`: `Class`
- Results directory `-p/-o`: `MLP-adam-prescale/sonar_hp_bacc_iid3p_F10`
- Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`

```
> mkdir MLP-adam-prescale/sonar_hp_bacc_iid3p_F10
> python local_hpo.py -f MLP-adam-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/dataset_40_sonar.csv -l Class -p MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -P 3 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M MLP-adam -X

```

This create the per-party loss pairs

```
> tree MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/
 |- hp_balanced_accuracy_pairs_data__full.csv
 |- hp_balanced_accuracy_pairs_data__party_0.csv
 |- hp_balanced_accuracy_pairs_data__party_1.csv
 |- hp_balanced_accuracy_pairs_data__party_2.csv
```

Now we create the different loss surfaces and perform single-shot FLoRA:

```
> python loss_surface_analysis.py -p MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/hp_balanced_accuracy_pairs_data__full.csv -l Class -s balanced_accuracy -o MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M MLP-adam -X 

```

This command prints the regret for the different loss surfaces in the logs and saves the results in `final_stats.pkl` in the output directory specified via `-o`.

#### Remaining datasets

We need to create the dataset specific results directories and run the above pair of commands (`local_hpo.py` and `loss_surface_analysis.py`) for the remaining datasets with the following corresponding options:

- Sonar: (used in above example)
  - Dataset `-d`: `datasets/dataset_40_sonar.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `MLP-adam-prescale/sonar_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Electricity:
  - Dataset `-d`: `datasets/electricity-normalized.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `MLP-adam-prescale/elec_norm_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- EEG eye state:
  - Dataset `-d`: `datasets/eeg_eye_state.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`: `MLP-adam-prescale/eeg_eye_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Heart Statlog:
  - Dataset `-d`: `datasets/dataset_53_heart-statlog.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Oil Spill:
  - Dataset `-d`: `datasets/oil_spill.csv`
  - Label column `-l`: `class`
  - Results directory `-p/-o`: `MLP-adam-prescale/oil_spill_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/oil_spill_class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- PC3:
  - Dataset `-d`: `datasets/pc3.csv`
  - Label column `-l`: `c`
  - Results directory `-p/-o`: `MLP-adam-prescale/pc3_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/pc3_c_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Pollen:
  - Dataset `-d`: `datasets/pollen.csv`
  - Label column `-l`: `binaryClass`
  - Results directory `-p/-o`: `MLP-adam-prescale/pollen_hp_bacc_iid3p_F10`
  - Full HPO file `-f`: `MLP-adam-prescale/hpo_v_default/pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv`

# Generating results for Figure 1 & 2: Comparison to Multi-shot Baseline

These results does not require any new experiments but can be generated using the output of the centralized HPO and the results of FLoRA to compute the number of FL model trainings needed in multi-shot HPO to match the performance of single-shot FLoRA. The detailed results are presented in Appendix B.5. The `fig_1_2_B.5.ipynb` jupyter notebook processes the results to generate the figures posted in Figure 1 and 2, and the complete results in Figures 4, 5, 6 in Appendix B.5.

# Generating results for Table 2: Effect of increasing number of parties

In this set of experiments, we evaluate FLoRA with increasing number of parties `-P`. We just use the `HGB` method and the datasets EEG eye state, Electricity and Pollen. In the main paper, we present the results for a subset of the loss surfaces in Table 2. The results for all the loss surfaces are presented in Appendix B.6.

In the following, we present the results for the Electricity dataset.

- Number of parties `-P`: 3, 6, 10, 25, 50, 100
- Dataset `-d`: `datasets/electricity-normalized.csv`
- Label column `-l`: `class`
- Results directory `-p/-o`:
  - `HGB/elec_norm_hp_bacc_iid3p_F10` for `-P 3`
  - `HGB/elec_norm_hp_bacc_iid6p_F10` for `-P 6`
  - `HGB/elec_norm_hp_bacc_iid10p_F10` for `-P 10`
  - `HGB/elec_norm_hp_bacc_iid25p_F10` for `-P 25`
  - `HGB/elec_norm_hp_bacc_iid50p_F10` for `-P 50`
  - `HGB/elec_norm_hp_bacc_iid100p_F10` for `-P 100`
- Full HPO file `-f`: `HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv`

For each of the number of parties, we execute the following commands to compute the relative regret for all the loss surfaces, varying the `-P <NP>` option and the corresponding directory for results.

## 3 parties: `-P 3`

```
> mkdir HGB/elec_norm_hp_bacc_iid3p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid3p_F10/ -P 3 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid3p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid3p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid3p_F10/ -H -F 10 -M HGB 
```

## 6 parties: `-P 6`

```
> mkdir HGB/elec_norm_hp_bacc_iid6p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid6p_F10/ -P 6 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid6p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid6p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid6p_F10/ -H -F 10 -M HGB 
```

## 10 parties: `-P 10`

```
> mkdir HGB/elec_norm_hp_bacc_iid10p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid10p_F10/ -P 10 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid10p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid10p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid10p_F10/ -H -F 10 -M HGB 
```

## 25 parties: `-P 25`

```
> mkdir HGB/elec_norm_hp_bacc_iid25p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid25p_F10/ -P 25 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid25p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid25p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid25p_F10/ -H -F 10 -M HGB 
```

## 50 parties: `-P 50`

```
> mkdir HGB/elec_norm_hp_bacc_iid50p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid50p_F10/ -P 50 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid50p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid50p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid50p_F10/ -H -F 10 -M HGB 
```

## 100 parties: `-P 100`

```
> mkdir HGB/elec_norm_hp_bacc_iid100p_F10
> python local_hpo.py -f HGB/hpo_v_default/electricity-normalized_class_balanced_accuracy_k10_I30_R5_S4_full.csv -d datasets/electricity-normalized.csv -l class -p HGB/elec_norm_hp_bacc_iid100p_F10/ -P 100 -s balanced_accuracy -F 10 -i 100 -r 1 -S 1 -M HGB
> python loss_surface_analysis.py -p HGB/elec_norm_bacc_iid100p_F10/ -d datasets/electricity-normalized.csv -f HGB/elec_norm_hp_bacc_iid100p_F10/hp_balanced_accuracy_pairs_data__full.csv -l class -s balanced_accuracy -o HGB/elec_norm_hp_bacc_iid100p_F10/ -H -F 10 -M HGB 
```

## Remaining datasets

The above commands needs to be executed for the following two datasets with the corresponding number of parties, datasets and results directories listed below:

- EEG eye state:
  - Number of parties `-P`: 3, 6, 10, 25, 50
  - Dataset `-d`: `datasets/eeg_eye_state.csv`
  - Label column `-l`: `Class`
  - Results directory `-p/-o`:
    - `HGB/eeg_eye_hp_bacc_iid3p_F10` for `-P 3`
    - `HGB/eeg_eye_hp_bacc_iid6p_F10` for `-P 6`
    - `HGB/eeg_eye_hp_bacc_iid10p_F10` for `-P 10`
    - `HGB/eeg_eye_hp_bacc_iid25p_F10` for `-P 25`
    - `HGB/eeg_eye_hp_bacc_iid50p_F10` for `-P 50`
  - Full HPO file `-f`: `HGB/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv`
- Pollen:
  - Number of parties `-P`: 3, 6, 10
  - Dataset `-d`: `datasets/pollen.csv`
  - Label column `-l`: `binaryClass`
  - Results directory `-p/-o`:
    - `HGB/pollen_hp_bacc_iid3p_F10` for `-P 3`
    - `HGB/pollen_hp_bacc_iid6p_F10` for `-P 6`
    - `HGB/pollen_hp_bacc_iid10p_F10` for `-P 10`
  - Full HPO file `-f`: `HGB/hpo_v_default/pollen_binaryClass_balanced_accuracy_k10_I30_R5_S4_full.csv`

# Generating results for Figure 3: Effect of different choices in FLoRA

## Figure 3a: Effect of the number of per-party local HPO rounds

Figure 3a is a visualization of the results presented in Appendix B.7. We make use of the `-T` argument in `loss_surface_analysis.py` that varies the number of (HP, loss) pairs generated at each party:

```
  -T NUM_PAIRS_PER_PARTY, --num_pairs_per_party NUM_PAIRS_PER_PARTY
                        Number of (HP, loss) pairs to send to aggregator from each party
```

### MLP-adam with Heart Statlog

The following command evaluates the relative regret of the different FLoRA loss surfaces for different number of (HP, loss) pairs, varying it as `{5, 10, 20, 40, 60, 80}`. The dataset used is Heart Statlog for FL-HPO with MLP. Note that the following command runs the `loss_surface_analysis.py` script sequentially for each of the number of the (HP, loss) pairs specified via `-T`, each outputting in their respective results in the logs and in a pickle file such as `final_stats_T5.pkl` in the output directory.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/ -d datasets/dataset_53_heart-statlog.csv -f MLP-adam-prescale/hpo_v_default/dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv -l class -s balanced_accuracy -o MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/ -H -F 10 -M MLP-adam -X -T $t; \
  done
```

The above command will result in the following files in the `MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/` directory specified via `-o`:

```
final_stats_T5.pkl
final_stats_T10.pkl
final_stats_T20.pkl
final_stats_T40.pkl
final_stats_T60.pkl
final_stats_T80.pkl
```

### MLP-adam with Sonar

The following command runs the same experiment and generates results in the same form as above with the Sonar dataset for FL-HPO with MLP.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f MLP-adam-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M MLP-adam -X -T $t; \
  done
```

### SVM with Sonar

The following command runs the same experiment and generates results in the same form as above with the Sonar dataset for FL-HPO with SVM.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p SVM-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f SVM-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o SVM-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M SVM -X -T $t; \
  done
```

### SVM with EEG Eye State

The following command runs the same experiment and generates results in the same form as above with the EEG Eye State dataset for FL-HPO with SVM.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p SVM-prescale/eeg_eye_hp_bacc_iid3p_F10/ -d datasets/eeg_eye_state.csv -f SVM-prescale/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o SVM-prescale/eeg_eye_hp_bacc_iid3p_F10/ -H -F 10 -M SVM -X -T $t; \
  done
```

## Figure 3b: Effect of the communication overhead in FLoRA

Figure 3b is a visualization of the results presented in Appendix B.8. We make use of the `-T` option in conjunction with the `-R` option in `loss_surface_analysis.py` to specify the communication overhead of collecting all the (HP, loss) pairs at the aggregator by sending only the specified number (by `-T`) of pairs with the best loss values:

```
  -T NUM_PAIRS_PER_PARTY, --num_pairs_per_party NUM_PAIRS_PER_PARTY
                        Number of (HP, loss) pairs to send to aggregator from each party
  -R, --send_top        Whether to send the best-T (HP, loss) pairs
```

### MLP-adam with Heart Statlog

The following command evaluates the relative regret of the different FLoRA loss surfaces for different number of (HP, loss) pairs, varying it as `{5, 10, 20, 40, 60, 80}`. The dataset used is Heart Statlog for FL-HPO with MLP. Note that the following command runs the `loss_surface_analysis.py` script sequentially for each of the number of the (HP, loss) pairs specified via `-T`, each outputting in their respective results in the logs and in a pickle file such as `final_stats_R_T5.pkl` in the output directory.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/ -d datasets/dataset_53_heart-statlog.csv -f MLP-adam-prescale/hpo_v_default/dataset_53_heart-statlog_class_balanced_accuracy_k10_I30_R5_S4_full.csv -l class -s balanced_accuracy -o MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/ -H -F 10 -M MLP-adam -X -T $t -R; \
  done
```

The above command will result in the following files in the `MLP-adam-prescale/heart_statlog_hp_bacc_iid3p_F10/` directory specified via `-o`:

```
final_stats_R_T5.pkl
final_stats_R_T10.pkl
final_stats_R_T20.pkl
final_stats_R_T40.pkl
final_stats_R_T60.pkl
final_stats_R_T80.pkl
```

### MLP-adam with Sonar

The following command runs the same experiment and generates results in the same form as above with the Sonar dataset for FL-HPO with MLP.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f MLP-adam-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o MLP-adam-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M MLP-adam -X -T $t -R; \
  done
```

### SVM with Sonar

The following command runs the same experiment and generates results in the same form as above with the Sonar dataset for FL-HPO with SVM.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p SVM-prescale/sonar_hp_bacc_iid3p_F10/ -d datasets/dataset_40_sonar.csv -f SVM-prescale/hpo_v_default/dataset_40_sonar_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o SVM-prescale/sonar_hp_bacc_iid3p_F10/ -H -F 10 -M SVM -X -T $t -R; \
  done
```

### SVM with EEG Eye State

The following command runs the same experiment and generates results in the same form as above with the EEG Eye State dataset for FL-HPO with SVM.

```
> for t in 5 10 20 40 60 80 -1; do \
    python loss_surface_analysis.py -p SVM-prescale/eeg_eye_hp_bacc_iid3p_F10/ -d datasets/eeg_eye_state.csv -f SVM-prescale/hpo_v_default/eeg_eye_state_Class_balanced_accuracy_k10_I30_R5_S4_full.csv -l Class -s balanced_accuracy -o SVM-prescale/eeg_eye_hp_bacc_iid3p_F10/ -H -F 10 -M SVM -X -T $t -R; \
  done
```
