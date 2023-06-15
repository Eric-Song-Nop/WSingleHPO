import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

data_files = [
    "electricity-normalized",
    "dataset_40_sonar",
    "dataset_53_heart-statlog",
    "oil_spill",
    "pc3",
    "pollen",
    "eeg_eye_state",
]

label_cols = [
    "class",
    "Class",
    "class",
    "class",
    "c",
    "binaryClass",
    "Class",
]

d2l_map = {d: l for d, l in zip(data_files, label_cols)}

def labels2binary(olabels):
    lmap = {
        'P': 1, 'N': 0,
        'g': 0, 'h': 1,
        'V1': 0, 'V2': 1,
        "'-1'": 0, "'1'": 1,
        -1: 0, 2: 0,
        'UP': 1, 'DOWN': 0,
        'Rock': 1, 'Mine': 0,
        'g': 0, 'b': 1,
        False: 0, True: 1,
        'absent': 0, 'present': 1,
        "'Normal'": 0, "'Anomaly'": 1,
    }
    ret = []
    for l in olabels:
        assert l in lmap, (f'{l} -> {lmap.keys()}')
        ret += [lmap[l]]
    return np.array(ret)

def get_full_data(data_path, clabel, prescale=False):
    fulldf = pd.read_csv(data_path)
    assert clabel in list(fulldf)
    X = fulldf.drop([clabel], axis=1).values
    y = labels2binary(fulldf[clabel])
    print(
        f'Data: {fulldf.shape} -> {X.shape}, {y.shape} {np.unique(y)}\n'
        f'{fulldf[clabel].value_counts()}'
    )
    if prescale:
        print('Scaling the features with min-max scaler ... ')
        X = minmax_scale(X)
    return X, y
