import warnings

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

warnings.filterwarnings("ignore")


def random_split(df, seed, test_split, val_split):
    df.loc[:, f"{seed}_rand"] = 1

    df_train, df_test = train_test_split(
        df, test_size=1 / test_split, random_state=seed, shuffle=True
    )
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test.loc[:, f"{seed}_rand"] = 0

    df_train, df_valid = train_test_split(
        df_train, test_size=1 / val_split, random_state=seed, shuffle=True
    )
    df_valid.loc[:, f"{seed}_rand"] = 2
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    df = pd.concat([df_train, df_valid, df_test], axis=0)
    df.sort_values(by="pid", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def stratified(df, col, seed, test_split, val_split):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / test_split, random_state=seed
    )
    # train / test
    for train_idx, test_idx in split.split(df, df[col]):
        df_strat_test = df.loc[test_idx].reset_index(drop=True)
        df_strat_train = df.loc[train_idx].reset_index(drop=True)
        df_strat_test[col + f"_{seed}_strat"] = 0

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / val_split, random_state=seed
    )
    for train_idx, val_idx in split.split(df_strat_train, df_strat_train[col]):
        df_strat_valid = df_strat_train.loc[val_idx].reset_index(drop=True)
        df_strat_train = df_strat_train.loc[train_idx].reset_index(drop=True)
        df_strat_valid[col + f"_{seed}_strat"] = 2

    df = pd.concat([df_strat_train, df_strat_valid, df_strat_test], axis=0)
    df.sort_values(by="pid", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def multiclass_multilabel_stratified(df, col, seed):
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0]])

    msss = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train_index, test_index in msss.split(
        np.array(df["pid"].tolist()), np.array(df["dx"].tolist())
    ):
        print("TRAIN:", train_index, "TEST:", test_index)
        df.loc[test_index, col + f"_{seed}_strat"] = 0

        break
    df_test = df[df[col + f"_{seed}_strat"] == 0].reset_index(drop=True)
    df_train = df[df[col + f"_{seed}_strat"] == 1].reset_index(drop=True)

    msss = MultilabelStratifiedKFold(n_splits=9, shuffle=True, random_state=seed)

    for train_index, valid_index in msss.split(
        np.array(df_train["pid"].tolist()), np.array(df_train["dx"].tolist())
    ):
        print("TRAIN:", train_index, "valid:", valid_index)
        df_train.loc[valid_index, col + f"_{seed}_strat"] = 2
        break

    df_train.reset_index(drop=True, inplace=True)
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    return df


def stratified_split(df, seed, test_split, val_split):
    pred_tasks = ["mort", "los3", "los7", "readm", "dx"]

    for col in pred_tasks:
        print("columns : ", col)
        if col in df.columns:
            df[col + f"_{seed}_strat"] = 1
            if col == "dx":
                df = multiclass_multilabel_stratified(df, col, seed)
            else:
                df = stratified(df, col, seed, test_split, val_split)
        else:
            raise AssertionError("Wrong, check!")

    return df
