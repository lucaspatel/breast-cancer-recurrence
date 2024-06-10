from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.decomposition import PCA

from scripts.models import make_model, train_and_evaluate_model
from scripts.data_loader import load_data

from itertools import product 
import numpy as np
import pandas as pd
import warnings
import copy
import os

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)


hyperparameter_options = {
    "logreg":{
        "c": [0.1, 0.5, 1, 10, 100],
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 1, None],
        "penalty": ["elasticnet", "l2", "l1", None],
    },
    "xgboost": {
        "colsample_bytree": [0.5, 0.75],
        "gamma": [0.5, 0.8],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 10, 20],
        "min_child_weight": [0.5, 1, 3],
        "n_estimators": [10, 20, 30],
    },
    "randomforest": {
        "max_depth": [3, 5, 10, 20],
        "max_leaf_nodes": [2, 20, 30],
        "min_samples_leaf": [3, 5, 10, 25],
        "min_samples_split": [2, 4, 6],
        "n_estimators": [10, 20, 30],
    }
}


def generate_hyperparameter_combinations(model_type):
    hyperparameters = hyperparameter_options[model_type]
    combs = product(*hyperparameters.values())
    for comb in combs:
        yield dict(zip(hyperparameters.keys(), comb))
    

with open("recurrence_genes.txt", "r") as f:
    text = f.read()
    recurrence_genes = text.split("\n")


def hyperoptimize_model(
    label,
    model_type,
    output_folder="./results",
    features="all",
    n_splits=5,
):
    if features not in ["all", "pc", "recurrence_pc", "cancer_1", "cancer_2", "cancer_12"]:
        raise ValueError("Unkown set of features")
    output_folder = os.path.join(output_folder, f"{model_type}-{label}-{features}")  
    os.makedirs(output_folder, exist_ok=True)
    if os.path.isfile(os.path.join(output_folder, "test-results-all.csv")):
        print("Results already calculated for this model")
        return
   
    kf = StratifiedKFold(n_splits=n_splits, random_state=2024, shuffle=True)
    tpm_table, _, val_tpm_table, _ = load_data()
    featues_columns = list(tpm_table.columns)[:-30]
    X = tpm_table[featues_columns].astype(float)
    
    if features == "cancer_1":
        X = X[["ENSG00000216184"]]
    elif features == "cancer_2":
        X = X[["ENSG00000199165"]]
    elif features == "cancer_12":
        X = X[["ENSG00000216184", "ENSG00000199165"]]
    elif features == "pc":
        X = X[featues_columns]
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X, index=tpm_table.index)
    elif features == "recurrence_pc":
        X = X[recurrence_genes]
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X, index=tpm_table.index)
    # X=(X-X.min())/(X.max()-X.min())

    tpm_table["recurrence"] = tpm_table["recurStatus"].str.replace("N", "0").str.replace("R", "1").astype(float)
    y = copy.deepcopy(tpm_table[["recurrence"]])
    y["cancer"] = 1
    non_cancer = y["recurrence"].isnull()
    y.loc[non_cancer, "cancer"] = 0
    y = y[label]

    X_val = val_tpm_table[featues_columns].astype(float)
    if features == "cancer_1":
        X_val = X_val[["ENSG00000216184"]]
    elif features == "cancer_2":
        X_val = X_val[["ENSG00000199165"]]
    elif features == "cancer_12":
        X_val = X_val[["ENSG00000216184", "ENSG00000199165"]]
    elif features == "pc":
        X_val = X_val[featues_columns]
        X_val = pca.transform(X_val)
        X_val = pd.DataFrame(X_val, index=val_tpm_table.index)
    elif features == "recurrence_pc":
        X_val = X_val[recurrence_genes]
        X_val = pca.transform(X_val)
        X_val = pd.DataFrame(X_val, index=val_tpm_table.index)
        
    val_tpm_table["recurrence"] = val_tpm_table.loc[:, "Recurrence Staus at the time of collection"].str.replace("Nonrecurrent", "0").str.replace("Recurrent", "1").astype(float)
    y_val = copy.deepcopy(val_tpm_table[["recurrence"]])
    y_val.loc[:, "cancer"] = 1
    y_val = y_val[label]
    
    if label == "recurrence":
        y = y[~non_cancer]
        X = X[~non_cancer]
        non_cancer = y_val.isnull()
        y_val = y_val[~non_cancer]
        X_val = X_val[~non_cancer]    

    rows = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Training models for split {i+1}/{n_splits}...")
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        for j, hyperparameters in enumerate(generate_hyperparameter_combinations(model_type)):
            if model_type == "logreg":
                if hyperparameters["penalty"] != "elasticnet" and hyperparameters["l1_ratio"] is not None:
                    continue
                elif hyperparameters["penalty"] == "elasticnet" and hyperparameters["l1_ratio"] is None:
                    continue
                elif hyperparameters["penalty"] is None and hyperparameters["c"] != 1:
                    continue
            model = make_model(model_type, hyperparameters)
            performance_metrics = train_and_evaluate_model(
                model, (X_train, y_train), (X_val, y_val), (X_test, y_test), output_folder=None
            )
            row = [model_type, j, i, performance_metrics["test_auroc"], performance_metrics["test_auprc"]]
            rows.append(row)
    df = pd.DataFrame(rows, columns=["model", "hyperparams", "k", "auroc", "auprc"])
    df.to_csv(os.path.join(output_folder, "test-results-all.csv"), index=False)
    df_mean = df.groupby(["model", "hyperparams"]).mean().reset_index()
    df_mean = df_mean.sort_values(by=["model", "auroc"], ascending=False)
    model_row = df_mean.drop_duplicates(subset=["model"], keep='first')
    model_row = model_row.iloc[-1]
    print("Test performance: \n", model_row)

    rows = []
    model_type = model_row.model
    hyperparameters_sets = list(generate_hyperparameter_combinations(model_type))
    hyperparameters = hyperparameters_sets[model_row.hyperparams]
    print(hyperparameters)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        output_folder_ = os.path.join(output_folder, f"{i}")
        os.makedirs(output_folder_, exist_ok=True)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        model = make_model(model_type, hyperparameters)
        performance_metrics = train_and_evaluate_model(
            model, (X_train, y_train), (X_val, y_val), (X_test, y_test), output_folder=output_folder_
        )
        row = [model_type, i, performance_metrics["valid_auroc"], performance_metrics["valid_auprc"]]
        rows.append(row)  
    if label == "cancer":
        return
    df = pd.DataFrame(rows, columns=["model", "k", "auroc", "auprc"])
    df.to_csv(os.path.join(output_folder, "validation-results-all.csv"), index=False)
    df_mean = df.groupby(["model"]).mean().sort_values(by=["model"], ascending=False)
    df_std = df.groupby(["model"]).std().sort_values(by=["model"], ascending=False)
    df = df_mean.round(3).astype(str) + " +/- " + df_std.round(3).astype(str)
    df = df.drop(["k"], axis=1)
    df.to_csv(os.path.join(output_folder, "validation-results.csv"), index=False)
    print("Valid performance: \n", df.iloc[-1])
              