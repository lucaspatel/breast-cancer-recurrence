from xgboost import XGBClassifier
from sklearn.metrics import log_loss, hinge_loss, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import os
from typing import Dict, Union


SKLEARN_MODELS = Union[
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
]

RANDOM_SEED = 2024

def _gini_loss(y_true: np.ndarray, y_est: np.ndarray):
    y_est = np.array([np.where(x == max(x))[0][0] for x in y_est])
    gini = 0.0
    for label_est in np.unique(y_est):
        indices = np.where(y_est == label_est)[0]
        p = 0.0
        for label_true in np.unique(y_true):
            p = len(np.where(y_true[indices] == label_true)[0]) / len(indices)
            gini += p * (1 - p) * len(indices)
    return gini


def make_model(
    model_type: str,
    hyperparameters: Dict[str, float],
    num_workers: int = 5,
) -> SKLEARN_MODELS:
    if model_type == "logreg":
        model = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=1000,
            C=hyperparameters["c"],
            l1_ratio=hyperparameters["l1_ratio"],
            penalty=hyperparameters["penalty"],
            n_jobs=num_workers,
            tol = 0.1,
            random_state=RANDOM_SEED,
        )
        model.loss_function = log_loss
        model.loss_name = "log"
    elif model_type == "randomforest":
        model = RandomForestClassifier(
            class_weight="balanced",
            max_depth=hyperparameters["max_depth"],
            max_leaf_nodes=hyperparameters["max_leaf_nodes"],
            min_samples_leaf=hyperparameters["min_samples_leaf"],
            min_samples_split=hyperparameters["min_samples_split"],
            n_estimators=hyperparameters["n_estimators"],
            n_jobs=num_workers,
            random_state=RANDOM_SEED,
        )
        model.loss_function = _gini_loss
        model.loss_name = "gini"
    elif model_type == "xgboost":
        model = XGBClassifier(
            colsample_bytree=hyperparameters["colsample_bytree"],
            gamma=hyperparameters["gamma"],
            learning_rate=hyperparameters["learning_rate"],
            max_depth=hyperparameters["max_depth"],
            min_child_weight=hyperparameters["min_child_weight"],
            n_estimators=hyperparameters["n_estimators"],
            objective="binary:logistic",
            use_label_encoder=False,
            n_jobs=num_workers,
            random_state=RANDOM_SEED,
        )
        model.loss_function = log_loss
        model.loss_name = "log"
    else:
        raise ValueError(f"Invalid model name: {model_type}")
    model.name = model_type
    return model


def train_and_evaluate_model(model, train, valid, test, output_folder=None):
    model.fit(train[0], train[1])
    performance_metrics = {}
    for split, dataset in [("train", train), ("valid", valid), ("test", test)]: 
        y_pred = model.predict_proba(dataset[0])[:, 1]
        y_true = dataset[1]
        if all(y_true == 1):
            auroc = None
        else:
            auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)

        results = {
            "auroc": auroc, "auprc": auprc, 
        }             
        performance_metrics.update(
            {f"{split}_{key}": value for key, value in results.items()},
        )
        if output_folder is not None:
            output_file = os.path.join(output_folder, f"{split}-predictions.csv")
            df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            df.to_csv(output_file)
    return performance_metrics
