import gc
import os
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")

from storage.mongo_db import register_model, get_models_validation_score

data_path = "../../../../../data/credit-card-fraud-detection/"


def generate_static_catalog():
    path = os.path.abspath(os.path.dirname(__file__))
    path_to_catalog = f"{path}/training.json"
    with open(path_to_catalog, encoding='utf-8') as file:
        static_catalog = json.load(file)
    return static_catalog["training_parameters"], static_catalog["model_parameters"]


training_parameters, model_parameters = generate_static_catalog()


def read_data():
    data_df = pd.read_csv(os.path.join(data_path, "credit_card_transactions.csv"))
    print(f"Credit Card Fraud Detection data -  rows: {data_df.shape[0]} columns: {data_df.shape[1]}")
    return data_df


def data_train_test_split(data_df):
    train_df, test_df = train_test_split(data_df, test_size=training_parameters["test_size"], shuffle=True,
                                         random_state=training_parameters["random_state"], stratify=data_df["Class"])
    return train_df, test_df


def get_predictors_target():
    target = "Class"
    predictors = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                  "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                  "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                  "Amount"]
    return predictors, target


def train_model(train_df, test_df, predictors, target, run_optimization):
    kf = StratifiedKFold(n_splits=training_parameters["folds"],
                         random_state=training_parameters["random_state"], shuffle=True)

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance = []
    validation_score= []
    n_fold = 0

    if run_optimization:
        print("Not implemented")
        return None

    for fold_, (train_idx, valid_idx) in enumerate(kf.split(train_df, y=train_df[target])):
        train_x, train_y = train_df[predictors].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[predictors].iloc[valid_idx], train_df[target].iloc[valid_idx]

        model = LGBMClassifier(**model_parameters)
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                  eval_metric=model_parameters["metric"], verbose=training_parameters["verbose_eval"],
                  early_stopping_rounds=training_parameters["early_stop"])

        oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
        test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits

        fold_importance = []
        for i, item in enumerate(predictors):
            fold_importance.append({"feature": str(predictors[i]), "importance": str(model.feature_importances_[i])})
        feature_importance.append({"fold": str(fold_ + 1), "fold_importance": fold_importance})

        print(f"Fold {fold_ + 1} AUC : {round(roc_auc_score(valid_y, oof_preds[valid_idx]), 4)}")
        validation_score.append({"fold": str(fold_ + 1), "auc": round(roc_auc_score(valid_y, oof_preds[valid_idx]), 4)})

        y_pred = model.predict_proba(test_df[predictors])[:, 1]
        test_auc_score = roc_auc_score(test_df[target], y_pred)
        print(f"===========================\n[TEST] fold: {fold_ + 1} AUC score test set: {round(test_auc_score, 4)}\n")

        del model, train_x, train_y, valid_x, valid_y
        gc.collect()
    train_auc_score = roc_auc_score(train_df[target], oof_preds)
    print(f"Full AUC validation score {round(train_auc_score, 4)}\n")

    print("Train using all data")
    model = LGBMClassifier(**model_parameters)
    model.fit(train_df[predictors], train_df[target],
              eval_set=[(train_df[predictors], train_df[target]), (test_df[predictors], test_df[target])],
              eval_metric="auc", verbose=training_parameters["verbose_eval"],
              early_stopping_rounds=training_parameters["early_stop"])

    y_pred = model.predict_proba(test_df[predictors])[:, 1]
    test_auc_score = roc_auc_score(test_df["Class"], y_pred)
    print(f"===========================\n[TEST] AUC score test set: {round(test_auc_score, 4)}\n")

    model_data = {"train_rows": train_df.shape[0], "train_columns": len(predictors)}
    validation_data = {"validation_score_folds": validation_score,
                       "validation_score_all": round(train_auc_score, 4),
                       "feature_importance": feature_importance}

    model_id = register_model(model_data=model_data, model_parameters=model_parameters,
                              training_parameters=training_parameters, validation_data=validation_data)
    return model, model_id, validation_data["validation_score_all"]


def save_model(model):
    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    try:
        joblib.dump(model, os.path.join(path, "model", "model_light_gbm.pkl"))
    except:
        print("Error writing model")
        pass


def load_model():
    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    try:
        model = joblib.load(os.path.join(path, "model", "model_light_gbm.pkl"))
        return model
    except:
        print("Error reading model")
        pass


def test_model(model, test_df, predictors):
    model = load_model()
    y_pred = model.predict_proba(test_df[predictors])[:, 1]
    test_auc_score = roc_auc_score(test_df["Class"], y_pred)
    print(f"===========================\nAUC score test set: {round(test_auc_score, 4)}")


def check_validation_score(model_id, validation_score):
    validation_scores = get_models_validation_score()
    if validation_scores:
        for current_score in validation_scores:
            if current_score["model_id"] != model_id and validation_score > current_score["validation_score"]:
                return True

    return False


def run_all(run_optimization=False, run_test=False):

    data_df = read_data()
    train_df, test_df = data_train_test_split(data_df)
    predictors, target = get_predictors_target()
    model, model_id, validation_score = train_model(train_df, test_df, predictors, target, run_optimization)
    if check_validation_score(model_id, validation_score):
        save_model(model)
    if run_test:
        model = load_model()
        test_model(model, test_df, predictors)
    return model_id, validation_score


if __name__ == "__main__":
    run_all(run_test=True)
