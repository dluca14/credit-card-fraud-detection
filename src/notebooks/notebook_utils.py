import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as clrs
from dateutil import parser
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.simplefilter("ignore")
sys.path.append("..")
from storage.mongo_db import get_models, get_transactions
from sklearn.metrics import classification_report


def get_all_models():
    models = get_models()
    feature_importance_set = []
    for model in models:
        model_id = model["model_id"]
        time = model["created_timestamp"]
        validation_score_all = model["validation_data"]["validation_score_all"]
        feature_importance = model["validation_data"]["feature_importance"]
        for fold in feature_importance:
            crt_fold_id = fold["fold"]
            crt_fold = fold["fold_importance"]
            for item in crt_fold:
                feature_importance_set.append((model_id, time, validation_score_all,
                                               crt_fold_id, item["feature"], item["importance"]))
    feature_importance_df = pd.DataFrame(feature_importance_set)
    feature_importance_df.columns = ["model_id", "timestamp", "validation_score", "fold", "feature", "importance"]
    feature_importance_df["importance"] = feature_importance_df["importance"].apply(lambda x: float(x))
    print(f"Models: {feature_importance_df.model_id.nunique()}")
    print(f"Features: {feature_importance_df.feature.nunique()}")
    return feature_importance_df


def get_all_transactions():
    transactions = get_transactions()
    transaction_set = []
    transaction_details = []
    transaction_manual = []
    for transaction in transactions:
        transaction_id = transaction["_id"].split("/")[2]
        reg_type = transaction["_id"].split("/")[3]
        if reg_type == "inference.json":
            inference = transaction["inference"]["is_fraud"]
            timestamp = transaction["created_timestamp"]
            transaction_set.append((transaction_id, timestamp, inference))
        elif reg_type == "data.json":
            trans_id = transaction["file"]["transaction_id"]
            transaction_data = transaction["file"]["transaction_data"]
            transaction_details.append((trans_id, transaction_data))
        elif reg_type == "manual.json":
            manual_decision = transaction["manual"]["is_fraud"]
            transaction_manual.append((transaction_id, manual_decision))

    transaction_set_df = pd.DataFrame(transaction_set)
    transaction_set_df.columns = ["transaction_id", "timestamp", "is_fraud"]

    transaction_data_df = pd.DataFrame(transaction_details)
    transaction_data_df.columns = ["transaction_id", "transaction_data"]

    transaction_manual_df = pd.DataFrame(transaction_manual)
    transaction_manual_df.columns = ["transaction_id", "manual"]

    transaction_set_df = transaction_set_df.merge(transaction_data_df, on=["transaction_id"], how="inner")
    transaction_set_df = transaction_set_df.merge(transaction_manual_df, on=["transaction_id"], how="left")

    print(f"Transactions: {transaction_set_df.transaction_id.nunique()}")
    return transaction_set_df


def plot_feature_importance(model_id, feature_importance_df):
    df = feature_importance_df.loc[feature_importance_df.model_id == model_id]
    validation_score = df.validation_score.unique()[0]
    cols = (df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = df.loc[df.feature.isin(cols)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="feature", data=df.sort_values(by="importance", ascending=False))
    plt.title(f'Features importance (averaged/folds) \n model id: {model_id} validation score: {validation_score}')
    plt.tight_layout()


def plot_validation_score(feature_importance_df, x="model_id", sort_by="validation_score", ascending_flag=True):

    df = feature_importance_df[["model_id", "timestamp", "validation_score"]].drop_duplicates()
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(x=x, y="validation_score", data=df.sort_values(by=sort_by, ascending=ascending_flag))
    plt.title(f'Validation score / model')
    plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height, '{:1.4f}'.format(height),
                ha="center")


def plot_transactions(transaction_set_df, x="transaction_id", sort_by="timestamp", ascending_flag=True):

    df = transaction_set_df[["transaction_id", "timestamp", "is_fraud", "manual"]].sort_values(by="timestamp",
                                                                                               ascending=True)
    df.loc[df['manual'].isna(), "manual"] = "None"
    figure(num=None, figsize=(16, 5), dpi=200, facecolor='w', edgecolor='k')
    plt.title(f'Transactions: diamond: inference, cross: manual')
    plt.xticks(rotation=90, size=9)
    cmap = clrs.ListedColormap(['red', 'green'])
    cmap_manual = clrs.ListedColormap(['black', 'black'])
    plt.yticks([1.0, 0.0], ["Fraud", "Not Fraud"])
    plt.scatter(x=df[x], y=df["is_fraud"], c=(df["is_fraud"] != 'True').astype(float), s=100, marker='D', cmap=cmap)
    df_manual = df.loc[df.manual != "None"]
    plt.scatter(x=df_manual[x], y=df_manual["manual"], c='black', s=300, marker='x')


def get_metrics_manual_vs_automatic(transaction_set_df):
    total = transaction_set_df.shape[0]
    manual_df = transaction_set_df.loc[~transaction_set_df.manual.isna()]
    manual = manual_df.shape[0]
    print(f"Percent of manual confirmations: {round(manual / total * 100, 3)}%")
    print(classification_report(manual_df['manual'].values, manual_df['is_fraud']))
    return manual_df
