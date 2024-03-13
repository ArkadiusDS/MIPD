"""
Helper functions and labels in multi labels problems
"""
import itertools
import re

import numpy as np
import torch
import pandas as pd
from datasets import DatasetDict
from transformers import EvalPrediction
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, accuracy_score, precision_score, recall_score
)


# Create a PyTorch Dataset for the training and validation sets
class DisinformationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def cartesian_product(hyperparameters):
    """ Returns cartesian product of all hyperparameters given in dictionary of lists"""
    return (dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values()))


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    f1 = f1_score(labels, y_pred)
    f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=labels, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=labels, y_pred=y_pred, average='weighted')
    acc = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    return {
        'f1': f1,
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_macro_weighted': f1_macro_weighted,
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }


def compute_metrics_for_test_data_binary(y_true, y_pred):

    clf_report = classification_report(y_true, y_pred,  output_dict=True)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        'f1': f1,
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_macro_weighted': f1_macro_weighted,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'classification_report': clf_report
    }

    return metrics
