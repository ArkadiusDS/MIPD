"""
Helper functions and labels in multi labels problems
"""
import itertools
import numpy as np
import torch
import yaml
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


def preprocess_data(examples, tokenizer, labels, padding, truncation, max_length):
    """
    Tokenizing and preprocessing data for model training
    """
    # take a batch of texts
    text = examples["article"]
    # encode them
    encoding = tokenizer(text, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


def load_config(file_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


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


def compute_metrics_for_test_data(y_true, y_pred):

    clf_report = classification_report(y_true, y_pred, output_dict=True)
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


def predict_disinformation(text, tokenizer, model):
    """
    Function that predicts the label for input text using argmax
    """

    tokenized_text = tokenizer([text], truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_text)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_label = np.argmax(probabilities)
    return predicted_label


def multi_label_metrics(predictions, labels, target_names, threshold=0.15):
    """
    Function with last layer for multilabel classification and computing metrics
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    clf_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_macro_weighted': f1_macro_weighted,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'classification_report': clf_report}
    return metrics


config = load_config()
LABELS_INTENTION = config["intention"]["data"]["labels"]


def compute_metrics_intention(p: EvalPrediction):
    """
    Function for computing metrics
    """
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        target_names=LABELS_INTENTION,
        labels=p.label_ids)
    return result
