import json
from time import sleep

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)

from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (compute_metrics, cartesian_product,
                         compute_metrics_for_test_data_binary, DisinformationDataset)


transformers.set_seed(123)
train_data = pd.read_csv("data/disinformation/train.csv", encoding='utf-8')
validation_data = pd.read_csv("data/disinformation/validation.csv", encoding='utf-8')

model = AutoModelForSequenceClassification.from_pretrained('allegro/herbert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')

train_encodings = tokenizer(
    train_data['article'].tolist(), truncation=True, padding=True, max_length=512
)
val_encodings = tokenizer(
    validation_data['article'].tolist(), truncation=True, padding=True, max_length=512
)

train_dataset = DisinformationDataset(train_encodings, train_data['disinformation'].tolist())
val_dataset = DisinformationDataset(val_encodings, validation_data['disinformation'].tolist())

hyper_parameters_dict = {
    "evaluation_strategy": ["steps"], "learning_rate": [0.00003],
    "per_device_train_batch_size": [16],
    "per_device_eval_batch_size": [16], "num_train_epochs": [5], "warmup_steps": [200],
    "weight_decay": [0.1],
    "fp16": [True], "metric_for_best_model": ["f1_macro_weighted"], "load_best_model_at_end": [True],
    "save_total_limit": [2], "greater_is_better": [True], "save_strategy": ["steps"],
    "eval_steps": [500]
}

set_of_hyper_parameters = cartesian_product(hyper_parameters_dict)

for it, hyperparameters in enumerate(set_of_hyper_parameters):
    args = TrainingArguments(
        output_dir="output/training/dis_pl_hb" + str(it),
        evaluation_strategy=hyperparameters["evaluation_strategy"],
        learning_rate=hyperparameters["learning_rate"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=hyperparameters["per_device_eval_batch_size"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        warmup_steps=hyperparameters["warmup_steps"],
        weight_decay=hyperparameters["weight_decay"],
        fp16=hyperparameters["fp16"],
        metric_for_best_model=hyperparameters["metric_for_best_model"],
        load_best_model_at_end=hyperparameters["load_best_model_at_end"],
        save_total_limit=hyperparameters["save_total_limit"],
        greater_is_better=hyperparameters["greater_is_better"],
        save_strategy=hyperparameters["save_strategy"],
        eval_steps=hyperparameters["eval_steps"],
        seed=123
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SaveMetricsCallback(
            csv_file_name="metrics/disinformation/herbert_base/valid/dis_pl_hb" + str(it) + ".csv",
            hyperparameters=hyperparameters)
        ]
    )

    # Fine-tune the model for the binary classification task
    trainer.train()

    trainer.evaluate()
    model_saved_path = "output/final/dis_pl_hb" + str(it)
    trainer.save_model(model_saved_path)

    print("######################## Unseen data evaluation ########################################")

    test_data = pd.read_csv("data/disinformation/test.csv", encoding='utf-8')
    # test_data.rename({"disinformation": "labels"}, axis=True, inplace=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)


    def predict_label_for_test(text, tokenizer, model):
        """
        Function that predicts the label for input text using argmax
        """

        tokenized_text = tokenizer([text], truncation=True, padding=True, max_length=512, return_tensors="pt")
        # tokenized_text.to("cuda")
        # model.to("cuda")
        with torch.no_grad():
            outputs = model(**tokenized_text)

        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

        predicted_label = np.argmax(probabilities)
        return predicted_label


    test_data["predictions"] = test_data.article.apply(
        lambda x: predict_label_for_test(x, tokenizer, model))

    print(classification_report(test_data.disinformation, test_data["predictions"]))
    evaluation_results = compute_metrics_for_test_data_binary(test_data.disinformation, test_data["predictions"])
    # Save evaluation metrics to a JSON file
    output_file_path = "metrics/disinformation/herbert_base/test/dis_pl_hb" + str(it) + ".json"

    with open(output_file_path, 'w') as output_file:
        json.dump(evaluation_results, output_file, indent=4)

    print(f"Evaluation metrics saved to: {output_file_path}")
