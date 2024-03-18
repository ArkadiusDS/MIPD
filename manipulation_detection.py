import json

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (
    load_config, preprocess_data, extract_unique_strings,
    labels_one_hot_encoding, compute_metrics_for_test_data_int, compute_metrics_manipulation
)


if __name__ == '__main__':

    config = load_config()

    train_data = pd.read_csv(config["manipulation"]["data"]["train"], encoding='utf-8')
    validation_data = pd.read_csv(config["manipulation"]["data"]["validation"], encoding='utf-8')
    train_data = Dataset.from_pandas(train_data)
    validation_data = Dataset.from_pandas(validation_data)

    dataset = DatasetDict({
        'train': train_data,
        'valid': validation_data
    })

    LABELS_MANIPULATION = config["manipulation"]["data"]["labels"]

    id2label = {idx: label for idx, label in enumerate(LABELS_MANIPULATION)}
    label2id = {label: idx for idx, label in enumerate(LABELS_MANIPULATION)}

    for experiment in config["manipulation"]["models"]:
        for seed in [128, 256, 512, 1024, 2048]:

            transformers.set_seed(seed=seed)
            tokenizer = AutoTokenizer.from_pretrained(experiment["model"])
            encoded_dataset = dataset.map(
                lambda example: preprocess_data(
                    examples=example,
                    tokenizer=tokenizer,
                    labels=LABELS_MANIPULATION,
                    padding=config["manipulation"]["tokenizer"]["padding"],
                    truncation=config["manipulation"]["tokenizer"]["truncation"],
                    max_length=config["manipulation"]["tokenizer"]["max_length"]
                ),
                batched=True, remove_columns=dataset['train'].column_names)
            encoded_dataset.set_format("torch")

            bert_model = AutoModelForSequenceClassification.from_pretrained(
                experiment["model"],
                problem_type="multi_label_classification",
                num_labels=len(LABELS_MANIPULATION),
                id2label=id2label,
                label2id=label2id
            )

            args = TrainingArguments(
                output_dir=experiment["output"] + str(seed),
                evaluation_strategy=experiment["hyperparameters"]["evaluation_strategy"],
                learning_rate=experiment["hyperparameters"]["learning_rate"],
                per_device_train_batch_size=experiment["hyperparameters"]["per_device_train_batch_size"],
                per_device_eval_batch_size=experiment["hyperparameters"]["per_device_eval_batch_size"],
                num_train_epochs=experiment["hyperparameters"]["num_train_epochs"],
                warmup_steps=experiment["hyperparameters"]["warmup_steps"],
                weight_decay=experiment["hyperparameters"]["weight_decay"],
                fp16=experiment["hyperparameters"]["fp16"],
                metric_for_best_model=experiment["hyperparameters"]["metric_for_best_model"],
                load_best_model_at_end=experiment["hyperparameters"]["load_best_model_at_end"],
                save_total_limit=experiment["hyperparameters"]["save_total_limit"],
                greater_is_better=experiment["hyperparameters"]["greater_is_better"],
                save_strategy=experiment["hyperparameters"]["save_strategy"],
                eval_steps=experiment["hyperparameters"]["eval_steps"],
                seed=seed
            )

            trainer = Trainer(
                bert_model,
                args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset["valid"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_manipulation,
                callbacks=[SaveMetricsCallback(
                    csv_file_name=experiment["valid_metrics"] + str(seed) + ".csv",
                    hyperparameters=experiment["hyperparameters"])
                ]
            )

            # Fine-tune the model for the binary classification task
            trainer.train()
            model_saved_path = experiment["path_to_save_model"] + str(seed)

            trainer.save_model(model_saved_path)

            print("######################## Unseen data evaluation ########################################")

            model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)


            def predict_label_for_test(text, tokenizer, model, id2label):
                """
                Function that predicts labels for development dataset
                """
                encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                encoding = {k: v.to(model.device) for k, v in encoding.items()}

                outputs = model(**encoding)

                logits = outputs.logits

                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze().cpu())
                predictions = np.zeros(probs.shape)
                predictions[np.where(probs >= 0.15)] = True
                predicted_labels = ",".join(id2label[idx] for idx, label in enumerate(predictions) if label == 1.0)
                return predicted_labels


            test_data = pd.read_csv(config["manipulation"]["data"]["test"], encoding='utf-8')
            test_predictions = test_data.copy()
            test_predictions["predicted_labels"] = test_data.article.apply(
                lambda x: predict_label_for_test(x, tokenizer, model, id2label))

            labels = test_data.iloc[:, 1:]
            pred = pd.DataFrame(test_predictions.predicted_labels.apply(extract_unique_strings))
            pred = labels_one_hot_encoding(pred, "predicted_labels")
            test_pred = pd.DataFrame(np.zeros(labels.shape))
            test_pred.columns = labels.columns
            test_pred[pred.columns] = pred

            evaluation_results = compute_metrics_for_test_data_int(
                y_true=np.array(labels[labels.columns]),
                y_pred=np.array(test_pred[labels.columns].applymap(lambda x: 1 if x is True else 0)),
                target_names=labels.columns
            )

            # # Save evaluation metrics to a JSON file
            output_file_path = experiment["test_metrics"] + str(seed) + ".json"

            with open(output_file_path, 'w') as output_file:
                json.dump(evaluation_results, output_file, indent=4)

            print(f"Evaluation metrics saved to: {output_file_path}")
