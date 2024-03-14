import json
import pandas as pd
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (
    load_config, compute_metrics, predict_disinformation,
    compute_metrics_for_test_data, DisinformationDataset
)

if __name__ == '__main__':

    config = load_config()

    train_data = pd.read_csv(config["disinformation"]["data"]["train"], encoding='utf-8')
    validation_data = pd.read_csv(config["disinformation"]["data"]["validation"], encoding='utf-8')

    for experiment in config["disinformation"]["models"]:
        for seed in [128, 256, 512, 1024, 2048]:

            transformers.set_seed(seed=seed)

            model = AutoModelForSequenceClassification.from_pretrained(experiment["model"])
            tokenizer = AutoTokenizer.from_pretrained(experiment["model"])

            train_encodings = tokenizer(
                train_data['article'].tolist(),
                truncation=config["disinformation"]["tokenizer"]["truncation"],
                padding=config["disinformation"]["tokenizer"]["padding"],
                max_length=config["disinformation"]["tokenizer"]["max_length"]
            )
            val_encodings = tokenizer(
                validation_data['article'].tolist(),
                truncation=config["disinformation"]["tokenizer"]["truncation"],
                padding=config["disinformation"]["tokenizer"]["padding"],
                max_length=config["disinformation"]["tokenizer"]["max_length"]
            )

            train_dataset = DisinformationDataset(train_encodings, train_data['disinformation'].tolist())
            val_dataset = DisinformationDataset(val_encodings, validation_data['disinformation'].tolist())

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
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
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

            test_data = pd.read_csv(config["disinformation"]["data"]["test"], encoding='utf-8')
            model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)

            test_data["predictions"] = test_data.article.apply(
                lambda x: predict_disinformation(x, tokenizer, model)
            )

            evaluation_results = compute_metrics_for_test_data(test_data.disinformation, test_data["predictions"])
            # Save evaluation metrics to a JSON file
            output_file_path = experiment["test_metrics"] + str(seed) + ".json"

            with open(output_file_path, 'w') as output_file:
                json.dump(evaluation_results, output_file, indent=4)
