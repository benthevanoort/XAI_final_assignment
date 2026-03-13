from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

from data_utils import load_and_prepare, SAVED_MODEL_PATH

train, val, test, train_texts_raw, test_texts_raw, tokenizer = load_and_prepare()

if os.path.exists(SAVED_MODEL_PATH):
    print("Model already trained and saved at", SAVED_MODEL_PATH)
    print("Delete ./results/best_model to retrain.")
else:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(SAVED_MODEL_PATH)
    print("Model saved to", SAVED_MODEL_PATH)
