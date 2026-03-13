from datasets import load_dataset
from transformers import AutoTokenizer

SAVED_MODEL_PATH = "./results/best_model"
TOKENIZER_NAME   = "bert-base-uncased"

LABEL_MAP = {"select": 1, "reject": 0}


def load_and_prepare():
    """
    Loads the dataset, applies a 60/20/20 train/val/test split,
    fixes labels, tokenizes, and returns everything needed for
    training and analysis.
    """
    dataset = load_dataset("ranaatef/Resume-Screening-Dataset")

    # 3-way split: 60% train / 20% val / 20% test
    temp  = dataset["train"].train_test_split(test_size=0.2, seed=42)
    split = temp["train"].train_test_split(test_size=0.25, seed=42)
    train = split["train"]
    val   = split["test"]
    test  = temp["test"]

    # Save raw texts before tokenization (needed for TCAV/counterfactuals)
    train_texts_raw = list(train["Resume"])
    test_texts_raw  = list(test["Resume"])

    def fix_labels(example):
        val_str = str(example["Decision"]).strip().lower()
        example["Decision"] = LABEL_MAP.get(val_str, int(val_str) if val_str.isdigit() else 0)
        return example

    train = train.map(fix_labels)
    val   = val.map(fix_labels)
    test  = test.map(fix_labels)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def tokenize(batch):
        return tokenizer(batch["Resume"], padding="max_length", truncation=True, max_length=256)

    train = train.map(tokenize, batched=True)
    train = train.rename_column("Decision", "labels")
    train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    val = val.map(tokenize, batched=True)
    val = val.rename_column("Decision", "labels")
    val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    test = test.map(tokenize, batched=True)
    test = test.rename_column("Decision", "labels")
    test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train = train.select(range(500))
    val   = val.select(range(125))
    test  = test.select(range(125))
    train_texts_raw = train_texts_raw[:500]
    test_texts_raw  = test_texts_raw[:125]

    return train, val, test, train_texts_raw, test_texts_raw, tokenizer
