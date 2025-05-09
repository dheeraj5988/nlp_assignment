from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def load_data():
    df = pd.read_csv("data/cleaned_chat.csv")
    df = df.dropna()
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return Dataset.from_pandas(df[['cleaned_message', 'label']])

def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(batch["cleaned_message"], padding=True, truncation=True)

def train():
    dataset = load_data()
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    model.save_pretrained("models/bert_sentiment_model")
    print("Model saved to models/bert_sentiment_model")

# Example usage
# train()
