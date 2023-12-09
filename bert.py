import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch

csv_file_path = './db/CVEFixes.csv'
df = pd.read_csv(csv_file_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

languages = df['language'].unique()

for lang in languages:
    lang_train_df = train_df[train_df['language'] == lang]
    lang_test_df = test_df[test_df['language'] == lang]

    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)
    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            code = str(self.data.iloc[idx]['code'])
            text = f"{code} in {lang}"

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            label = 1 if self.data.iloc[idx]['safety'] == 'vulnerable' else 0
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    lang_train_dataset = CodeDataset(lang_train_df, tokenizer)
    lang_test_dataset = CodeDataset(lang_test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./{lang}_model/",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir=f"./{lang}_logs/",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    def compute_metrics(p):
        accuracy = (p.predictions.argmax(axis=1) == p.label_ids).mean()
        with open(f"./{lang}_logs/accuracy.txt", 'a') as f:
            f.write(f"Epoch: {accuracy}\n")
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lang_train_dataset,
        eval_dataset=lang_test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
