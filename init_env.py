import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_labels=2):
        """Initializes a bidirectional LSTM architecture for sequence classification."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Performs a forward pass, concatenating the final hidden states of both LSTM directions."""
        x = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(last_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def train_models(dataset):
    """Orchestrates the training and saving of multiple transformer-based models and BiLSTM variants."""
    models_to_train = {
        "CodeBERT": {"path": "microsoft/codebert-base", "type": "transformer"},
        "GraphCodeBERT": {"path": "microsoft/graphcodebert-base", "type": "transformer"},
        "UniXcoder": {"path": "microsoft/unixcoder-base", "type": "transformer"},
    }

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)

    for name, info in models_to_train.items():
        save_path = f"./models/binary_diversevul_{name.lower()}"
        if os.path.exists(save_path):
            print(f"[*] Model {name} already exists, skipping.")
            continue

        print(f"\n🚀 Preparing model: {name}")

        if info["type"] == "transformer":
            model = AutoModelForSequenceClassification.from_pretrained(
                info["path"],
                num_labels=2,
                trust_remote_code=True
            )
            model = get_peft_model(model, peft_config)
        else:
            model = BiLSTMClassifier(vocab_size=len(tokenizer), num_labels=2)

        def tokenize_func(examples):
            """Tokenizes the source code snippets for model input."""
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = dataset.map(tokenize_func, batched=True)

        if info["type"] == "transformer":
            tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        else:
            tokenized_ds.set_format("torch", columns=["input_ids", "label"])

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./temp_{name}",
                per_device_train_batch_size=16,
                num_train_epochs=3,
                learning_rate=5e-5,
                save_strategy="epoch",
                report_to="none",
                fp16=torch.cuda.is_available()
            ),
            train_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer) if info["type"] == "transformer" else None,
        )

        trainer.train()

        if info["type"] == "transformer":
            model.merge_and_unload().save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)

        print(f"[+] {name} training completed and saved to: {save_path}")


def prepare_dataset(parquet_path):
    """Loads a Parquet dataset, labels vulnerability samples, and performs balanced sampling."""
    print(f"[*] Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    df['label'] = df['cwe'].apply(lambda x: 1 if (x is not None and str(x).strip() != "") else 0)

    df_safe = df[df['label'] == 0]
    df_vul = df[df['label'] == 1]

    target_safe = 10000
    target_vul = 20000

    if len(df_safe) < target_safe or len(df_vul) < target_vul:
        raise ValueError(
            f"Insufficient data samples! Needed Safe={target_safe} (Found {len(df_safe)}), "
            f"Needed Vul={target_vul} (Found {len(df_vul)})"
        )

    df_final = pd.concat([
        df_safe.sample(n=target_safe, random_state=42),
        df_vul.sample(n=target_vul, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(
        f"[+] Sampling completed: Safe={len(df_final[df_final['label'] == 0])}, Vul={len(df_final[df_final['label'] == 1])}")
    return Dataset.from_pandas(df_final[['func', 'label']])


if __name__ == "__main__":
    ds = prepare_dataset("data/diverse_vul.parquet")
    train_models(ds)