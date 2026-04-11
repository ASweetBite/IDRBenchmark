import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report
from tqdm import tqdm
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate_models(test_data_path, model_names):
    """Loads a test dataset and evaluates multiple models, providing a detailed classification report."""
    print(f"[*] Loading test data from: {test_data_path}")
    df = pd.read_parquet(test_data_path)
    df['label'] = df['cwe'].apply(lambda x: 1 if x and x != "" else 0)

    TOTAL_SAMPLES = 2000

    safe_df = df[df['label'] == 0]
    vuln_df = df[df['label'] == 1]

    print(f"[*] Original data distribution: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

    if len(safe_df) + len(vuln_df) < TOTAL_SAMPLES:
        raise ValueError(f"[!] Error: Total data count ({len(safe_df) + len(vuln_df)}) is less than {TOTAL_SAMPLES}!")

    target_safe = TOTAL_SAMPLES // 2
    target_vuln = TOTAL_SAMPLES - target_safe

    if len(safe_df) < target_safe:
        target_safe = len(safe_df)
        target_vuln = TOTAL_SAMPLES - target_safe
    elif len(vuln_df) < target_vuln:
        target_vuln = len(vuln_df)
        target_safe = TOTAL_SAMPLES - target_vuln

    safe_sampled = safe_df.sample(n=target_safe, random_state=42)
    vuln_sampled = vuln_df.sample(n=target_vuln, random_state=42)

    df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[*] Sampled distribution: Safe={len(df[df['label'] == 0])}, Vuln={len(df[df['label'] == 1])}, Total={len(df)}")

    test_ds = Dataset.from_pandas(df[['func', 'label']])
    y_true = df['label'].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}, Test sample size: {len(test_ds)}")

    for name in model_names:
        model_path = f"./models/binary_diversevul_{name.lower()}"
        if not os.path.exists(model_path):
            print(f"[!] Model {name} not found at {model_path}, skipping.")
            continue

        print(f"\n{'=' * 50}\n[*] Evaluating model: {name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        def tokenize(batch):
            """Tokenizes the code function for model input."""
            return tokenizer(batch['func'], truncation=True, max_length=512)

        tokenized_ds = test_ds.map(tokenize, batched=True)
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(
            tokenized_ds,
            batch_size=32,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
            pin_memory=True
        )

        y_pred = []

        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                y_pred.extend(preds)

        print(f"\n[+] {name} Evaluation Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=["Safe", "Vulnerable"],
            digits=4
        ))


if __name__ == "__main__":
    evaluate_models("./data/big_vul.parquet", ["CodeBERT"])