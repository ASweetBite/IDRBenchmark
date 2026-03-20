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
    # 1. 加载并处理测试集
    print(f"[*] 正在加载测试数据: {test_data_path}")
    df = pd.read_parquet(test_data_path)

    # 二分类逻辑：cwe 有内容则为 1 (Vulnerable)，否则为 0 (Safe)
    df['label'] = df['cwe'].apply(lambda x: 1 if x and x != "" else 0)

    # 转为 HuggingFace Dataset
    test_ds = Dataset.from_pandas(df[['func', 'label']])
    y_true = df['label'].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}, 测试样本数: {len(test_ds)}")
    print(f"[*] 样本分布: {pd.Series(y_true).value_counts().to_dict()}")

    for name in model_names:
        model_path = f"./models/binary_diversevul_{name.lower()}"
        if not os.path.exists(model_path):
            print(f"[!] 模型 {name} 不存在于 {model_path}，跳过。")
            continue

        print(f"\n{'=' * 50}\n[*] 正在评估模型: {name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        # 2. Tokenization
        def tokenize(batch):
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

        # 3. 推理循环
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                y_pred.extend(preds)

        # 4. 生成报告
        print(f"\n[+] {name} 评估报告:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=["Safe", "Vulnerable"],
            digits=4
        ))


if __name__ == "__main__":
    # 请确保路径正确，并且测试集数据格式与训练集一致
    evaluate_models("./data/test_dataset.parquet", ["CodeBERT"])