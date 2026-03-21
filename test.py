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
    # 1. 加载测试集
    print(f"[*] 正在加载测试数据: {test_data_path}")
    df = pd.read_parquet(test_data_path)
    df['label'] = df['cwe'].apply(lambda x: 1 if x and x != "" else 0)

    # ==========================================================
    # 【修改：固定总样本数为 2000】
    # ==========================================================
    TOTAL_SAMPLES = 2000

    # 统计各类别数据
    safe_df = df[df['label'] == 0]
    vuln_df = df[df['label'] == 1]

    print(f"[*] 原数据分布: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

    # 检查总数据量是否够 2000
    if len(safe_df) + len(vuln_df) < TOTAL_SAMPLES:
        raise ValueError(f"[!] 错误：总数据量 ({len(safe_df) + len(vuln_df)}) 不足 {TOTAL_SAMPLES} 条！")

    # 策略：默认尽量保持 1:1 平衡 (各 1000 条)
    target_safe = TOTAL_SAMPLES // 2
    target_vuln = TOTAL_SAMPLES - target_safe

    # 动态补偿：如果某一边的数据不足 1000，则把剩下的额度给另一边
    if len(safe_df) < target_safe:
        target_safe = len(safe_df)
        target_vuln = TOTAL_SAMPLES - target_safe
    elif len(vuln_df) < target_vuln:
        target_vuln = len(vuln_df)
        target_safe = TOTAL_SAMPLES - target_vuln

    # 随机采样并打乱
    safe_sampled = safe_df.sample(n=target_safe, random_state=42)
    vuln_sampled = vuln_df.sample(n=target_vuln, random_state=42)

    df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[*] 抽样后分布: Safe={len(df[df['label'] == 0])}, Vuln={len(df[df['label'] == 1])}, 总计={len(df)}")
    # ==========================================================

    # 转为 HuggingFace Dataset
    test_ds = Dataset.from_pandas(df[['func', 'label']])
    y_true = df['label'].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}, 测试样本数: {len(test_ds)}")

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
    evaluate_models("./data/big_vul.parquet", ["CodeBERT"])