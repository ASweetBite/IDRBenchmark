import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import json

# 强制镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_binary_dataset(parquet_path, ratio=2):
    """
    ratio: Vul样本数量 / Safe样本数量 的比例
    例如 ratio=2，则如果取 5000 个 Safe，就会取 10000 个 Vul
    """
    print(f"[*] 正在加载数据集 (按比例采样: 1 Safe : {ratio} Vul)...")
    df = pd.read_parquet(parquet_path)
    df['label'] = df['cwe'].apply(lambda x: 1 if x and x != "" else 0)

    df_safe = df[df['label'] == 0]
    df_vul = df[df['label'] == 1]

    print(f"[*] 原始分布: Safe={len(df_safe)}, Vul={len(df_vul)}")

    # 设定 Safe 采样上限，例如 20,000 条
    target_safe_count = 20000
    # 确保不超出总数
    n_safe = min(len(df_safe), target_safe_count)
    n_vul = min(len(df_vul), n_safe * ratio)

    df_safe_sampled = df_safe.sample(n=n_safe, random_state=42)
    df_vul_sampled = df_vul.sample(n=n_vul, random_state=42)

    # 合并并打乱
    df_final = pd.concat([df_safe_sampled, df_vul_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[+] 训练集采样完成: Safe={len(df_safe_sampled)}, Vul={len(df_vul_sampled)}")

    dataset = Dataset.from_pandas(df_final[['func', 'label']])
    return dataset


def train_and_save_models(dataset):
    model_map = {
        "CodeBERT": "microsoft/codebert-base",
    }

    # 二分类，num_labels 固定为 2
    num_labels = 2
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)

    for name, hf_path in model_map.items():
        save_path = f"./models/binary_diversevul_{name.lower()}"
        if os.path.exists(save_path):
            print(f"[*] 模型已存在: {save_path}，跳过。")
            continue

        print(f"\n🚀 开始微调: {name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model = AutoModelForSequenceClassification.from_pretrained(hf_path, num_labels=num_labels)
        model = get_peft_model(model, peft_config)

        def tokenize_function(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = dataset.map(tokenize_function, batched=True)
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # 使用标准 Trainer，因为数据已经均衡(1:1)，无需额外加权
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./temp_binary_{name}",
                per_device_train_batch_size=8,
                num_train_epochs=3,
                learning_rate=5e-5,
                save_strategy="epoch",
                report_to="none",
                fp16=torch.cuda.is_available()
            ),
            train_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        trainer.train()
        model.merge_and_unload().save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[+] {name} 已微调并保存至 {save_path}")


if __name__ == "__main__":
    # 执行流程
    dataset = prepare_binary_dataset("./data/full_dataset.parquet")
    train_and_save_models(dataset)