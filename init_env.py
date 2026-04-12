import os
import re
import json
import random
import logging
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
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

from test_spt import obfuscate

# =============== 环境配置 ===============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(last_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def augment_vulnerable_data(df_vul: pd.DataFrame, needed_count: int) -> pd.DataFrame:
    vul_funcs = df_vul['func'].tolist()

    # 预过滤：跳过超过 3000 字符的超大样本，这是卡死和拖慢速度的元凶
    vul_funcs = [f for f in vul_funcs if len(f) <= 3000]
    n_samples = len(vul_funcs)

    if n_samples == 0:
        return pd.DataFrame({'func': [], 'label': []})

    # 计算每个样本应该被增强的基础次数 (比如需要 50k, 现有 20k，每人分配 2 次)
    base_aug = needed_count // n_samples
    remainder = needed_count % n_samples

    # 均匀分配增强配额
    aug_counts = [base_aug] * n_samples
    for idx in random.sample(range(n_samples), remainder):
        aug_counts[idx] += 1

    augmented_funcs = []
    print(f"[*] 样本分配策略: 平均每个样本进行 {base_aug} 到 {base_aug + 1} 次混淆...")

    pbar = tqdm(total=needed_count, desc="Augmenting")

    # 遍历每个样本，执行指定次数的混淆
    for code, target_count in zip(vul_funcs, aug_counts):
        successful = 0
        attempts = 0
        # 给予容错空间：如果一个样本由于语法问题一直混淆失败，最多尝试 target_count * 3 次后放弃
        while successful < target_count and attempts < target_count * 3:
            attempts += 1
            try:
                new_code = obfuscate(code)
                augmented_funcs.append(new_code)
                successful += 1
                pbar.update(1)
            except Exception:
                continue

    # 如果因为部分样本报错导致总数没凑齐，进行最后的小额补足
    missing = needed_count - len(augmented_funcs)
    attempts = 0
    while missing > 0 and attempts < missing * 5:
        attempts += 1
        code = random.choice(vul_funcs)
        try:
            new_code = obfuscate(code)
            augmented_funcs.append(new_code)
            missing -= 1
            pbar.update(1)
        except Exception:
            pass

    pbar.close()
    return pd.DataFrame({'func': augmented_funcs, 'label': 1})


# =============== 🛠️ 核心数据准备逻辑 ===============

def prepare_dataset(parquet_path):
    print(f"[*] Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 直接使用 vul 字段作为 label
    # 假设 vul 已经是 0/1 或 True/False
    df['label'] = df['vul'].astype(int)

    # 基础清洗：长度过滤
    max_char_length = 4000
    df = df[df['func'].str.len() <= max_char_length].copy()

    df_safe = df[df['label'] == 0]
    df_vul = df[df['label'] == 1]

    # 实时统计打印，确保不再出现之前的统计错误
    real_vul_count = len(df_vul)
    real_safe_count = len(df_safe)
    print(f"[*] 真实标签分布: 安全(0)={real_safe_count}, 漏洞(1)={real_vul_count}")

    # 设定 1:1 平衡目标
    target_count = 80000

    # 1. 安全样本：从 31 万中随机下采样 3 万
    if real_safe_count > target_count:
        sampled_safe = df_safe.sample(n=target_count, random_state=42)
    else:
        sampled_safe = df_safe  # 如果不足 3 万则全取

    # 2. 漏洞样本：补齐到 3 万
    if real_vul_count < target_count:
        needed = target_count - real_vul_count
        print(f"[*] 漏洞样本缺口: {needed}，启动增强程序...")

        # 调用增强函数
        aug_df_vul = augment_vulnerable_data(df_vul, needed)

        # 合并【原生漏洞】+【增强漏洞】
        sampled_vul = pd.concat([df_vul, aug_df_vul], ignore_index=True)
    else:
        # 如果原生漏洞已经超过 3 万，则采样
        sampled_vul = df_vul.sample(n=target_count, random_state=42)

    # 3. 最终合并并彻底打乱顺序
    df_final = pd.concat([sampled_safe, sampled_vul]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[+] 训练集构建完成:")
    print(f"    - 原生安全样本: {len(sampled_safe)}")
    print(f"    - 原生漏洞样本: {min(real_vul_count, target_count)}")
    print(f"    - 混淆增强漏洞: {max(0, target_count - real_vul_count)}")
    print(f"    - 总计: {len(df_final)}")

    return Dataset.from_pandas(df_final[['func', 'label']])


# =============== 训练执行 (维持原样) ===============

def train_models(dataset):
    models_to_train = {
        "CodeBERT": {"path": "microsoft/codebert-base", "type": "transformer"},
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


if __name__ == "__main__":
    # 路径根据实际情况修改
    if os.path.exists("data/cleaned_dataset.parquet"):
        ds = prepare_dataset("data/cleaned_dataset.parquet")
        train_models(ds)
    else:
        print("Dataset not found.")