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

# 强制镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 1. BiLSTM 模型定义 (自定义)
# ==========================================
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
        # 取双向 LSTM 的最后一个 hidden state 拼接
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(last_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==========================================
# 2. 数据处理与采样
# ==========================================
def prepare_dataset(parquet_path):
    print(f"[*] 正在加载数据集: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 构造 label: CWE 有内容为 1，无内容为 0
    df['label'] = df['cwe'].apply(lambda x: 1 if (x is not None and str(x).strip() != "") else 0)

    df_safe = df[df['label'] == 0]
    df_vul = df[df['label'] == 1]

    # 采样目标
    target_safe = 10000
    target_vul = 20000

    if len(df_safe) < target_safe or len(df_vul) < target_vul:
        raise ValueError(
            f"数据集数量不足！Safe需要{target_safe}(现有{len(df_safe)})，Vul需要{target_vul}(现有{len(df_vul)})")

    df_final = pd.concat([
        df_safe.sample(n=target_safe, random_state=42),
        df_vul.sample(n=target_vul, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[+] 采样完成: Safe={len(df_final[df_final['label'] == 0])}, Vul={len(df_final[df_final['label'] == 1])}")
    return Dataset.from_pandas(df_final[['func', 'label']])


# ==========================================
# 3. 训练主逻辑
# ==========================================
def train_models(dataset):
    # 配置模型信息
    models_to_train = {
        "CodeBERT": {"hf_path": "microsoft/codebert-base", "type": "transformer"},
        "GraphCodeBERT": {"hf_path": "microsoft/graphcodebert-base", "type": "transformer"},
        # "BiLSTM": {"hf_path": None, "type": "bilstm"}
    }

    # 统一使用 CodeBERT 的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # PEFT 配置 (仅用于 Transformer)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)

    for name, info in models_to_train.items():
        save_path = f"./models/binary_diversevul_{name.lower()}"
        if os.path.exists(save_path):
            print(f"[*] 模型 {name} 已存在，跳过。")
            continue

        print(f"\n🚀 开始微调: {name}")

        # 模型初始化
        if info["type"] == "transformer":
            model = AutoModelForSequenceClassification.from_pretrained(info["hf_path"], num_labels=2)
            model = get_peft_model(model, peft_config)
        else:
            model = BiLSTMClassifier(vocab_size=len(tokenizer), num_labels=2)

        # 数据集准备
        def tokenize_func(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = dataset.map(tokenize_func, batched=True)
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # 训练参数
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
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        trainer.train()

        # 保存模型
        if info["type"] == "transformer":
            model.merge_and_unload().save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            # 对于 BiLSTM，保存 state_dict 并额外存一个配置文件以便识别
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)

        print(f"[+] {name} 训练完成并已保存至: {save_path}")


if __name__ == "__main__":
    # 请根据实际路径修改
    ds = prepare_dataset("data/diverse_vul.parquet")
    train_models(ds)