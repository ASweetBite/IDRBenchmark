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
# 1. BiLSTM 模型定义
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
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(last_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

# ==========================================
# 2. 训练主逻辑
# ==========================================
def train_models(dataset):
    # 【修复点 1】修改 UniXcoder 的路径，并规范化配置
    models_to_train = {
        "CodeBERT": {"path": "microsoft/codebert-base", "type": "transformer"},
        "GraphCodeBERT": {"path": "microsoft/graphcodebert-base", "type": "transformer"},
        "UniXcoder": {"path": "microsoft/unixcoder-base", "type": "transformer"},
        # "BiLSTM": {"path": None, "type": "bilstm"}
    }

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)

    for name, info in models_to_train.items():
        save_path = f"./models/binary_diversevul_{name.lower()}"
        if os.path.exists(save_path):
            print(f"[*] 模型 {name} 已存在，跳过。")
            continue

        print(f"\n🚀 开始准备: {name}")

        # 【修复点 2】更稳健的模型初始化
        if info["type"] == "transformer":
            # 关键：一定要加 trust_remote_code=True
            model = AutoModelForSequenceClassification.from_pretrained(
                info["path"],
                num_labels=2,
                trust_remote_code=True
            )
            model = get_peft_model(model, peft_config)
        else:
            model = BiLSTMClassifier(vocab_size=len(tokenizer), num_labels=2)

        # 数据预处理
        def tokenize_func(examples):
            # BiLSTM 也可以共用 tokenizer 进行 ids 转换，但模型只取 input_ids
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = dataset.map(tokenize_func, batched=True)
        # 注意：BiLSTM 不需要 attention_mask，这里通过 columns 控制传入参数
        if info["type"] == "transformer":
            tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        else:
            tokenized_ds.set_format("torch", columns=["input_ids", "label"])

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
            # Transformer 用 DataCollator，BiLSTM 在 Trainer 内部处理
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer) if info["type"] == "transformer" else None,
        )

        trainer.train()

        # 保存模型
        if info["type"] == "transformer":
            model.merge_and_unload().save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)

        print(f"[+] {name} 训练完成并已保存至: {save_path}")


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

if __name__ == "__main__":
    # 请根据实际路径修改
    ds = prepare_dataset("data/diverse_vul.parquet")
    train_models(ds)