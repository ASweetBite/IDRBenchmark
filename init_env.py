import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset

# 1. 强制镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 自定义模型包装类，解决架构不匹配问题
class CustomClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 获取序列的第一个 token (CLS) 的向量
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def prepare_data():
    print("[*] 准备真实漏洞数据集 (CodeXGLUE Devign)...")
    dataset = load_dataset("code_x_glue_cc_defect_detection", split="train")
    # 缩小数据量以便快速测试
    raw_data = dataset.shuffle(seed=42).select(range(500))
    return raw_data


def train_and_save_models(raw_dataset):
    model_map = {
        "CodeBERT": "microsoft/codebert-base",
        "GraphCodeBERT": "microsoft/graphcodebert-base",
        "UniXcoder": "microsoft/unixcoder-base"
    }

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] 使用设备: {device}")

    for name, hf_path in model_map.items():
        save_path = f"./models/{name.lower()}_finetuned"
        if os.path.exists(save_path):
            continue

        print(f"\n🚀 开始微调模型: {name}")

        tokenizer = AutoTokenizer.from_pretrained(hf_path)

        # 使用自定义包装类
        model = CustomClassificationModel(hf_path, num_labels=2)

        def tokenize_function(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = raw_dataset.map(tokenize_function, batched=True)
        # 注意：CodeXGLUE数据集原始列名是 'target'
        tokenized_ds = tokenized_ds.rename_column("target", "label")
        tokenized_ds = tokenized_ds.remove_columns(["func", "id", "project", "commit_id"])
        tokenized_ds.set_format("torch")

        training_args = TrainingArguments(
            output_dir=f"./temp_{name}",
            per_device_train_batch_size=4,
            num_train_epochs=1,  # 演示用，设为1
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        trainer.train()

        # 保存模型
        os.makedirs(save_path, exist_ok=True)
        model.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        # 保存分类头权重
        torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier.bin"))
        print(f"[+] {name} 微调完成")


if __name__ == "__main__":
    data = prepare_data()
    train_and_save_models(data)