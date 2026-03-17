import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from collections import Counter

# 强制镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 自定义带权重的 Trainer
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 关键修改：将 labels 从输入中剥离，防止模型自动计算内部 loss
        labels = inputs.pop("labels")

        # 2. 现在 inputs 里没有 labels 了，模型只会返回 logits
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 3. 使用你的加权交叉熵损失
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

        # 4. 计算 loss (确保 logits 和 labels 维度对齐)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def prepare_data():
    print("[*] 正在从 HuggingFace 加载全量数据集...")
    # 使用训练集全量数据
    dataset = load_dataset("code_x_glue_cc_defect_detection", split="train")
    return dataset.shuffle(seed=42)


def train_and_save_models(raw_dataset):
    model_map = {
        "CodeBERT": "microsoft/codebert-base",
        "GraphCodeBERT": "microsoft/graphcodebert-base",
        "UniXcoder": "microsoft/unixcoder-base"
    }

    # 1. 自动计算类别权重 (解决不平衡问题)
    all_labels = [int(x['target']) for x in raw_dataset]
    label_counts = Counter(all_labels)
    total = len(all_labels)
    # 计算权重: 数量越少的类，权重越高 (这里简单用总数除以每类数量)
    weights = [total / (2 * count) for count in [label_counts[0], label_counts[1]]]
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[*] 类别分布: {label_counts}, 计算权重: {weights}")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    for name, hf_path in model_map.items():
        save_path = f"./models/{name.lower()}_finetuned"
        if os.path.exists(save_path): continue

        print(f"\n🚀 开始使用 LoRA 微调: {name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_path)

        # 不要使用 problem_type="single_label_classification"
        # 因为我们自己通过 WeightedTrainer 控制 loss 计算
        model = AutoModelForSequenceClassification.from_pretrained(hf_path, num_labels=2)
        model = get_peft_model(model, peft_config)

        def tokenize_function(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = raw_dataset.map(tokenize_function, batched=True)
        tokenized_ds = tokenized_ds.rename_column("target", "label")
        tokenized_ds = tokenized_ds.map(lambda x: {"label": int(x["label"])})
        tokenized_ds = tokenized_ds.remove_columns(["func", "id", "project", "commit_id"])
        tokenized_ds.set_format("torch")

        training_args = TrainingArguments(
            output_dir=f"./temp_{name}",
            per_device_train_batch_size=8,
            num_train_epochs=3,  # 增加到 3 个 Epoch
            learning_rate=5e-5,  # 稍微调高一点
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        trainer.train()
        model = model.merge_and_unload()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[+] {name} 已微调并保存")


if __name__ == "__main__":
    data = prepare_data()
    train_and_save_models(data)