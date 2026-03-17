import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# 强制镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def evaluate_models():
    device = get_device()
    print(f"[*] 使用设备: {device}")

    # 1. 加载测试集 (CodeXGLUE Devign)
    print("[*] 加载测试数据集...")
    dataset = load_dataset("code_x_glue_cc_defect_detection", split="test")
    # 为了速度，可以先只测 1000 条
    dataset = dataset.select(range(min(1000, len(dataset))))

    model_names = ["codebert", "graphcodebert", "unixcoder"]

    for name in model_names:
        model_path = f"./models/{name}_finetuned"
        if not os.path.exists(model_path):
            print(f"[!] 模型 {name} 不存在，跳过...")
            continue

        print(f"\n>>> 正在评估模型: {name}")

        # 2. 加载模型和 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        # 3. 数据预处理
        def tokenize(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512, padding="max_length")

        tokenized_ds = dataset.map(tokenize, batched=True)
        tokenized_ds = tokenized_ds.rename_column("target", "label")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        dataloader = DataLoader(tokenized_ds, batch_size=16)

        # 4. 推理
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 5. 计算指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Results for {name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(
            f"  Predictions distribution: {sum(all_preds)} positive (1) vs {len(all_preds) - sum(all_preds)} negative (0)")
        print("-" * 30)
        print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    evaluate_models()