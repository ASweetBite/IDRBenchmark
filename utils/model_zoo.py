import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM


class ModelZoo:
    def __init__(self, model_configs: Dict[str, str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_names = list(model_configs.keys())

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                self.models[name] = None
                continue

            try:
                # 兼容 UniXcoder 等模型需要 trust_remote_code
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(self.device)
                model.eval()
                self.models[name] = {"tokenizer": tokenizer, "model": model}
            except Exception as e:
                print(f"[!] Error loading {name}: {e}")
                self.models[name] = None
        self.mlm_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.mlm_model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device)
        self.mlm_model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        # 这里以 CodeBERT 的 [CLS] token 输出作为整个代码的 embedding
        tokens = self.mlm_tokenizer(code, return_tensors="pt", truncation=True, max_length=512,
                                    padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.mlm_model.base_model(**tokens)
            # 取 [CLS] token 的 embedding: (batch_size, hidden_size)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        m = self.models.get(target_model)
        if m is None:
            # 防御性逻辑：如果模型加载失败，返回随机预测
            return [0.5, 0.5], 0

        inputs = m["tokenizer"](
            code, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = m["model"](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
            pred_label = int(np.argmax(probs))
        return probs, pred_label

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        return probs[label]

