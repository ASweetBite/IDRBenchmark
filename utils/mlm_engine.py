import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

class MLMEngine:
    """专门负责处理基于上下文的代码掩码预测和特征提取"""
    def __init__(self, model_name="microsoft/codebert-base-mlm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 正在加载 MLM 基础工具模型 ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        tokens = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512,
                                padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**tokens)
            # 取 [CLS] token 的 embedding: (batch_size, hidden_size)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()