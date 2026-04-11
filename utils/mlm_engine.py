import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

class MLMEngine:
    """Handles context-based code mask prediction and feature extraction using MLM."""

    def __init__(self, model_name="microsoft/codebert-base-mlm"):
        """Initializes the tokenizer and model on the available hardware device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading MLM base tool model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        """Generates a high-dimensional embedding for a code snippet using the CLS token."""
        tokens = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512,
                                padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**tokens)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()