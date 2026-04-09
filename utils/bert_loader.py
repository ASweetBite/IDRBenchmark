import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ==========================================
# 1. CodeBERT 双头包装器
# ==========================================
class CodeBERTDualHeadWrapper(nn.Module):
    def __init__(self, base_model, hidden_size, num_cwe_classes):
        super().__init__()
        self.base_model = base_model
        # 独立的双输出头
        self.detection_head = nn.Linear(hidden_size, 2)
        self.classification_head = nn.Linear(hidden_size, num_cwe_classes)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        # CodeBERT (基于 RoBERTa) 提取表征
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 获取 hidden_states
            **kwargs,
        )

        # CodeBERT 的序列首 token 是 <s>，对应索引 0，用作整段代码的聚合表征 (CLS token equivalent)
        cls_rep = out.last_hidden_state[:, 0, :]

        det_logits = self.detection_head(cls_rep)
        cls_logits = self.classification_head(cls_rep)

        return det_logits, cls_logits, out.hidden_states

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()


# ==========================================
# 2. CodeBERT 专用模型加载器 (去除了 LLM 特有优化)
# ==========================================
class CodeBERTModelLoader:
    def __init__(self, config):
        self.config = config
        # 通常这里传入 "microsoft/codebert-base" 或本地路径
        self.model_dir = config["model"]["model_name"]
        self.max_seq_len = config["model"]["max_seq_len"]

        # CodeBERT 最长支持 512 (RoBERTa 架构限制)
        if self.max_seq_len > 512:
            logger.warning(f"CodeBERT 原生最大序列长度为 512，已将 max_seq_len 从 {self.max_seq_len} 截断至 512。")
            self.max_seq_len = 512

        # RTX 4060 环境，直接挂载到单卡即可
        target_device = config["model"].get("device", "cuda:0")
        if torch.cuda.is_available():
            self.device = target_device if "cuda" in target_device else "cuda:0"
        else:
            self.device = "cpu"

        self.num_classes = config["data"].get("num_classes", 16)

    def load_model(self) -> Tuple["CodeBERTModel", AutoTokenizer]:
        # 1. 加载 Tokenizer (CodeBERT 基于 RoBERTa tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # 2. 加载基座模型
        # RTX 4060 显存充足，直接使用默认 FP32 精度即可，不需要 device_map="auto"
        logger.info(f"加载 CodeBERT 基座模型，路径: {self.model_dir}")
        base_model = AutoModel.from_pretrained(self.model_dir)

        # 3. 包装为双头结构
        hidden_size = base_model.config.hidden_size
        model = CodeBERTDualHeadWrapper(base_model, hidden_size, self.num_classes)

        # 4. 加载双头权重 (如果存在)
        head_path = os.path.join(self.model_dir, "dual_heads.pt")
        if os.path.exists(head_path):
            logger.info(f"加载已训练的双头分类器权重: {head_path}")
            head_state = torch.load(head_path, map_location="cpu")
            model.detection_head.load_state_dict(head_state["detection_head"])
            model.classification_head.load_state_dict(head_state["classification_head"])
            model.eval()
        else:
            logger.warning("未检测到 dual_heads.pt，使用随机初始化头权重准备训练。")
            model.train()

        # 5. 整体模型移至显卡 (CodeBERT 体积小，一次性 to(device) 即可)
        model.to(self.device)

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"CodeBERT 加载完毕，当前显存占用: {alloc:.2f} MB")

        return CodeBERTModel(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            device=self.device,
        ), tokenizer


# ==========================================
# 3. 封装推理接口 (接口行为与之前完全一致)
# ==========================================
class CodeBERTModel:
    def __init__(self, model, tokenizer, max_seq_len, device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    @torch.no_grad()
    def predict(self, code: str) -> Dict[str, Union[float, List[float]]]:
        self.model.eval()
        inputs = self.tokenizer(
            code, truncation=True, padding="max_length",
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)

        det_logits, cls_logits, _ = self.model(**inputs)

        det_probs = torch.softmax(det_logits, dim=1).float().cpu().numpy()[0]
        cls_probs = torch.softmax(cls_logits, dim=1).float().cpu().numpy()[0]

        return {
            "f_det": float(det_probs[1]),  # 1类代表漏洞
            "f_cls": cls_probs.tolist(),   # 多分类的具体 CWE 概率
        }

    @torch.no_grad()
    def batch_predict(self, code_list: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        # CodeBERT 比较轻量，batch_size 默认值可以调大一点 (例如 16 或 32)
        self.model.eval()
        all_det_probs = []
        all_cls_probs = []

        for i in range(0, len(code_list), batch_size):
            batch = code_list[i: i + batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=self.max_seq_len, return_tensors="pt",
            ).to(self.device)

            det_logits, cls_logits, _ = self.model(**inputs)

            det_probs = torch.softmax(det_logits, dim=1).float().cpu().numpy()
            cls_probs = torch.softmax(cls_logits, dim=1).float().cpu().numpy()

            all_det_probs.append(det_probs)
            all_cls_probs.append(cls_probs)

        if not all_det_probs:
            return {"f_det": np.array([]), "f_cls": np.array([])}

        return {
            "f_det": np.vstack(all_det_probs)[:, 1],
            "f_cls": np.vstack(all_cls_probs),
        }

    @torch.no_grad()
    def encode(self, code: str):
        """返回隐藏状态，用于对比学习分析或获取特征"""
        self.model.eval()
        inputs = self.tokenizer(
            code, truncation=True, padding="max_length",
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)

        _, _, hidden_states = self.model(**inputs)
        return hidden_states