"""
轻量级候选词生成器
使用预训练的 FastText 词向量 + FAISS 索引实现 O(1) 语义近邻检索。
"""

import json
import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# 延迟导入 (环境中可能没有)
_FAISS_AVAILABLE = False
_FT_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss 未安装，候选生成将退化为随机替换")

try:
    import fasttext
    _FT_AVAILABLE = True
except ImportError:
    logger.warning("fasttext 未安装，候选生成将退化为随机替换")

from utils.ast_tools import IdentifierAnalyzer


class LightweightCandidateGenerator:
    """
    S(v_i) = FAISS.top_M( E_ft(v_i) )
    """

    def __init__(self, config):
        self.config = config
        cand_cfg = config.get("candidate", {})
        self.top_m = cand_cfg.get("top_m", 10)

        self.ft_model = None
        self.faiss_index = None
        self.vocab = []
        self.analyzer = IdentifierAnalyzer()

        # 加载 FastText
        ft_path = cand_cfg.get("fasttext_model_path", "")
        if _FT_AVAILABLE and ft_path:
            try:
                self.ft_model = fasttext.load_model(ft_path)
                logger.info(f"FastText 模型加载成功: {ft_path}")
            except Exception as e:
                logger.warning(f"FastText 加载失败: {e}")

        # 加载 FAISS 索引 + 词表
        faiss_path = cand_cfg.get("faiss_index_path", "")
        vocab_path = cand_cfg.get("faiss_vocab_path", "")
        if _FAISS_AVAILABLE and faiss_path:
            try:
                self.faiss_index = faiss.read_index(faiss_path)
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
                logger.info(f"FAISS 索引加载成功: {faiss_path}, 词表大小: {len(self.vocab)}")
            except Exception as e:
                logger.warning(f"FAISS 加载失败: {e}")

        # 退化用的随机变量名池
        self._fallback_pool = [
            f"var_{i}" for i in range(200)
        ] + [
            f"tmp_{i}" for i in range(50)
        ] + [
            f"arg_{i}" for i in range(50)
        ]

    def _get_faiss_neighbors(self, word: str) -> List[str]:
        """通过 FAISS 查找语义最近的 M 个候选替换词"""
        if self.ft_model is None or self.faiss_index is None:
            return []

        vec = self.ft_model.get_word_vector(word).astype(np.float32).reshape(1, -1)
        # L2 归一化 (余弦相似度)
        faiss.normalize_L2(vec)
        _, indices = self.faiss_index.search(vec, self.top_m + 1)

        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(self.vocab):
                cand = self.vocab[idx]
                if cand != word:
                    candidates.append(cand)
        return candidates[: self.top_m]

    def _fallback_candidates(self, word: str) -> List[str]:
        """退化: 随机生成候选词"""
        pool = [w for w in self._fallback_pool if w != word]
        return random.sample(pool, min(self.top_m, len(pool)))

    def generate_candidates(self, code: str) -> Dict[str, List[str]]:
        """
        给定一段代码，返回每个标识符的候选替换词集合。
        返回: { identifier: [candidate_1, ..., candidate_M], ... }
        """
        code_bytes = code.encode("utf-8") if isinstance(code, str) else code
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        result = {}
        for ident in identifiers:
            neighbors = self._get_faiss_neighbors(ident)
            if not neighbors:
                neighbors = self._fallback_candidates(ident)
            result[ident] = neighbors

        return result

    def get_random_replacement(self, code: str, target_vars: List[str]) -> Dict[str, str]:
        """
        为指定的变量列表各随机选一个替换词。
        用于随机平滑(Randomized Smoothing)和安全样本增强。
        """
        candidates = self.generate_candidates(code)
        mapping = {}
        for var in target_vars:
            cands = candidates.get(var, self._fallback_candidates(var))
            if cands:
                mapping[var] = random.choice(cands)
        return mapping
