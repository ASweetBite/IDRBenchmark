from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from typing import List

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier
from utils.model_zoo import ModelZoo


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo: ModelZoo, analyzer: IdentifierAnalyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer
        # --- 新增缓存字典 ---
        self.embedding_cache = {}

    def _get_embedding_with_cache(self, code: str):
        """带有缓存的 Embedding 获取方法"""
        if code not in self.embedding_cache:
            # 只有没计算过的代码才进行模型推理
            embedding = self.model_zoo.get_embedding(code).reshape(1, -1)
            self.embedding_cache[code] = embedding
        return self.embedding_cache[code]

    def clear_cache(self):
        """调用完一个样本后，手动清理缓存以防止内存溢出"""
        self.embedding_cache.clear()

    def generate_candidates(self, code: str, var_name: str, top_k_mlm=50, top_n_keep=10) -> List[str]:
        # 每次生成新候选词时，清理一下缓存，防止不同代码样本间缓存污染（除非代码非常相似）
        self.clear_cache()

        mask_token = self.model_zoo.mlm_tokenizer.mask_token
        masked_code = re.sub(r'\b' + re.escape(var_name) + r'\b', mask_token, code)

        mask_str_idx = masked_code.find(mask_token)
        if mask_str_idx == -1:
            return []

        start_idx = max(0, mask_str_idx - 1000)
        end_idx = min(len(masked_code), mask_str_idx + 1000)
        mlm_context = masked_code[start_idx:end_idx]

        inputs = self.model_zoo.mlm_tokenizer(
            mlm_context, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model_zoo.device)

        mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
        mask_token_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_token_indices) == 0:
            return []

        mask_idx = mask_token_indices[0]

        with torch.no_grad():
            logits = self.model_zoo.mlm_model(**inputs).logits
            mask_logits = logits[0, mask_idx, :]
            top_k_weights, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)
            candidates = [
                self.model_zoo.mlm_tokenizer.decode([idx], clean_up_tokenization_spaces=True).strip()
                for idx in top_k_indices
            ]

        # --- 使用缓存的 Embedding ---
        orig_embedding = self._get_embedding_with_cache(code)
        valid_candidates = []

        for cand in set(candidates):
            if not is_valid_identifier(cand) or cand == var_name:
                continue

            new_code = masked_code.replace(mask_token, cand)

            try:
                # --- 这里会极大地加速 ---
                new_embedding = self._get_embedding_with_cache(new_code)
                sim = cosine_similarity(orig_embedding, new_embedding)[0][0]
                valid_candidates.append((cand, sim))
            except Exception as e:
                continue

        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in valid_candidates[:top_n_keep]]