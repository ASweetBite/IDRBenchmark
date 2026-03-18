import re
import torch
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier
from utils.model_zoo import ModelZoo


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo: ModelZoo, analyzer: IdentifierAnalyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer
        self.embedding_cache = {}

    def _get_embedding_with_cache(self, code: str):
        if code not in self.embedding_cache:
            embedding = self.model_zoo.get_embedding(code).reshape(1, -1)
            self.embedding_cache[code] = embedding
        return self.embedding_cache[code]

    def clear_cache(self):
        self.embedding_cache.clear()

    def generate_candidates(self, code: str, var_name: str, top_k_mlm=50, top_n_keep=10) -> List[str]:
        self.clear_cache()

        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        # 目标变量不存在，直接返回空
        if var_name not in identifiers:
            return []

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
            _, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)
            candidates = [
                self.model_zoo.mlm_tokenizer.decode([idx], clean_up_tokenization_spaces=True).strip()
                for idx in top_k_indices
            ]

        orig_embedding = self._get_embedding_with_cache(code)
        valid_candidates = []

        for cand in set(candidates):
            # 1. 基础合法性过滤
            if not is_valid_identifier(cand):
                continue
            if cand == var_name:
                continue

            # 2. 作用域感知过滤（核心修改）
            # 如果 cand 已存在，但与 var_name 生命周期/作用域不重叠，则允许
            if not self.analyzer.can_rename_to(code_bytes, var_name, cand):
                continue

            # 3. 生成新代码
            new_code = re.sub(r'\b' + re.escape(var_name) + r'\b', cand, code)

            try:
                new_embedding = self._get_embedding_with_cache(new_code)
                sim = cosine_similarity(orig_embedding, new_embedding)[0][0]
                valid_candidates.append((cand, sim))
            except Exception:
                continue

        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in valid_candidates[:top_n_keep]]