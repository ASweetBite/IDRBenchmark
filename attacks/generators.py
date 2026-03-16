from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from typing import List, Dict, Tuple

from utils.ast_tools import IdentifierAnalyzer
from utils.model_zoo import ModelZoo


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo: ModelZoo, analyzer: IdentifierAnalyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer

    def generate_candidates(self, code: str, var_name: str, top_k_mlm=50, top_n_keep=10) -> List[str]:
        # 1. 构造 Mask 后的代码
        # 注意：这里需要替换所有出现的 var_name，使用正则表达式
        masked_code = re.sub(r'\b' + re.escape(var_name) + r'\b', '<mask>', code)

        # 2. 用 MLM 预测候选词
        inputs = self.model_zoo.mlm_tokenizer(masked_code, return_tensors="pt", truncation=True, max_length=512).to(
            self.model_zoo.device)
        mask_token_index = (inputs.input_ids == self.model_zoo.mlm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        with torch.no_grad():
            logits = self.model_zoo.mlm_model(**inputs).logits
            mask_logits = logits[0, mask_token_index, :]
            top_k_weights, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)
            candidates = [self.model_zoo.mlm_tokenizer.decode([idx]) for idx in top_k_indices[0]]

        # 3. 过滤并计算相似度 (Code-based Embedding)
        orig_embedding = self.model_zoo.get_embedding(code).reshape(1, -1)
        valid_candidates = []

        for cand in candidates:
            cand = cand.strip()
            if not is_valid_identifier(cand) or cand == var_name: continue

            # 替换产生新代码
            new_code = re.sub(r'\b<mask>\b', cand, masked_code)
            new_embedding = self.model_zoo.get_embedding(new_code).reshape(1, -1)

            # 计算余弦相似度
            sim = cosine_similarity(orig_embedding, new_embedding)[0][0]
            valid_candidates.append((cand, sim))

        # 4. 根据相似度排序，保留前 Top N
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in valid_candidates[:top_n_keep]]
