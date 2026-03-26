import numpy as np
import torch
from utils.model_zoo import ModelZoo


class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self._embedding_cache = {}

    def _get_word_embedding(self, word: str):
        if word in self._embedding_cache:
            return self._embedding_cache[word]

        emb = self.model_zoo.get_embedding(word)
        self._embedding_cache[word] = emb
        return emb

    def _cosine_similarity(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return -1.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def rank_variables(self, code, variables, subs_pool, reference_label, test_sample_size=8):
        # 1. 获取原始代码在目标标签下的初始概率
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)
        scores = []

        for var in variables:
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            # A. 获取原始变量的 embedding
            var_emb = self._get_word_embedding(var)

            # B. 计算所有候选词与原变量的相似度
            candidate_sims = []
            for cand in candidates:
                cand_emb = self._get_word_embedding(cand)
                sim = self._cosine_similarity(var_emb, cand_emb)
                candidate_sims.append((cand, sim))

            # C. 按照相似度从大到小排序，选择最近邻 (Nearest Neighbors)
            candidate_sims.sort(key=lambda x: x[1], reverse=True)

            # D. 挑选最相似的 Top-N 个候选词进入测试阶段
            test_subs = [item[0] for item in candidate_sims[:test_sample_size]]
            # ================== RNNS 核心逻辑结束 ==================

            max_prob_drop = -float('inf')

            # 只对筛选出的最相似的词进行攻击测试
            for test_sub in test_subs:
                renamed_code = self.rename_fn(code, {var: test_sub})
                # 预测替换后的概率
                new_prob = self.model_zoo.predict_label_conf(renamed_code, reference_label, self.target_model)
                prob_drop = orig_prob - new_prob

                if prob_drop > max_prob_drop:
                    max_prob_drop = prob_drop

            scores.append((var, max_prob_drop))

        # 按概率下降幅度从大到小排序（最重要的变量排在前面）
        scores.sort(key=lambda x: x[1], reverse=True)

        ranked_vars = [var for var, score in scores]
        score_dict = {var: score for var, score in scores}

        return ranked_vars, score_dict