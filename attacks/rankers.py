import numpy as np
import heapq
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

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=3, top_k=5, filter_short_vars=True):
        """
        :param test_sample_size: 每个变量挑选多少个最相似的词去测试 (建议设为 3)
        :param top_k: 最终只返回排名前 K 的重要变量
        :param filter_short_vars: 是否过滤掉长度 <= 2 的变量 (如 i, j, x)
        """
        # 1. 获取原始概率
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)

        # 2. 启发式优化：过滤过短的、无语义的循环变量
        if filter_short_vars:
            valid_vars = [v for v in variables if len(v) > 2]
        else:
            valid_vars = variables

            # 记录所有需要批量推断的任务
        mutation_tasks = []  # 格式: [(var_name, renamed_code), ...]

        # 3.1 预先收集所有需要进行替换的 (变量, 候选词) 组合
        pending_renames = []
        for var in valid_vars:
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            var_emb = self._get_word_embedding(var)

            # 计算相似度 (这里的 NumPy 运算极快，不用放进线程池)
            candidate_sims = [
                (cand, self._cosine_similarity(var_emb, self._get_word_embedding(cand)))
                for cand in candidates
            ]

            # 获取 Top-N 最相似候选词
            top_candidates = heapq.nlargest(test_sample_size, candidate_sims, key=lambda x: x[1])

            for cand, _ in top_candidates:
                pending_renames.append((var, cand))

            # 3.2 使用多线程并发执行 rename_fn，彻底消除 AST 串行阻塞
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # 提交所有的 AST 替换任务
                future_to_task = {
                    executor.submit(self.rename_fn, code, {var: cand}): var
                    for var, cand in pending_renames
                }

                # 收集并发执行的结果
            for future in concurrent.futures.as_completed(future_to_task):
                var = future_to_task[future]
                try:
                    renamed_code = future.result()
                    mutation_tasks.append((var, renamed_code))
                except Exception as e:
                    # 捕获异常：如果发生作用域冲突或语法树错误，直接跳过这个变体
                    # print(f"    [RNNS Skip] Conflict generating code for var: {var}")
                    continue

            # 边界情况处理：如果没有任何可替换的词
        if not mutation_tasks:
            return [], {}

        # ================= 核心提速区 =================
        # 4. 提取所有代码进行 GPU 批量推断 (Global Batching)
        codes_to_predict = [task[1] for task in mutation_tasks]
        # ... 后续逻辑保持不变 ...

        # 调用 ModelZoo 提供的 batch_predict
        # batch_size 取决于你的显存大小，ModelZoo 默认是 32
        all_probs, _ = self.model_zoo.batch_predict(codes_to_predict, self.target_model)

        # 5. 整理推断结果，计算每个变量造成的最大概率下降
        var_max_drop = {var: -float('inf') for var in valid_vars}

        for (var, _), probs in zip(mutation_tasks, all_probs):
            new_prob = probs[reference_label]
            prob_drop = orig_prob - new_prob

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop

        # 过滤掉没有成功替换的变量 (值为负无穷的)
        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]

        # 6. 优化点B：使用 heapq 只获取全局下降最大的 Top-K 个变量，而不是全量排序
        top_k_vars_with_scores = heapq.nlargest(top_k, valid_scores, key=lambda x: x[1])

        ranked_vars = [var for var, score in top_k_vars_with_scores]
        score_dict = {var: score for var, score in top_k_vars_with_scores}

        return ranked_vars, score_dict