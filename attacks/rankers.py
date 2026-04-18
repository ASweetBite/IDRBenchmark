import concurrent.futures
import heapq


class RNNS_Ranker:
    def __init__(self, model_zoo, target_model: str, rename_fn):
        """
        极简版 Ranker：不再依赖 Embedding，专注于通过黑盒查询计算变量的脆弱性得分。
        """
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=10, top_k=10, filter_short_vars=True):

        oref_idx = 0 if reference_label == -1 else reference_label

        # 使用修正后的索引去获取概率
        orig_prob = self.model_zoo.predict_label_conf(code, oref_idx, self.target_model)

        if filter_short_vars:
            valid_vars = [v for v in variables if len(v) > 2]
        else:
            valid_vars = variables

        mutation_tasks = []
        pending_renames = []

        for var in valid_vars:
            # 💡 注意：直接获取列表，不要用 set() 打乱上游已经排好的高质量顺序！
            # 假设你的 subs_pool 里已经去重且按相似度降序排列
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            # 🌟 核心优化：因为候选词已经极其优质，直接取前 test_sample_size 个即可
            # 不需要再算任何 Cosine Similarity
            top_candidates = candidates[:test_sample_size]

            for cand in top_candidates:
                if cand != var:  # 确保不是自己替换自己
                    pending_renames.append((var, cand))

        # --- 下面的多线程测算逻辑保持不变 ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {
                executor.submit(self.rename_fn, code, {var: cand}): (var, cand)
                for var, cand in pending_renames
            }

            for future in concurrent.futures.as_completed(future_to_task):
                var, cand = future_to_task[future]
                try:
                    renamed_code = future.result()
                    if renamed_code:
                        mutation_tasks.append((var, cand, renamed_code))
                except Exception:
                    continue

        if not mutation_tasks:
            return [], {}, {}

        codes_to_predict = [task[2] for task in mutation_tasks]
        all_probs, _ = self.model_zoo.batch_predict(codes_to_predict, self.target_model)

        var_max_drop = {var: -float('inf') for var in valid_vars}
        var_best_cand = {var: var for var in valid_vars}

        for (var, cand, _), probs in zip(mutation_tasks, all_probs):
            # 🌟 修复点：这里也必须使用修正后的物理索引
            new_prob = probs[oref_idx]

            prob_drop = orig_prob - new_prob

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop
                var_best_cand[var] = cand

        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]

        # 取出导致置信度下降最多的 top_k 个变量
        top_k_vars_with_scores = heapq.nlargest(top_k, valid_scores, key=lambda x: x[1])

        ranked_vars = [var for var, score in top_k_vars_with_scores]
        score_dict = {var: score for var, score in top_k_vars_with_scores}

        # 只返回 ranked_vars 对应的最佳种子词
        best_seeds = {var: var_best_cand[var] for var in ranked_vars if var_best_cand[var] != var}

        # 返回：高危变量列表，重要性得分，以及为 GA 准备的最佳单步种子
        return ranked_vars, score_dict, best_seeds