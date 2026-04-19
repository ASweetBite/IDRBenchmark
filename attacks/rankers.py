import gc
import torch
import heapq


class RNNS_Ranker:
    def __init__(self, model_zoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=10, top_k=10, filter_short_vars=True):

        oref_idx = 0 if reference_label == -1 else reference_label
        orig_prob = self.model_zoo.predict_label_conf(code, oref_idx, self.target_model)

        valid_vars = [v for v in variables if len(v) > 2] if filter_short_vars else variables
        mutation_tasks = []

        # 串行执行重命名，彻底规避 AST Parser 多线程崩溃风险
        for var in valid_vars:
            candidates = subs_pool.get(var, [])[:test_sample_size]
            for cand in candidates:
                if cand != var:
                    try:
                        renamed_code = self.rename_fn(code, {var: cand})
                        if renamed_code:
                            mutation_tasks.append((var, cand, renamed_code))
                    except Exception:
                        continue

        if not mutation_tasks:
            return [], {}, {}

        codes_to_predict = [task[2] for task in mutation_tasks]
        all_probs = []
        BATCH_SIZE = 16

        # 分块推理，彻底杜绝 OOM
        for i in range(0, len(codes_to_predict), BATCH_SIZE):
            chunk = codes_to_predict[i:i + BATCH_SIZE]
            chunk_probs, _ = self.model_zoo.batch_predict(chunk, self.target_model)
            all_probs.extend(chunk_probs)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        var_max_drop = {var: -float('inf') for var in valid_vars}
        var_best_cand = {var: var for var in valid_vars}

        for (var, cand, _), probs in zip(mutation_tasks, all_probs):
            prob_drop = orig_prob - probs[oref_idx]

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop
                var_best_cand[var] = cand

        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]
        top_k_vars_with_scores = heapq.nlargest(top_k, valid_scores, key=lambda x: x[1])

        ranked_vars = [var for var, _ in top_k_vars_with_scores]
        score_dict = {var: score for var, score in top_k_vars_with_scores}
        best_seeds = {var: var_best_cand[var] for var in ranked_vars if var_best_cand[var] != var}

        return ranked_vars, score_dict, best_seeds