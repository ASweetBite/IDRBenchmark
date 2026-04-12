import numpy as np
import heapq
import concurrent.futures
from utils.model_zoo import ModelZoo


class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        """Initializes the RNNS Ranker with a model zoo, target model, and renaming function."""
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self._embedding_cache = {}

    def _get_word_embedding(self, word: str):
        """Retrieves and caches the embedding vector for a given word."""
        if word in self._embedding_cache:
            return self._embedding_cache[word]
        emb = self.model_zoo.get_embedding(word)
        self._embedding_cache[word] = emb
        return emb

    def _cosine_similarity(self, vec1, vec2):
        """Calculates the cosine similarity between two embedding vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return -1.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=3, top_k=5, filter_short_vars=True):
        """Ranks variables based on their importance by evaluating the impact of renaming on model prediction confidence."""
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)

        if filter_short_vars:
            valid_vars = [v for v in variables if len(v) > 2]
        else:
            valid_vars = variables

        mutation_tasks = []

        pending_renames = []
        for var in valid_vars:
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            var_emb = self._get_word_embedding(var)

            candidate_sims = [
                (cand, self._cosine_similarity(var_emb, self._get_word_embedding(cand)))
                for cand in candidates
            ]

            top_candidates = heapq.nlargest(test_sample_size, candidate_sims, key=lambda x: x[1])

            for cand, _ in top_candidates:
                pending_renames.append((var, cand))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {
                executor.submit(self.rename_fn, code, {var: cand}): var
                for var, cand in pending_renames
            }

            for future in concurrent.futures.as_completed(future_to_task):
                var = future_to_task[future]
                try:
                    renamed_code = future.result()
                    mutation_tasks.append((var, renamed_code))
                except Exception as e:
                    continue

        if not mutation_tasks:
            return [], {}

        codes_to_predict = [task[1] for task in mutation_tasks]

        all_probs, _ = self.model_zoo.batch_predict(codes_to_predict, self.target_model)

        var_max_drop = {var: -float('inf') for var in valid_vars}

        for (var, _), probs in zip(mutation_tasks, all_probs):
            new_prob = probs[reference_label]
            prob_drop = orig_prob - new_prob

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop

        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]

        top_k_vars_with_scores = heapq.nlargest(top_k, valid_scores, key=lambda x: x[1])

        ranked_vars = [var for var, score in top_k_vars_with_scores]
        score_dict = {var: score for var, score in top_k_vars_with_scores}

        return ranked_vars, score_dict