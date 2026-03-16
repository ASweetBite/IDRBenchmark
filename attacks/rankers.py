import random
from utils.model_zoo import ModelZoo


class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label):
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)
        scores = []

        for var in variables:
            candidates = subs_pool.get(var, [])
            if not candidates: continue

            test_sub = random.choice(candidates)
            renamed_code = self.rename_fn(code, {var: test_sub})
            new_prob = self.model_zoo.predict_label_conf(renamed_code, reference_label, self.target_model)

            prob_drop = orig_prob - new_prob
            scores.append((var, prob_drop))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [var for var, score in scores]
