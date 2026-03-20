import random
from utils.model_zoo import ModelZoo


class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    # 【修改1】将 test_sample_size 从 3 提升到 8 (或 5)
    def rank_variables(self, code, variables, subs_pool, reference_label, test_sample_size=8):
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)
        scores =[]

        for var in variables:
            candidates = subs_pool.get(var,[])
            if not candidates:
                continue

            # 挑选更多候选词进行测试，以防遗漏致命的替换
            test_subs = random.sample(candidates, min(len(candidates), test_sample_size))

            max_prob_drop = -float('inf')

            for test_sub in test_subs:
                renamed_code = self.rename_fn(code, {var: test_sub})
                new_prob = self.model_zoo.predict_label_conf(renamed_code, reference_label, self.target_model)
                prob_drop = orig_prob - new_prob

                if prob_drop > max_prob_drop:
                    max_prob_drop = prob_drop

            scores.append((var, max_prob_drop))

        # 按概率下降幅度从大到小排序
        scores.sort(key=lambda x: x[1], reverse=True)

        ranked_vars = [var for var, score in scores]
        score_dict = {var: score for var, score in scores}

        return ranked_vars, score_dict