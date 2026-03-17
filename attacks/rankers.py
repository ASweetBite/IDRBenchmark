import random
from utils.model_zoo import ModelZoo


class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label, test_sample_size=3):
        """
        :param test_sample_size: 为每个变量抽取几个候选词进行测试。值越大越准确，但开销也越大。
        """
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)
        scores = []

        for var in variables:
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            # 随机挑选几个候选词进行测试，以防单个词带来的误差
            test_subs = random.sample(candidates, min(len(candidates), test_sample_size))

            # 记录导致概率下降最厉害的那一次替换
            max_prob_drop = -float('inf')

            for test_sub in test_subs:
                renamed_code = self.rename_fn(code, {var: test_sub})
                new_prob = self.model_zoo.predict_label_conf(renamed_code, reference_label, self.target_model)
                prob_drop = orig_prob - new_prob

                if prob_drop > max_prob_drop:
                    max_prob_drop = prob_drop

            # 将最大概率下降幅度作为该变量的显著性得分
            scores.append((var, max_prob_drop))

        # 按概率下降幅度从大到小排序 (下降越多，变量越脆弱/越重要)
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的变量名列表
        return [var for var, score in scores]