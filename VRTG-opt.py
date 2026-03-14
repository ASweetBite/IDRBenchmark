import random
import numpy as np
from typing import List, Dict, Callable, Tuple


# ==========================================
# 1. 目标模型接口定义
# ==========================================
class VulModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, code: str, true_label: int) -> Tuple[float, int]:
        """
        输入代码，返回预测结果。
        :param code: C/C++ 源码
        :param true_label: 真实标签 (例如 0: 安全, 1: 漏洞)
        :return: (真实标签的置信度概率, 当前预测的标签)
        """
        # TODO: 接入你自己的深度学习模型 (如基于HuggingFace的Pipeline)
        # 这里是伪代码示例：
        # inputs = self.tokenizer(code, return_tensors="pt")
        # logits = self.model(**inputs).logits
        # probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # pred_label = int(np.argmax(probs))
        # return probs[true_label], pred_label
        pass


# ==========================================
# 2. RNNS 敏感度排序器
# ==========================================
class RNNS_Ranker:
    def __init__(self, model_wrapper: VulModelWrapper, rename_fn: Callable):
        self.model_wrapper = model_wrapper
        self.rename_fn = rename_fn  # 你的 AST 重命名函数

    def rank_variables(self, code: str, variables: List[str], subs_pool: Dict[str, List[str]], true_label: int) -> List[
        str]:
        """
        使用 RNNS 策略对变量重要性进行排序。
        不同于 MASK，这里直接使用真实的替换词测试概率下降。
        """
        orig_prob, _ = self.model_wrapper.predict(code, true_label)
        scores = []

        for var in variables:
            candidates = subs_pool.get(var, [])
            if not candidates:
                scores.append((var, 0.0))
                continue

            # 随机选取1个（或多个取平均）候选词进行探针测试
            test_sub = random.choice(candidates)
            renamed_code = self.rename_fn(code, {var: test_sub})

            new_prob, _ = self.model_wrapper.predict(renamed_code, true_label)

            # 计算置信度下降幅度 (Probability Drop)
            prob_drop = orig_prob - new_prob
            scores.append((var, prob_drop))

        # 按照下降幅度降序排列 (Drop 越大，说明变量越关键)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [var for var, score in scores]


# ==========================================
# 3. 遗传算法优化器
# ==========================================
class GeneticAlgorithmOptimizer:
    def __init__(self, model_wrapper: VulModelWrapper, rename_fn: Callable,
                 pop_size=20, max_generations=15, mutation_rate=0.2):
        self.model_wrapper = model_wrapper
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate

    def generate_initial_population(self, target_vars: List[str], subs_pool: Dict[str, List[str]]) -> List[
        Dict[str, str]]:
        population = []
        # 保留一个全量原始变量的个体作为基线
        population.append({var: var for var in target_vars})

        for _ in range(self.pop_size - 1):
            individual = {}
            for var in target_vars:
                candidates = subs_pool.get(var, [var])
                # 包含原始变量作为候选，防止无意义突变
                choices = candidates + [var]
                individual[var] = random.choice(choices)
            population.append(individual)
        return population

    def evaluate_fitness(self, code: str, population: List[Dict[str, str]], true_label: int) -> List[
        Tuple[Dict, float, int]]:
        results = []
        for individual in population:
            # 过滤掉不需要修改的变量
            renaming_map = {k: v for k, v in individual.items() if k != v}
            if not renaming_map:
                renamed_code = code
            else:
                renamed_code = self.rename_fn(code, renaming_map)

            prob, pred_label = self.model_wrapper.predict(renamed_code, true_label)
            # 适应度函数：我们希望最小化真实标签的概率 -> 最大化 (1 - prob)
            fitness = 1.0 - prob
            results.append((individual, fitness, pred_label))
        return results

    def crossover(self, parent1: Dict[str, str], parent2: Dict[str, str]) -> Dict[str, str]:
        """均匀交叉 (Uniform Crossover)"""
        child = {}
        for var in parent1.keys():
            child[var] = parent1[var] if random.random() > 0.5 else parent2[var]
        return child

    def mutate(self, individual: Dict[str, str], subs_pool: Dict[str, List[str]]) -> Dict[str, str]:
        """变异操作 (Mutation)"""
        mutated = individual.copy()
        for var in mutated.keys():
            if random.random() < self.mutation_rate:
                candidates = subs_pool.get(var, [var])
                mutated[var] = random.choice(candidates + [var])
        return mutated

    def run(self, original_code: str, true_label: int, target_vars: List[str], subs_pool: Dict[str, List[str]]):
        population = self.generate_initial_population(target_vars, subs_pool)

        for gen in range(self.max_generations):
            evaluated = self.evaluate_fitness(original_code, population, true_label)

            # 检查是否有攻击成功的样本 (预测标签改变)
            for ind, fitness, pred_label in evaluated:
                if pred_label != true_label:
                    print(f"🔥 Attack Succeeded at Generation {gen}!")
                    best_map = {k: v for k, v in ind.items() if k != v}
                    return True, self.rename_fn(original_code, best_map)

            # 按适应度排序
            evaluated.sort(key=lambda x: x[1], reverse=True)

            # 锦标赛选择/轮盘赌 -> 这里简单取 Top 50% 作为精英
            elites = [x[0] for x in evaluated[: self.pop_size // 2]]

            new_population = elites.copy()
            # 繁殖下一代
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child, subs_pool)
                new_population.append(child)

            population = new_population

        print("❌ Attack Failed: Max generations reached.")
        # 返回概率被降得最低的代码
        best_ind = evaluated[0][0]
        best_map = {k: v for k, v in best_ind.items() if k != v}
        return False, self.rename_fn(original_code, best_map)


# ==========================================
# 4. IRTG 攻击调度器
# ==========================================
class IRTG_Attacker:
    def __init__(self, model_wrapper, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=5):
        self.model_wrapper = model_wrapper
        self.get_all_vars_fn = get_all_vars_fn  # AST 提取变量函数
        self.get_subs_pool_fn = get_subs_pool_fn  # AST 提取代码级替换词函数
        self.rename_fn = rename_fn
        self.top_k = top_k  # 选取前K个最重要的变量进行遗传算法，避免搜索空间爆炸

        self.rnns_ranker = RNNS_Ranker(model_wrapper, rename_fn)
        self.ga_optimizer = GeneticAlgorithmOptimizer(model_wrapper, rename_fn)

    def attack(self, code: str, true_label: int):
        # 1. 前置检查：如果原代码模型就预测错了，直接返回
        orig_prob, orig_pred = self.model_wrapper.predict(code, true_label)
        if orig_pred != true_label:
            return True, code

            # 2. 预处理：利用 AST 获取所有变量和替换池
        variables = self.get_all_vars_fn(code)
        subs_pool = self.get_subs_pool_fn(code, variables)

        if not variables:
            return False, code

        # 3. RNNS: 对变量基于重要度进行排序
        ranked_vars = self.rnns_ranker.rank_variables(code, variables, subs_pool, true_label)

        # 截取 Top-K 进行基因突变，减少 GA 计算量并提高收敛速度
        target_vars = ranked_vars[:self.top_k]

        # 4. GA: 遗传算法生成最佳对抗样本
        is_success, adv_code = self.ga_optimizer.run(code, true_label, target_vars, subs_pool)

        return is_success, adv_code