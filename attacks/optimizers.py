import random
import math
from utils.model_zoo import ModelZoo


class GeneticAlgorithmOptimizer:
    # 建议将默认参数放大：pop_size=30, iterations=40
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, pop_size=40, iterations=60):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = iterations

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        # 1. 预计算变异概率权重
        mutation_probs = {}
        if variable_scores and target_vars:
            scores = [variable_scores.get(v, 0) for v in target_vars]
            min_s, max_s = min(scores), max(scores)
            for var in target_vars:
                score = variable_scores.get(var, 0)
                if max_s > min_s:
                    mutation_probs[var] = 0.1 + 0.4 * ((score - min_s) / (max_s - min_s))
                else:
                    mutation_probs[var] = 0.3
        else:
            mutation_probs = {v: 0.2 for v in target_vars}

        def get_safe_choice(var, pool):
            choices = list(set(pool)) if pool else []
            return random.choice(choices) if choices else var

        fitness_cache = {}
        # 【修改点1】将 best_fitness 初始化为极小值，因为基于 Log-Odds 的适应度可能是负数
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred
        stagnation_counter = 0

        # 初始化种群
        population = [{var: var for var in target_vars}]
        for _ in range(self.pop_size - 1):
            population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})

        # 核心遗传算法循环
        for gen in range(self.max_generations):
            evaluated = []
            codes_to_predict = []
            keys_to_predict = []

            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key not in fitness_cache:
                    mutated_code = self.rename_fn(code, rename_map)
                    codes_to_predict.append(mutated_code)
                    keys_to_predict.append(cache_key)

            # 2. 批量预测
            if codes_to_predict:
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
                for i in range(len(codes_to_predict)):
                    probs = batch_probs[i]
                    pred = batch_preds[i]

                    # 【核心修改点2：Log-Odds 适应度函数】
                    # 使用 1e-9 防止 math.log(0) 崩溃
                    orig_prob = max(probs[original_pred], 1e-9)

                    if len(probs) == 2:
                        target_label = 1 - original_pred
                        target_prob = max(probs[target_label], 1e-9)
                    else:
                        # 多分类：选取非原标签中概率最大的
                        other_probs = [p for idx, p in enumerate(probs) if idx != original_pred]
                        target_prob = max(max(other_probs), 1e-9)

                    # Log-Odds 转换：相当于 Logit_target - Logit_original
                    # 即便原概率是 0.9999 变成 0.9990，Fitness 也会有非常明显的上升梯度！
                    fitness = math.log(target_prob) - math.log(orig_prob)

                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            # 3. 评估并检查攻击是否成功
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                # 这里直接读取步骤 2 计算好的、包含最新梯度的 fitness
                fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                if pred != original_pred:
                    print(f"    [SUCCESS] Gen {gen} | FOOLED! Probs: {[round(p, 4) for p in probs]}")
                    return True, mutated_code, probs, pred

                if fitness > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred
                    # 打印 LogOdds 适应度，你会看到即使概率都是 [0.001, 0.999]，Fitness 也会呈阶梯状稳定增长
                    print(
                        f"    [DEBUG] Gen {gen} | Fit(LogOdds): {fitness:.4f} | Probs: {[round(p, 4) for p in probs]}")

            # 4. 进化逻辑
            current_gen_max_fitness = max([x[1] for x in evaluated])
            # 注意这里直接比较浮点数
            if current_gen_max_fitness <= best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # 停滞3代触发灾变（大幅引入新基因）
            if stagnation_counter >= 3:
                evaluated.sort(key=lambda x: x[1], reverse=True)
                elites = [x[0] for x in evaluated[:self.pop_size // 4]]
                population = elites + [{v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars}
                                       for _ in range(self.pop_size - len(elites))]
                stagnation_counter = 0
                continue

            evaluated.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in evaluated[:max(2, self.pop_size // 4)]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in target_vars}
                for v in child:
                    # 提高变异率底线，增加探索能力
                    if random.random() < mutation_probs.get(v, 0.3):
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                new_pop.append(child)
            population = new_pop

        return False, best_code, best_probs, best_pred