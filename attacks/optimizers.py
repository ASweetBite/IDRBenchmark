import random
import math
from utils.model_zoo import ModelZoo

class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, pop_size=40, iterations=60, run_mode="attack"):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = iterations
        self.run_mode = run_mode

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
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

        def get_safe_choice(var, pool, current_val=None):
            choices = list(set(pool)) if pool else []
            if not choices:
                return var
            if current_val and len(choices) > 1 and current_val in choices:
                choices.remove(current_val)
            return random.choice(choices)

        fitness_cache = {}
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred
        stagnation_counter = 0

        population = [{var: var for var in target_vars}]
        for _ in range(self.pop_size - 1):
            population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})

        for gen in range(self.max_generations):
            evaluated = []
            codes_to_predict = []
            keys_to_predict = []

            previous_best_fitness = best_fitness

            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key not in fitness_cache:
                    mutated_code = self.rename_fn(code, rename_map)
                    codes_to_predict.append(mutated_code)
                    keys_to_predict.append(cache_key)

            if codes_to_predict:
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
                for i in range(len(codes_to_predict)):
                    probs = batch_probs[i]
                    pred = batch_preds[i]

                    orig_prob = max(probs[original_pred], 1e-9)

                    if len(probs) == 2:
                        target_label = 1 - original_pred
                        target_prob = max(probs[target_label], 1e-9)
                    else:
                        other_probs = [p for idx, p in enumerate(probs) if idx != original_pred]
                        target_prob = max(max(other_probs), 1e-9)

                    fitness = math.log(target_prob) - math.log(orig_prob)
                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                # =================== [新增：基于 run_mode 的提前退出逻辑] ===================
                # 如果是攻击模式并且预测改变了，直接返回成功
                if pred != original_pred and self.run_mode == "attack":
                    # print(f"    [SUCCESS] Gen {gen} | FOOLED! Probs: {[round(p, 4) for p in probs]}")
                    return True, mutated_code, probs, pred
                # =========================================================================

                # 无论在哪种模式，都要实时记录最高适应度的样本
                if fitness > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred
                    # print(f"    [DEBUG] Gen {gen} | Fit(LogOdds): {fitness:.4f} | Probs: {[round(p, 4) for p in probs]}")

            unique_evaluated = []
            seen_genes = set()
            for ind_tuple in evaluated:
                gene_signature = frozenset(ind_tuple[0].items())
                if gene_signature not in seen_genes:
                    seen_genes.add(gene_signature)
                    unique_evaluated.append(ind_tuple)

            if best_fitness <= previous_best_fitness + 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            unique_evaluated.sort(key=lambda x: x[1], reverse=True)

            if stagnation_counter >= 5:
                best_elite = unique_evaluated[0][0]
                population = [best_elite]
                while len(population) < self.pop_size:
                    population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})
                stagnation_counter = 0
                continue

            num_elites = max(2, min(len(unique_evaluated), self.pop_size // 4))
            elites = [x[0] for x in unique_evaluated[:num_elites]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                if len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                else:
                    p1, p2 = elites[0], elites[0]

                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in target_vars}

                for v in child:
                    if random.random() < mutation_probs.get(v, 0.3):
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v], current_val=child[v])

                new_pop.append(child)

            population = new_pop

        return (best_pred != original_pred), best_code, best_probs, best_pred


import math
from utils.model_zoo import ModelZoo


class GreedyOptimizer:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, run_mode="attack"):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.run_mode = run_mode

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """
        贪心搜索算法实现
        :param code: 原始代码
        :param original_pred: 原始预测标签
        :param target_vars: 待攻击变量列表
        :param subs_pool: 候选词池 {var: [cand1, cand2, ...]}
        :param variable_scores: 变量显著度得分 {var: score}
        """
        # 1. 变量排序 (对应论文 4.2 节：按词显著度/显著度排序)
        # 贪心算法通常先修改对模型影响最大的变量
        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        current_code = code
        current_best_probs = None
        current_best_pred = original_pred

        # 记录整个过程中的最佳结果（用于非攻击模式或攻击失败时返回）
        overall_best_fitness = float('-inf')
        overall_best_code = code

        # 2. 逐个变量进行贪心搜索
        for var in sorted_vars:
            candidates = list(set(subs_pool.get(var, [])))
            if not candidates:
                continue

            # 为当前变量生成所有候选代码，准备批处理预测
            codes_to_predict = []
            for cand in candidates:
                if cand == var: continue
                # 仅替换当前这一个变量
                temp_code = self.rename_fn(current_code, {var: cand})
                codes_to_predict.append((cand, temp_code))

            if not codes_to_predict:
                continue

            # 3. 批处理预测 (提高效率)
            candidate_strings = [item[1] for item in codes_to_predict]
            batch_probs, batch_preds = self.model_zoo.batch_predict(candidate_strings, self.target_model)

            best_var_fitness = float('-inf')
            best_var_sub = None
            best_var_code = None
            best_var_probs = None
            best_var_pred = None

            for i in range(len(codes_to_predict)):
                cand_name = codes_to_predict[i][0]
                probs = batch_probs[i]
                pred = batch_preds[i]

                # 计算 Fitness (沿用你遗传算法中的 Log-Odds 逻辑)
                orig_prob = max(probs[original_pred], 1e-9)
                if len(probs) == 2:
                    target_prob = max(probs[1 - original_pred], 1e-9)
                else:
                    other_probs = [p for idx, p in enumerate(probs) if idx != original_pred]
                    target_prob = max(max(other_probs), 1e-9)

                fitness = math.log(target_prob) - math.log(orig_prob)

                # 寻找当前变量下的最优替换
                if fitness > best_var_fitness:
                    best_var_fitness = fitness
                    best_var_sub = cand_name
                    best_var_code = candidate_strings[i]
                    best_var_probs = probs
                    best_var_pred = pred

            # 4. 更新当前代码 (贪心选择)
            # 如果找到了能让模型更困惑的替换，则保留该修改
            if best_var_fitness > float('-inf'):
                current_code = best_var_code
                current_best_probs = best_var_probs
                current_best_pred = best_var_pred

                if best_var_fitness > overall_best_fitness:
                    overall_best_fitness = best_var_fitness
                    overall_best_code = best_var_code

                # =================== [提前退出逻辑] ===================
                if current_best_pred != original_pred and self.run_mode == "attack":
                    return True, current_code, current_best_probs, current_best_pred
                # =====================================================

        return (current_best_pred != original_pred), overall_best_code, current_best_probs, current_best_pred