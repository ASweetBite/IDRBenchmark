import random
import math
from utils.model_zoo import ModelZoo


class GeneticAlgorithmOptimizer:
    # 建议放大种群和迭代次数
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

        # 【优化点1】增强的变异选择器：强制尝试选择与当前不同的变量名
        def get_safe_choice(var, pool, current_val=None):
            choices = list(set(pool)) if pool else []
            if not choices:
                return var
            if current_val and len(choices) > 1 and current_val in choices:
                # 尽量不选当前已经在用的名字，增加探索性
                choices.remove(current_val)
            return random.choice(choices)

        fitness_cache = {}
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred
        stagnation_counter = 0

        # 初始化种群 (确保初始种群也是去重的，或者尽量多样)
        population = [{var: var for var in target_vars}]
        for _ in range(self.pop_size - 1):
            population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})

        # 核心遗传算法循环
        for gen in range(self.max_generations):
            evaluated = []
            codes_to_predict = []
            keys_to_predict = []

            # 【关键修复】在评估这一代之前，记录上一代的最佳适应度
            previous_best_fitness = best_fitness

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

                    orig_prob = max(probs[original_pred], 1e-9)

                    if len(probs) == 2:
                        target_label = 1 - original_pred
                        target_prob = max(probs[target_label], 1e-9)
                    else:
                        other_probs = [p for idx, p in enumerate(probs) if idx != original_pred]
                        target_prob = max(max(other_probs), 1e-9)

                    # Log-Odds 转换
                    fitness = math.log(target_prob) - math.log(orig_prob)
                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            # 3. 评估并检查攻击是否成功
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                if pred != original_pred:
                    print(f"    [SUCCESS] Gen {gen} | FOOLED! Probs: {[round(p, 4) for p in probs]}")
                    return True, mutated_code, probs, pred

                if fitness > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred
                    print(
                        f"    [DEBUG] Gen {gen} | Fit(LogOdds): {fitness:.4f} | Probs: {[round(p, 4) for p in probs]}")

            # 【优化点2】种群去重 (Deduplication) - 极其重要
            # 过滤掉完全相同的个体，防止近亲繁殖导致种群退化
            unique_evaluated = []
            seen_genes = set()
            for ind_tuple in evaluated:
                # 使用 frozenset 表示基因型以便哈希
                gene_signature = frozenset(ind_tuple[0].items())
                if gene_signature not in seen_genes:
                    seen_genes.add(gene_signature)
                    unique_evaluated.append(ind_tuple)

            # 4. 进化逻辑
            # 使用略微宽容的阈值判断是否停滞 (防浮点误差)
            if best_fitness <= previous_best_fitness + 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            unique_evaluated.sort(key=lambda x: x[1], reverse=True)

            # 停滞 5 代触发灾变（给了算法充分探索的时间，且不再误触）
            if stagnation_counter >= 5:
                # 灾变：保留绝对最优的 1 个个体，其余全部重新随机生成，引入新鲜血液
                best_elite = unique_evaluated[0][0]
                population = [best_elite]
                while len(population) < self.pop_size:
                    population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})
                stagnation_counter = 0
                continue

            # 正常进化：从去重后的精英中选择
            # 至少保留 2 个精英，如果去重后数量不够，就用最好的填补
            num_elites = max(2, min(len(unique_evaluated), self.pop_size // 4))
            elites = [x[0] for x in unique_evaluated[:num_elites]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                # 交叉
                if len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                else:
                    p1, p2 = elites[0], elites[0]

                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in target_vars}

                # 变异
                for v in child:
                    if random.random() < mutation_probs.get(v, 0.3):
                        # 传入 current_val，强制其尽量变异为另一个名字
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v], current_val=child[v])

                new_pop.append(child)

            population = new_pop

        return False, best_code, best_probs, best_pred