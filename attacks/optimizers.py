import random
from utils.model_zoo import ModelZoo


class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, pop_size=10, max_generations=5):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = max_generations

    def run(self, code, original_pred, target_vars, subs_pool, existing_vars):
        def get_safe_choice(var, pool):
            # 过滤掉冲突的变量，保留候选词
            choices = [c for c in pool if c not in existing_vars or c == var]
            return random.choice(choices) if choices else var

        # --- 缓存优化：使用 dict 的 item 序列化作为 Key，比代码字符串快得多 ---
        # Key: frozenset([(var1, cand1), (var2, cand2), ...])
        fitness_cache = {}

        def get_fitness_and_pred(rename_map):
            # 将 rename_map 转换为可哈希的 tuple，用于缓存
            cache_key = frozenset(rename_map.items())
            if cache_key in fitness_cache:
                return fitness_cache[cache_key]

            # 生成代码并预测
            mutated_code = self.rename_fn(code, rename_map)
            probs, pred = self.model_zoo.predict(mutated_code, self.target_model)

            # 计算适应度：我们想要原始预测的概率越低越好
            # 适应度 = 1.0 - (原始预测类别的置信度)
            # 这样 fitness 越高，说明攻击效果越好
            fitness = 1.0 - probs[original_pred]

            fitness_cache[cache_key] = (fitness, pred, mutated_code, probs)
            return fitness_cache[cache_key]

        # 初始化种群
        population = []
        # 1. 初始个体：全部不变（基准线）
        population.append({var: var for var in target_vars})
        # 2. 随机个体
        for _ in range(self.pop_size - 1):
            population.append({v: get_safe_choice(v, subs_pool.get(v, [v]) + [v]) for v in target_vars})

        best_code = code
        best_fitness = -1
        best_probs = None
        best_pred = original_pred

        for gen in range(self.max_generations):
            evaluated = []

            # 评估种群
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                fitness, pred, mutated_code, probs = get_fitness_and_pred(rename_map)

                evaluated.append((ind, fitness, pred, mutated_code, probs))

                # 如果达到目标（预测改变了），立即返回
                if pred != original_pred:
                    print(f"    [+] Attack Succeeded at Gen {gen}!")
                    return True, mutated_code, probs, pred

                # 记录最优（以防提前结束）
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_code = mutated_code
                    best_probs = probs
                    best_pred = pred

            # 选择精英 (按适应度降序)
            evaluated.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in evaluated[:max(2, self.pop_size // 4)]]

            # 生成下一代
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                # 交叉
                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in target_vars}
                # 变异
                for v in child:
                    if random.random() < 0.2:  # 稍微调低变异率，保持收敛性
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                new_pop.append(child)
            population = new_pop

        return False, best_code, best_probs, best_pred