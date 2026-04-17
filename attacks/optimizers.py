import math
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        ga_cfg = config.get('genetic_algorithm', {})
        run_cfg = config.get('run_params', {})

        self.pop_size = ga_cfg.get('pop_size', 40)
        self.max_generations = run_cfg.get('iterations', 60)
        self.run_mode = run_cfg.get('run_mode', 'attack')

        self.stagnation_limit = ga_cfg.get('stagnation_threshold', 5)
        self.m_rate_min = ga_cfg.get('mutation_rate_min', 0.1)
        self.m_rate_max = ga_cfg.get('mutation_rate_max', 0.5)

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """Executes a genetic algorithm to find the optimal adversarial variable substitutions."""
        mutation_probs = {}
        if variable_scores and target_vars:
            scores = [variable_scores.get(v, 0) for v in target_vars]
            min_s, max_s = min(scores), max(scores)
            for var in target_vars:
                score = variable_scores.get(var, 0)
                if max_s > min_s:
                    mutation_probs[var] = self.m_rate_min + (self.m_rate_max - self.m_rate_min) * (
                                (score - min_s) / (max_s - min_s))
                else:
                    mutation_probs[var] = (self.m_rate_min + self.m_rate_max) / 2
        else:
            mutation_probs = {v: 0.3 for v in target_vars}

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
                    try:
                        mutated_code = self.rename_fn(code, rename_map)
                        codes_to_predict.append(mutated_code)
                        keys_to_predict.append(cache_key)
                    except Exception:
                        fitness_cache[cache_key] = (float('-inf'), original_pred, code, None)

            if codes_to_predict:
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
                for i in range(len(codes_to_predict)):
                    probs = batch_probs[i]
                    pred = batch_preds[i]

                    orig_prob = max(probs[original_pred], 1e-9)

                    if self.mode == "binary":
                        target_label = 1 - original_pred
                        target_prob = max(probs[target_label], 1e-9)
                        fitness = math.log(target_prob) - math.log(orig_prob)
                    else:
                        fitness = -orig_prob

                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                if pred != original_pred and self.run_mode == "attack":
                    return True, mutated_code, probs, pred

                if fitness > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred

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

            if stagnation_counter >= self.stagnation_limit:
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


class GreedyOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """Executes a sequential greedy search to apply variable substitutions and bypass model defenses."""
        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        current_code = code
        current_best_probs = None
        current_best_pred = original_pred
        overall_best_fitness = float('-inf')
        overall_best_code = code

        for var in sorted_vars:
            candidates = list(set(subs_pool.get(var, [])))
            if not candidates:
                continue

            codes_to_predict = []
            for cand in candidates:
                if cand == var:
                    continue
                try:
                    temp_code = self.rename_fn(current_code, {var: cand})
                    if temp_code:
                        codes_to_predict.append((cand, temp_code))
                except Exception:
                    continue

            if not codes_to_predict:
                continue

            candidate_strings = [item[1] for item in codes_to_predict]

            batch_probs, batch_preds = self.model_zoo.batch_predict(candidate_strings, self.target_model)

            best_var_fitness = float('-inf')
            best_var_code = None
            best_var_probs = None
            best_var_pred = None

            for i in range(len(codes_to_predict)):
                probs = batch_probs[i]
                pred = batch_preds[i]

                orig_idx = 0 if original_pred == -1 else original_pred

                if orig_idx >= len(probs):
                    orig_idx = len(probs) - 1

                orig_prob = max(probs[orig_idx], 1e-9)

                if self.mode == "binary":
                    target_idx = 1 if original_pred == -1 else 0

                    if target_idx >= len(probs):
                        target_idx = len(probs) - 1

                    target_prob = max(probs[target_idx], 1e-9)

                    fitness = math.log(target_prob) - math.log(orig_prob)
                else:
                    fitness = -orig_prob

                if fitness > best_var_fitness:
                    best_var_fitness = fitness
                    best_var_code = candidate_strings[i]
                    best_var_probs = probs
                    best_var_pred = pred

            if best_var_code and best_var_fitness > float('-inf'):
                current_code = best_var_code
                current_best_probs = best_var_probs
                current_best_pred = best_var_pred

                if best_var_fitness > overall_best_fitness:
                    overall_best_fitness = best_var_fitness
                    overall_best_code = best_var_code

                if current_best_pred != original_pred and self.run_mode == "attack":
                    verify_probs, verify_pred = self.model_zoo.predict(current_code, self.target_model)

                    if verify_pred != original_pred:
                        return True, current_code, verify_probs, verify_pred
                    else:
                        current_best_pred = verify_pred

        final_probs, final_pred = self.model_zoo.predict(overall_best_code, self.target_model)
        is_success = (final_pred != original_pred)

        return is_success, overall_best_code, final_probs, final_pred


class BeamSearchOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')
        # 新增：Beam Search 的束宽参数，默认设为 3，你可以在 config 中调整
        self.beam_size = run_cfg.get('beam_size', 3)

    def _calculate_fitness(self, probs, original_pred):
        """辅助函数：计算适应度得分"""
        orig_idx = 0 if original_pred == -1 else original_pred
        orig_idx = min(orig_idx, len(probs) - 1)
        orig_prob = max(probs[orig_idx], 1e-9)

        if self.mode == "binary":
            target_idx = 1 if original_pred == -1 else 0
            target_idx = min(target_idx, len(probs) - 1)
            target_prob = max(probs[target_idx], 1e-9)
            return math.log(target_prob) - math.log(orig_prob)
        else:
            return -orig_prob

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """Executes a Beam Search to apply variable substitutions and bypass model defenses."""
        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        # 获取初始状态的 probs
        orig_probs, orig_pred = self.model_zoo.predict(code, self.target_model)
        initial_fitness = self._calculate_fitness(orig_probs, original_pred)

        # Beam 初始化：存储元组 (fitness, code, probs, pred)
        beam = [(initial_fitness, code, orig_probs, orig_pred)]

        overall_best_fitness = initial_fitness
        overall_best_code = code

        for var in sorted_vars:
            candidates = list(set(subs_pool.get(var, [])))
            if not candidates:
                continue

            new_beam_candidates = []
            seen_codes = set()  # 用于去重，防止重复预测相同代码

            for curr_fitness, curr_code, curr_probs, curr_pred in beam:
                # 1. 保留当前状态（即不替换该变量），防止强制替换导致性能退化
                if curr_code not in seen_codes:
                    new_beam_candidates.append((curr_fitness, curr_code, curr_probs, curr_pred))
                    seen_codes.add(curr_code)

                # 2. 生成所有替换候选项
                codes_to_predict = []
                for cand in candidates:
                    if cand == var:
                        continue
                    try:
                        temp_code = self.rename_fn(curr_code, {var: cand})
                        if temp_code and temp_code not in seen_codes:
                            codes_to_predict.append(temp_code)
                            seen_codes.add(temp_code)
                    except Exception:
                        continue

                if not codes_to_predict:
                    continue

                # 3. 批量预测
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)

                # 4. 计算并记录候选结果
                for i, temp_code in enumerate(codes_to_predict):
                    probs = batch_probs[i]
                    pred = batch_preds[i]
                    fitness = self._calculate_fitness(probs, original_pred)

                    # 记录全局最优解（防止 Beam 最后收敛到次优）
                    if fitness > overall_best_fitness:
                        overall_best_fitness = fitness
                        overall_best_code = temp_code

                    # 提前终止：发现攻击成功的对抗样本
                    if pred != original_pred and self.run_mode == "attack":
                        verify_probs, verify_pred = self.model_zoo.predict(temp_code, self.target_model)
                        if verify_pred != original_pred:
                            return True, temp_code, verify_probs, verify_pred

                    new_beam_candidates.append((fitness, temp_code, probs, pred))

            # 5. 排序并截断，保留 Top-K (Beam Size) 个最优状态进入下一轮
            new_beam_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam_candidates[:self.beam_size]

        # 遍历完所有变量后，验证全局最优代码是否成功
        final_probs, final_pred = self.model_zoo.predict(overall_best_code, self.target_model)
        is_success = (final_pred != original_pred)

        return is_success, overall_best_code, final_probs, final_pred



class BayesianOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        bo_cfg = config.get('bayesian', {}) if config else {}

        self.run_mode = run_cfg.get('run_mode', 'attack')
        self.max_iters = run_cfg.get('iterations', 50)

        # BO 专属参数
        self.init_samples = bo_cfg.get('init_samples', 10)  # 初始随机探索的样本数
        self.acq_samples = bo_cfg.get('acq_samples', 200)  # 每次用代理模型评估的候选状态数
        self.kappa = bo_cfg.get('kappa', 1.5)  # UCB 的探索-利用权衡系数 (越大越倾向于探索)

    def _calculate_fitness(self, probs, original_pred):
        orig_idx = min(0 if original_pred == -1 else original_pred, len(probs) - 1)
        orig_prob = max(probs[orig_idx], 1e-9)

        if self.mode == "binary":
            target_idx = min(1 if original_pred == -1 else 0, len(probs) - 1)
            target_prob = max(probs[target_idx], 1e-9)
            return math.log(target_prob) - math.log(orig_prob)
        else:
            return -orig_prob

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """使用贝叶斯优化在离散空间中寻找最优替换组合"""

        # 1. 构建离散搜索空间映射 (State Representation)
        # 过滤掉没有候选词的变量，减少搜索维度
        valid_vars = [v for v in target_vars if subs_pool.get(v)]
        if not valid_vars:
            return False, code, None, original_pred

        var_candidates = {}
        categories_for_encoder = []
        for var in valid_vars:
            # 将原始词也作为候选之一（索引为0）
            cands = [var] + [c for c in set(subs_pool[var]) if c != var]
            var_candidates[var] = cands
            categories_for_encoder.append(np.arange(len(cands)))

        num_vars = len(valid_vars)

        # 使用 One-Hot 编码器，让代理模型（随机森林）更好地理解离散类别特征
        encoder = OneHotEncoder(categories=categories_for_encoder, sparse_output=False)
        # Fit 一次伪数据以初始化 encoder
        dummy_data = np.zeros((1, num_vars), dtype=int)
        encoder.fit(dummy_data)

        # 辅助函数：将状态索引转换为代码
        def state_to_code(state_indices):
            rename_map = {}
            for i, var in enumerate(valid_vars):
                chosen_cand = var_candidates[var][state_indices[i]]
                if chosen_cand != var:
                    rename_map[var] = chosen_cand
            if not rename_map:
                return code
            try:
                return self.rename_fn(code, rename_map)
            except Exception:
                return None

        # 记录历史数据
        X_history = []  # 存储状态索引
        Y_history = []  # 存储适应度
        seen_states = set()

        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred

        # 2. 初始化阶段 (Initial Sampling)
        # 先用随机采样填充初始知识库
        initial_states = [np.zeros(num_vars, dtype=int)]  # 先把全不替换的原状态加进去
        for _ in range(self.init_samples - 1):
            state = [random.randint(0, len(var_candidates[var]) - 1) for var in valid_vars]
            initial_states.append(np.array(state))

        codes_to_predict = []
        valid_initial_states = []
        for state in initial_states:
            state_tuple = tuple(state)
            if state_tuple in seen_states:
                continue
            seen_states.add(state_tuple)

            mutated_code = state_to_code(state)
            if mutated_code:
                codes_to_predict.append(mutated_code)
                valid_initial_states.append(state)

        if codes_to_predict:
            batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
            for i in range(len(codes_to_predict)):
                fit = self._calculate_fitness(batch_probs[i], original_pred)
                X_history.append(valid_initial_states[i])
                Y_history.append(fit)

                if batch_preds[i] != original_pred and self.run_mode == "attack":
                    return True, codes_to_predict[i], batch_probs[i], batch_preds[i]

                if fit > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fit, codes_to_predict[i], batch_probs[i], \
                    batch_preds[i]

        # 3. 贝叶斯优化主循环
        for iteration in range(self.max_iters - self.init_samples):
            if not X_history:
                break

            # a. 训练代理模型 (Surrogate Model)
            # 将离散索引转换为 One-Hot 向量
            X_encoded = encoder.transform(X_history)

            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(X_encoded, Y_history)

            # b. 使用代理模型在内部探索 (Acquisition Phase)
            # 生成一批候选状态 (在最优解附近变异 + 纯随机探索)
            candidate_states = []
            best_historical_state = X_history[np.argmax(Y_history)]

            for _ in range(self.acq_samples):
                if random.random() < 0.7:
                    # 70% 概率：从当前最优状态进行微小变异（利用）
                    new_state = best_historical_state.copy()
                    # 随机改变 1 到 3 个变量
                    num_mutations = random.randint(1, min(3, num_vars))
                    mutate_indices = random.sample(range(num_vars), num_mutations)
                    for idx in mutate_indices:
                        new_state[idx] = random.randint(0, len(var_candidates[valid_vars[idx]]) - 1)
                    candidate_states.append(new_state)
                else:
                    # 30% 概率：纯随机生成（探索）
                    candidate_states.append([random.randint(0, len(var_candidates[var]) - 1) for var in valid_vars])

            # 过滤掉已经评估过的状态
            unseen_candidates = [s for s in candidate_states if tuple(s) not in seen_states]

            if not unseen_candidates:
                # 如果找不到新状态，跳过当前迭代（或者可以增加变异强度）
                continue

            unseen_candidates_encoded = encoder.transform(unseen_candidates)

            # c. 计算 UCB (Upper Confidence Bound) 采集函数
            # 从森林中的每一棵树获取预测值，以计算均值和方差
            tree_predictions = []
            for tree in rf.estimators_:
                # sklearn 的 decision tree 期望的数据类型
                tree_predictions.append(tree.predict(unseen_candidates_encoded))

            tree_predictions = np.array(tree_predictions)
            mean_preds = np.mean(tree_predictions, axis=0)
            std_preds = np.std(tree_predictions, axis=0)

            # UCB 公式：均值越高越好（有潜力），方差越高越好（模型没见过，有探索价值）
            ucb_scores = mean_preds + self.kappa * std_preds

            # d. 选出 UCB 得分最高的一个状态进行真实模型的评估
            best_candidate_idx = np.argmax(ucb_scores)
            chosen_state = unseen_candidates[best_candidate_idx]

            mutated_code = state_to_code(chosen_state)
            if not mutated_code:
                seen_states.add(tuple(chosen_state))
                continue

            # e. 在目标模型上进行真实查询
            probs, pred = self.model_zoo.predict(mutated_code, self.target_model)
            fitness = self._calculate_fitness(probs, original_pred)

            # f. 更新历史记录与最优解
            seen_states.add(tuple(chosen_state))
            X_history.append(chosen_state)
            Y_history.append(fitness)

            if pred != original_pred and self.run_mode == "attack":
                return True, mutated_code, probs, pred

            if fitness > best_fitness:
                best_fitness = fitness
                best_code = mutated_code
                best_probs = probs
                best_pred = pred

        return (best_pred != original_pred), best_code, best_probs, best_pred