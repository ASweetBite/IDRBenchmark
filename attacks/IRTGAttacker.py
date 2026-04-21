import gc
import json
import math
import os

import numpy as np
import torch

from attacks.optimizers import GeneticAlgorithmOptimizer, GreedyOptimizer, BeamSearchOptimizer, BayesianOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZooQueryTracker


class IRTGAttacker:
    def __init__(self, model_zoo, get_all_vars_fn, mlm_gen, llm_gen, rename_fn, mode: str, config: dict):
        self.model_zoo = ModelZooQueryTracker(model_zoo)

        self.model_names = self.model_zoo.model_names
        self.mode = mode
        self.config = config

        _global = config.get('global', {})
        run_params = config.get('run_params', {})
        irtg_config = config.get('irtg_attacker', {})
        hw_config = config.get('heavyweight_candidate', {})

        self.result_dir = _global.get('result_dir',"./results")
        self.top_k = irtg_config.get('top_k', 5)
        self.iterations = run_params.get('iterations', 10)
        self.run_mode = run_params.get('run_mode', 'attack')

        self.total_quota = hw_config.get('top_n_keep', 50)
        self.llm_target_quota = max(1, int(self.total_quota * 0.4))

        self.optimizer_type = str(run_params.get('algorithm', 'greedy')).lower()
        if self.optimizer_type not in ["greedy", "beam", "ga", "bo"]:
            raise ValueError(f"Unsupported algorithm: {self.optimizer_type}.")

        self.get_all_vars_fn = get_all_vars_fn
        self.mlm_gen = mlm_gen
        self.llm_gen = llm_gen
        self.rename_fn = rename_fn

    def _merge_candidate_pools(self, mlm_pool: dict, llm_pool: dict, final_quota: int = 20) -> dict:
        """
        严格按照优先级合并候选词池：
        1. LLM 生成的高质量词汇享有绝对优先权 (排在最前面)。
        2. MLM 生成的词汇作为 Padding (填充物)，用于补齐 LLM 数量不足的缺口。
        3. 强制去重并截断至 final_quota。
        """
        final_pool = {}

        # 收集所有至少被 MLM 或 LLM 处理过的变量名
        all_vars = set(mlm_pool.keys()).union(set(llm_pool.keys()))

        for var in all_vars:
            llm_cands = llm_pool.get(var, [])
            mlm_cands = mlm_pool.get(var, [])

            # Step 1: LLM 候选词作为第一梯队，率先入池
            merged_cands = list(llm_cands)

            # Step 2: 如果 LLM 数量不达标，用 MLM 候选词进行兜底填充
            if len(merged_cands) < final_quota:
                for cand in mlm_cands:
                    if cand not in merged_cands:  # 严格去重：防止 MLM 生成了和 LLM 一样的词
                        merged_cands.append(cand)
                        # 一旦填满所需的配额，立刻停止填充
                        if len(merged_cands) >= final_quota:
                            break

            # Step 3: 严格截断，防止总数溢出 (比如 LLM 自身就生成了超过 final_quota 的词)
            final_pool[var] = merged_cands[:final_quota]

        return final_pool

    from typing import List, Dict

    def attack(self, dataset: List[Dict]):
        stats = {atk: {vic: {"total": 0, "fooled": 0, "success_queries": []} for vic in self.model_names} for atk in
                 self.model_names}
        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        # 初始化 Ranker (现在它可以利用 MLM 真实扰动)
        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.rename_fn) for m in self.model_names}
        # 初始化 Optimizer
        optimizers = {}
        for m in self.model_names:
            opt_kwargs = {"model_zoo": self.model_zoo, "target_model": m, "rename_fn": self.rename_fn,
                          "mode": self.mode, "config": self.config}
            if self.optimizer_type == "greedy":
                optimizers[m] = GreedyOptimizer(**opt_kwargs)
            elif self.optimizer_type == "beam":
                optimizers[m] = BeamSearchOptimizer(**opt_kwargs)
            elif self.optimizer_type == "ga":
                optimizers[m] = GeneticAlgorithmOptimizer(**opt_kwargs)
            elif self.optimizer_type == "bo":
                optimizers[m] = BayesianOptimizer(**opt_kwargs)

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            variables = self.get_all_vars_fn(code)
            if not variables: continue

            # =====================================================================
            # 优化点 1: 引入 AST 折叠逻辑，大幅缩减无效上下文
            # =====================================================================
            code_bytes = code.encode("utf-8")
            analyzer = self.mlm_gen.analyzer  # 直接复用生成器中的 analyzer
            identifiers = analyzer.extract_identifiers(code_bytes)

            batch_tasks = []

            # print(f"\n[Sample {idx + 1}] Target Pool Size: {len(variables)} identifiers")
            # print("=" * 50)
            # print("🔍 AST FOLDED CODE SLICES (VARIABLES ONLY)")
            # print("=" * 50)

            for var in variables:
                # 如果 AST 没有提取到该标识符（极为罕见的 Parser 偏差），直接用全量代码兜底
                if var not in identifiers:
                    batch_tasks.append({"target_name": var, "code_str": code})
                    continue

                # 【核心逻辑】：判断当前标识符是否是纯函数名或类名
                # 如果它所有的使用场景都是 function/method/class，那它就不是普通变量
                is_callable_or_class = all(
                    occ.get("entity_type") in ["function", "method", "class"] for occ in identifiers[var])

                if is_callable_or_class:
                    # 【修改点】：不对函数名做切片，直接传入完整代码作为上下文
                    target_code_str = code
                    # print(f"\n--- ⚡ Skipped Slicing for Function/Class: '{var}' ---")
                else:
                    try:
                        # 仅对真正的普通变量执行折叠切片
                        target_code_str = analyzer.get_folded_code(code_bytes, var)
                        # print(f"\n--- ✂️ Slice for Target Variable: '{var}' ---")
                        # print(target_code_str)
                    except Exception as e:
                        target_code_str = code  # 降级兜底
                        # print(f"\n--- ⚠️ Slicing Failed for '{var}', using full code ---")

                batch_tasks.append({"target_name": var, "code_str": target_code_str})
                # print("-" * 50)

            # print(f"\n[Sample {idx + 1}] Target Pool Size: {len(variables)} variables")

            # =====================================================================
            # 优化点 2: 引入分块生成 (Chunking) 彻底解决 MLM OOM 问题
            # =====================================================================
            print(f" -> Running MLM Lightweight Generator (Target: {self.total_quota} cands/var)...")
            mlm_subs_pool = {}
            MAX_BATCH_SIZE = 4
            num_chunks = math.ceil(len(batch_tasks) / MAX_BATCH_SIZE)

            for i in range(0, len(batch_tasks), MAX_BATCH_SIZE):
                chunk = batch_tasks[i:i + MAX_BATCH_SIZE]
                try:
                    # [关键点] MLM 直接瞄准 total_quota (例如50) 进行满额生成，作为保底底座
                    chunk_pool = self.mlm_gen.generate_candidates(
                        chunk,
                        top_k_mlm=max(40, self.total_quota),  # 确保搜索空间足够大
                        top_n_keep=self.total_quota
                    )
                    mlm_subs_pool.update(chunk_pool)
                finally:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 剔除未成功生成 MLM 候选词的变量
            variables = [v for v in variables if mlm_subs_pool.get(v)]
            if not variables: continue

            # =====================================================================
            # [新增架构] 优化点 2.5: LLM 全局语义探针注入 (Proxy Semantic Probe)
            # =====================================================================
            print(f" -> Running LLM Semantic Probe (Target: 2 cands/var for ALL vars)...")
            probe_quota = 2
            try:
                # 针对所有有效变量，利用 1.5B 本地模型生成极少量的探针词汇
                llm_probe_pool = self.llm_gen.generate_candidates(
                    batch_tasks,
                    target_quota=probe_quota
                )
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 构建带有探针的混合池，专门用于破除 RNNS 的观测偏差
            rnns_eval_pool = self._merge_candidate_pools(
                mlm_pool=mlm_subs_pool,
                llm_pool=llm_probe_pool,
                final_quota=self.total_quota
            )

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]
                print(f"[{atk_model}] Optimizer={self.optimizer_type.upper()} ({self.run_mode} mode)")
                stats[atk_model][atk_model]["total"] += 1

                rnns_best_seed = None

                # 2. 对于所有的算法，都用包含探针的混合池进行靶向定位
                self.model_zoo.reset_counter()

                print(" -> Running RNNS Saliency Analysis...")
                top_k = max(self.top_k, int(len(variables) * 0.5))
                # 这里的查询会被 Tracker 自动记录
                rnns_output = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(),
                    subs_pool=rnns_eval_pool, # [修改点] 传入注入了探针的池子
                    reference_label=orig_pred, top_k = top_k
                )

                if len(rnns_output) == 3:
                    ranked_vars, all_scores, rnns_best_seed = rnns_output
                else:
                    ranked_vars, all_scores = rnns_output

                    # 1. 决定【优化器】能看见的搜索空间 (保持你的 Beam Search 无硬截断逻辑)
                if self.optimizer_type in ["greedy", "beam"]:
                    target_vars = ranked_vars
                else:
                    target_vars = ranked_vars[:top_k]

                target_scores = {var: all_scores[var] for var in target_vars}

                # =====================================================================
                # 优化点 3: 算力护城河 —— 严格只为 Top-K 脆弱变量补齐 LLM 候选词
                # =====================================================================
                remaining_quota = self.llm_target_quota

                # 【核心修复】：无论什么优化算法，需要重度调用 LLM 的变量永远被死死限制在 Top-K！
                llm_enrichment_vars = ranked_vars[:top_k]

                print(
                    f" -> LLM Generating remaining candidates for Top-{len(llm_enrichment_vars)} vars (Target: {remaining_quota} cands/var)...")

                # 依据 llm_enrichment_vars 过滤任务，而不是 target_vars
                vulnerable_tasks = [t for t in batch_tasks if t["target_name"] in llm_enrichment_vars]

                try:
                    # 只有真正脆弱的 Top-K 变量，才会享受这昂贵的算力
                    llm_main_pool = self.llm_gen.generate_candidates(
                        vulnerable_tasks,
                        target_quota=remaining_quota
                    )
                finally:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                # 合并探针词汇与新生成的补齐词汇，构成最终的 LLM 弹药库
                final_llm_pool = {}
                for var in variables:
                    # 安全合并：如果 var 不在 Top-K 里，llm_main_pool.get(var) 会是空列表
                    # 但它依然能保留最初轻量级生成的 2 个探针词汇
                    final_llm_pool[var] = list(set(llm_probe_pool.get(var, []) + llm_main_pool.get(var, [])))

                # =====================================================================
                # Phase 4: 弹药库融合 (优先级 Padding)
                # =====================================================================
                # [关键点] 把总配额传给合并函数
                final_subs_pool = self._merge_candidate_pools(
                    mlm_pool=mlm_subs_pool,
                    llm_pool=final_llm_pool, # [修改点] 传入合并后的完整 LLM 词汇池
                    final_quota=self.total_quota
                )

                print(" -> Attack execution started...")
                run_kwargs = {
                    "code": code, "original_pred": orig_pred,
                    "target_vars": target_vars, "subs_pool": final_subs_pool,
                    "variable_scores": target_scores
                }
                if self.optimizer_type == "ga":
                    if rnns_best_seed:
                        run_kwargs["rnns_best_seed"] = rnns_best_seed
                    run_kwargs["all_vars"] = ranked_vars
                    run_kwargs["variable_scores"] = all_scores
                if self.optimizer_type == "bo":
                    run_kwargs["rnns_best_seed"] = rnns_best_seed

                # 优化器的查询同样会被 Tracker 自动记录
                is_success, adv_code, adv_probs, adv_pred = optimizers[atk_model].run(**run_kwargs)

                queries_consumed = self.model_zoo.get_query_count()

                if self.run_mode == "dataset":
                    storage_orig[atk_model].append({"func": code, "label": ground_truth})
                    storage_adv[atk_model].append({"func": adv_code, "label": ground_truth})
                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        stats[atk_model][atk_model]["success_queries"].append(queries_consumed)
                        print(f"    ✅ Success | Queries: {queries_consumed}")
                    else:
                        print(f"    ❌ Failed | Queries: {queries_consumed}")
                else:
                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        stats[atk_model][atk_model]["success_queries"].append(queries_consumed)
                        storage_adv[atk_model].append(
                            {"original_code": code, "adversarial_code": adv_code, "label": ground_truth})
                        print(f"    ✅ Success | {orig_pred} -> {adv_pred} | Queries: {queries_consumed}")
                    else:
                        print(f"    ❌ Failed | Queries: {queries_consumed}")

                if is_success:
                    for vic_model in self.model_names:
                        if vic_model == atk_model: continue
                        vic_orig_pred = orig_predictions[vic_model]["pred"]
                        if vic_orig_pred == ground_truth:
                            stats[atk_model][vic_model]["total"] += 1
                            _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                            if vic_adv_pred != vic_orig_pred:
                                stats[atk_model][vic_model]["fooled"] += 1
                                print(f"      [Transfer] ✅ {vic_model} FOOLED")

        # self.print_summary(stats)
        print("\n" + "=" * 50)
        print("🎯 FINAL ATTACK SUMMARY")
        print("=" * 50)
        asr_matrix = {}
        avg_queries = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}

            # 计算白盒/自身攻击的 AvgQ (学术界通用指标)
            success_queries = stats[atk_m][atk_m]["success_queries"]
            avg_q = round(np.mean(success_queries), 2) if success_queries else 0.0
            avg_queries[atk_m] = avg_q

            total_atk = stats[atk_m][atk_m]["total"]
            fooled_atk = stats[atk_m][atk_m]["fooled"]
            asr_atk = (fooled_atk / total_atk * 100) if total_atk > 0 else 0.0

            print(f"🛡️ Target Model: {atk_m.upper()}")
            print(f"   ► ASR (Attack Success Rate) : {asr_atk:.2f}% ({fooled_atk}/{total_atk})")
            print(f"   ► Avg. Queries (Success)    : {avg_q}")
            print("-" * 50)

            # 计算可迁移性 (Transferability)
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        self.save_results(storage_orig, storage_adv)

        return asr_matrix, avg_queries

    def save_results(self, storage_orig, storage_adv):
        """Saves original and adversarial samples to JSON files based on the configured result directory."""
        result_dir = self.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for model in self.model_names:
            if self.run_mode == "dataset":
                if storage_orig[model]:
                    orig_filename = f"orig_dataset_{model}_{self.mode}.json"
                    orig_path = os.path.join(result_dir, orig_filename)
                    self._write_json(orig_path, storage_orig[model])

                if storage_adv[model]:
                    adv_filename = f"adv_dataset_{model}_{self.mode}.json"
                    adv_path = os.path.join(result_dir, adv_filename)
                    self._write_json(adv_path, storage_adv[model])
            else:
                if storage_adv[model]:
                    filename = f"adv_test_set_{model}_{self.mode}.json"
                    file_path = os.path.join(result_dir, filename)
                    self._write_json(file_path, storage_adv[model])

    def _write_json(self, filename, data):
        """Handles the standard JSON serialization and file writing process for sample results."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Saved {len(data)} samples to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")

    def print_summary(self, stats):
        """Prints a formatted matrix displaying the Attack Success Rate (ASR) across all target and victim models."""
        print("\n" + "=" * 90)
        print("📊 FINAL CROSS-MODEL TRANSFERABILITY MATRIX (ASR %)")
        print("=" * 90)
        header = f"{'Attacker \\ Victim':<20} |"
        for m in self.model_names:
            header += f" {m:<13} |"
        print(header)
        print("-" * len(header))
        for atk_m in self.model_names:
            row = f"{atk_m:<20} |"
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                row += f" {asr:>11.2f}% |"
            print(row)
        print("=" * 90 + "\n")