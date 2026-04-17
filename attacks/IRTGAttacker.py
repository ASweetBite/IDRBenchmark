import json
import os
from typing import List, Dict

from attacks.optimizers import GeneticAlgorithmOptimizer, GreedyOptimizer, BeamSearchOptimizer, BayesianOptimizer
from attacks.rankers import RNNS_Ranker


class IRTGAttacker:
    def __init__(self, model_zoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, mode: str, config: dict):
        """Initializes the attacker with model environments, function pointers, and configuration parameters."""
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.config = config

        run_params = config.get('run_params', {})
        irtg_config = config.get('irtg_attacker', {})
        global_config = config.get('global', {})

        self.top_k = irtg_config.get('top_k', 5)
        self.iterations = run_params.get('iterations', 10)
        self.run_mode = run_params.get('run_mode', 'attack')

        # 强制转换为小写，支持 'greedy', 'beam', 'ga', 'bo' 四种算法
        self.optimizer_type = str(run_params.get('algorithm', 'greedy')).lower()
        if self.optimizer_type not in ["greedy", "beam", "ga", "bo"]:
            raise ValueError(
                f"Unsupported algorithm: {self.optimizer_type}. Must be one of ['greedy', 'beam', 'ga', 'bo'].")

        self.result_dir = global_config.get('result_dir', './results')

        self.attacker_params = {
            "get_all_vars_fn": get_all_vars_fn,
            "get_subs_pool_fn": get_subs_pool_fn,
            "rename_fn": rename_fn
        }

    def attack(self, dataset: List[Dict]):
        """Executes the attack pipeline across the dataset and calculates the cross-model transferability matrix."""
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}

        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.attacker_params["rename_fn"]) for m in self.model_names}

        # 1. 动态实例化所选的优化算法
        optimizers = {}
        for m in self.model_names:
            opt_kwargs = {
                "model_zoo": self.model_zoo,
                "target_model": m,
                "rename_fn": self.attacker_params["rename_fn"],
                "mode": self.mode,
                "config": self.config
            }
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

            variables = self.attacker_params["get_all_vars_fn"](code)

            subs_pool = self.attacker_params["get_subs_pool_fn"](code, variables)
            for var in list(subs_pool.keys()):
                if not subs_pool[var]:
                    del subs_pool[var]
                    if var in variables: variables.remove(var)

            if not variables:
                continue

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                print(
                    f"\n[Sample {idx + 1}] Target={atk_model} | Optimizer={self.optimizer_type.upper()} ({self.run_mode} mode)...")
                stats[atk_model][atk_model]["total"] += 1

                rnns_best_seed = None

                # 2. 区分全局搜索与启发式降维搜索
                if self.optimizer_type in ["greedy", "beam"]:
                    # Greedy 和 Beam 直接在全局变量空间搜索
                    target_vars = variables.copy()
                    target_scores = None
                else:
                    # GA 和 BO 依赖 RNNS 进行前置薄弱点探测
                    print("RNNS-Start...")
                    rnns_output = rankers[atk_model].rank_variables(
                        code=code, variables=variables.copy(), subs_pool=subs_pool, reference_label=orig_pred,
                        top_k=max(self.top_k, int(len(variables) * 0.3))
                    )

                    # 兼容处理：检查 RNNS 是否返回了用于 GA 优化的最佳种子字典
                    if len(rnns_output) == 3:
                        ranked_vars, all_scores, rnns_best_seed = rnns_output
                    else:
                        ranked_vars, all_scores = rnns_output

                    target_vars = ranked_vars
                    target_scores = {var: all_scores[var] for var in target_vars}

                print(f"{self.optimizer_type.upper()}-Start...")

                # 3. 动态组织运行参数
                run_kwargs = {
                    "code": code,
                    "original_pred": orig_pred,
                    "target_vars": target_vars,
                    "subs_pool": subs_pool,
                    "variable_scores": target_scores
                }

                # 如果是遗传算法且存在 RNNS 探测到的极佳词组，注入种子
                if self.optimizer_type == "ga" and rnns_best_seed:
                    run_kwargs["rnns_best_seed"] = rnns_best_seed

                # 执行攻击
                is_success, adv_code, adv_probs, adv_pred = optimizers[atk_model].run(**run_kwargs)

                # 后续记录及可迁移性测试逻辑保持不变
                if self.run_mode == "dataset":
                    storage_orig[atk_model].append({"func": code, "label": ground_truth})
                    storage_adv[atk_model].append({"func": adv_code, "label": ground_truth})
                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        print(f"  * [{self.optimizer_type.upper()}] ✅ Success")
                    else:
                        print(f"  * [{self.optimizer_type.upper()}] ❌ Failed")
                else:
                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        storage_adv[atk_model].append({
                            "original_code": code,
                            "adversarial_code": adv_code,
                            "label": ground_truth
                        })
                        print(f"  * [{self.optimizer_type.upper()}] ✅ Success | {orig_pred} -> {adv_pred}")
                    else:
                        print(f"  * [{self.optimizer_type.upper()}] ❌ Failed")

                if is_success:
                    for vic_model in self.model_names:
                        if vic_model == atk_model: continue
                        vic_orig_pred = orig_predictions[vic_model]["pred"]
                        if vic_orig_pred == ground_truth:
                            stats[atk_model][vic_model]["total"] += 1
                            _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                            if vic_adv_pred != vic_orig_pred:
                                stats[atk_model][vic_model]["fooled"] += 1
                                print(f"    - [Transfer] ✅ {vic_model} FOOLED")

        self.print_summary(stats)
        self.save_results(storage_orig, storage_adv)

        asr_matrix = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        return asr_matrix

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