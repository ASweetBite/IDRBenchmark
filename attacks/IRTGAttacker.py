from typing import List, Dict
import json
import os
from attacks.optimizers import GeneticAlgorithmOptimizer, GreedyOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZoo


class IRTGAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, mode: str, config: dict):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.config = config

        # --- 从 config 中提取参数 ---
        run_params = config.get('run_params', {})
        irtg_config = config.get('irtg_attacker', {})
        global_config = config.get('global', {})

        self.top_k = irtg_config.get('top_k', 5)
        self.iterations = run_params.get('iterations', 10)
        self.run_mode = run_params.get('run_mode', 'attack')
        self.optimizer_type = run_params.get('algorithm', 'greedy').lower()
        self.result_dir = global_config.get('result_dir', './results')  # 动态读取保存目录

        self.attacker_params = {
            "get_all_vars_fn": get_all_vars_fn,
            "get_subs_pool_fn": get_subs_pool_fn,
            "rename_fn": rename_fn
        }

    def attack(self, dataset: List[Dict]):
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}

        # 存储容器
        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        # 只有在使用非贪心搜索（GA）时，才真正需要用到 rankers
        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.attacker_params["rename_fn"]) for m in self.model_names}

        # 3. 根据指定的 optimizer_type 选择优化器类
        optimizers = {}
        for m in self.model_names:
            if self.optimizer_type == "greedy":
                optimizers[m] = GreedyOptimizer(
                    self.model_zoo, m, self.attacker_params["rename_fn"],
                    mode=self.mode,
                    config=self.config  # 确保你把整个 config 传进去了
                )
            else:
                optimizers[m] = GeneticAlgorithmOptimizer(
                    self.model_zoo, m, self.attacker_params["rename_fn"],
                    mode=self.mode,
                    config=self.config
                )

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            raw_variables = self.attacker_params["get_all_vars_fn"](code)
            variables = [v for v in raw_variables if not v.isupper() and not v.startswith(("av_", "spapr_", "kvm"))]

            subs_pool = self.attacker_params["get_subs_pool_fn"](code, variables)
            for var in list(subs_pool.keys()):
                if not subs_pool[var]:
                    del subs_pool[var]
                    if var in variables: variables.remove(var)

            if not variables:
                continue

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                # 4. 打印当前使用的模式
                print(
                    f"\n[Sample {idx + 1}] Target={atk_model} | Optimizer={self.optimizer_type.upper()} ({self.run_mode} mode)...")
                stats[atk_model][atk_model]["total"] += 1

                if self.optimizer_type == "greedy":
                    # 贪心模式：跳过预处理，全量变量直接交给 Greedy 跑
                    target_vars = variables.copy()
                    target_scores = None
                else:
                    # GA 模式：必须跑 RNNS 缩小搜索空间并获取分数作为突变权重
                    print("RNNS-Start...")
                    ranked_vars, all_scores = rankers[atk_model].rank_variables(
                        code=code, variables=variables.copy(), subs_pool=subs_pool, reference_label=orig_pred,
                        top_k=max(self.top_k, int(len(variables) * 0.3))
                    )
                    target_vars = ranked_vars
                    target_scores = {var: all_scores[var] for var in target_vars}

                # 5. 调用统一命名的 optimizers 字典，打印日志自适应
                print(f"{self.optimizer_type.upper()}-Start...")
                is_success, adv_code, adv_probs, adv_pred = optimizers[atk_model].run(
                    code=code, original_pred=orig_pred, target_vars=target_vars,
                    subs_pool=subs_pool, variable_scores=target_scores
                )

                # --- 存储逻辑 ---
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
        # 使用从 config 读取的路径
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
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] 已保存 {len(data)} 个样本至: {filename}")
        except Exception as e:
            print(f"[ERROR] 保存 {filename} 失败: {e}")

    def print_summary(self, stats):
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