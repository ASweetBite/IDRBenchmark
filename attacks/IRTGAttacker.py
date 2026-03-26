from typing import List, Dict
import json
import os
from attacks.optimizers import GeneticAlgorithmOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZoo


class IRTGAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=5, mode="binary",
                 iterations=10, run_mode="attack"):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.top_k = top_k
        self.mode = mode
        self.iterations = iterations
        self.run_mode = run_mode
        self.attacker_params = {
            "get_all_vars_fn": get_all_vars_fn,
            "get_subs_pool_fn": get_subs_pool_fn,
            "rename_fn": rename_fn
        }

    def attack(self, dataset: List[Dict]):
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}

        # 存储容器
        # dataset 模式下：存储两个独立的列表
        # attack 模式下：存储原始-对抗对
        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.attacker_params["rename_fn"]) for m in self.model_names}
        ga_optimizers = {m: GeneticAlgorithmOptimizer(self.model_zoo, m, self.attacker_params["rename_fn"],
                                                      iterations=self.iterations, run_mode=self.run_mode) for m in
                         self.model_names}

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            # 预预测，用于过滤和后续对比
            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            # 提取变量
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

                # 只有模型原本预测正确的才参与攻击/生成
                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | Starting GA ({self.run_mode} mode)...")
                stats[atk_model][atk_model]["total"] += 1

                # 变量排序
                ranked_vars, all_scores = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(), subs_pool=subs_pool, reference_label=orig_pred
                )
                dynamic_top_k = min(max(self.top_k, int(len(ranked_vars) * 0.3)), len(ranked_vars))
                target_vars = ranked_vars[:dynamic_top_k]
                target_scores = {var: all_scores[var] for var in target_vars}

                # 运行优化器
                # 在 dataset 模式下，run 会跑满 iterations，并返回 best_code
                is_success, adv_code, adv_probs, adv_pred = ga_optimizers[atk_model].run(
                    code=code, original_pred=orig_pred, target_vars=target_vars,
                    subs_pool=subs_pool, variable_scores=target_scores
                )

                # --- 存储逻辑 ---
                if self.run_mode == "dataset":
                    # 无论成功与否，全部保存到对应的两个列表中
                    storage_orig[atk_model].append({"func": code, "label": ground_truth})
                    storage_adv[atk_model].append({"func": adv_code, "label": ground_truth})

                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        print(f"  * [GA] ✅ Success (Label flipped)")
                    else:
                        print(f"  * [GA] ❌ Failed to flip label (Best sample kept)")

                else:  # attack 模式
                    if is_success:
                        stats[atk_model][atk_model]["fooled"] += 1
                        # 仅保存成功的对
                        storage_adv[atk_model].append({
                            "original_code": code,
                            "adversarial_code": adv_code,
                            "label": ground_truth
                        })
                        print(f"  * [GA] ✅ Success | {orig_pred} -> {adv_pred}")
                    else:
                        print(f"  * [GA] ❌ Failed")

                # --- 迁移攻击统计 (仅在成功时评估) ---
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

        return asr_matrix  # 返回 ASR 结果字典

    def save_results(self, storage_orig, storage_adv):
        # 自动创建 results 文件夹（不存在则创建）
        result_dir = "./results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for model in self.model_names:
            if self.run_mode == "dataset":
                # 保存原始样本文件
                if storage_orig[model]:
                    orig_filename = f"orig_dataset_{model}_{self.mode}.json"
                    # 拼接路径：results/文件名
                    orig_path = os.path.join(result_dir, orig_filename)
                    self._write_json(orig_path, storage_orig[model])

                # 保存生成的对抗样本文件
                if storage_adv[model]:
                    adv_filename = f"adv_dataset_{model}_{self.mode}.json"
                    adv_path = os.path.join(result_dir, adv_filename)
                    self._write_json(adv_path, storage_adv[model])
            else:
                # 攻击模式：保存到一个文件
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