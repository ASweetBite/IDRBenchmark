from typing import List, Dict
import json
from attacks.optimizers import GeneticAlgorithmOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZoo


class VRTGAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=5, mode="binary",
                 iterations=10):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.top_k = top_k
        self.mode = mode
        self.iterations = iterations  # 保存迭代次数
        self.attacker_params = {
            "get_all_vars_fn": get_all_vars_fn,
            "get_subs_pool_fn": get_subs_pool_fn,
            "rename_fn": rename_fn
        }

    def attack(self, dataset: List[Dict]):
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        # 1. 初始化 Ranker 和 Optimizer
        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.attacker_params["rename_fn"]) for m in self.model_names}
        ga_optimizers = {m: GeneticAlgorithmOptimizer(self.model_zoo, m, self.attacker_params["rename_fn"],
                                                      iterations=self.iterations) for m in self.model_names}

        # 2. 外层循环
        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            # --- [步骤 A]：获取所有模型的预测结果 ---
            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            # 提取变量池 (仅在有模型预测正确时才真正需要，但为了逻辑统一先提取)
            raw_variables = self.attacker_params["get_all_vars_fn"](code)
            variables = [v for v in raw_variables if not v.isupper() and not v.startswith(("av_", "spapr_", "kvm"))]
            subs_pool = self.attacker_params["get_subs_pool_fn"](code, variables)
            for var in list(subs_pool.keys()):
                if not subs_pool[var]:
                    del subs_pool[var]
                    if var in variables: variables.remove(var)
            if not variables: continue

            # --- [步骤 B]：攻击阶段 ---
            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                # 核心过滤：只有当攻击者模型本身预测正确时，才发起攻击
                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | Correct prediction detected. Starting Attack...")

                # 计数：基准测试样本数增加
                stats[atk_model][atk_model]["total"] += 1

                # A. 排序 & B. 优化
                ranked_vars, all_scores = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(), subs_pool=subs_pool, reference_label=orig_pred
                )
                dynamic_top_k = min(max(self.top_k, int(len(ranked_vars) * 0.3)), len(ranked_vars))
                target_vars = ranked_vars[:dynamic_top_k]
                target_scores = {var: all_scores[var] for var in target_vars}

                _, adv_code, adv_probs, adv_pred = ga_optimizers[atk_model].run(
                    code=code, original_pred=orig_pred, target_vars=target_vars,
                    subs_pool=subs_pool, variable_scores=target_scores
                )

                is_success = (adv_pred != orig_pred)

                if is_success:
                    stats[atk_model][atk_model]["fooled"] += 1
                    print(f"  * [Origin-Model] ✅ SUCCESS | {orig_pred} -> {adv_pred}")
                    adversarial_test_sets[atk_model].append({
                        "original_code": code, "adversarial_code": adv_code,
                        "label": ground_truth, "original_label": ground_truth
                    })
                else:
                    print(f"  * [Origin-Model] ❌ FAILED")

                # --- [步骤 C]：迁移攻击评估 (黑盒) ---
                # 只有当白盒攻击生成的对抗样本存在时才测试
                if is_success:
                    for vic_model in self.model_names:
                        if vic_model == atk_model: continue

                        vic_orig_pred = orig_predictions[vic_model]["pred"]
                        # 迁移攻击的前提：受害者模型在原始代码上也预测正确
                        if vic_orig_pred == ground_truth:
                            stats[atk_model][vic_model]["total"] += 1
                            _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)

                            if vic_adv_pred != vic_orig_pred:
                                stats[atk_model][vic_model]["fooled"] += 1
                                print(f"    - [Transfer] ✅ {vic_model} FOOLED ({vic_orig_pred} -> {vic_adv_pred})")
                            else:
                                print(f"    - [Transfer] ❌ {vic_model} resisted")

        self.print_summary(stats)
        for atk_model in self.model_names:
            if adversarial_test_sets[atk_model]:
                self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])
            else:
                print(f"[INFO] {atk_model} 没有生成任何成功的对抗样本，跳过保存。")

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        # 改为以 .json 保存
        filename = f"adv_test_set_{model_name}_{self.mode}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(test_set, f, indent=4, ensure_ascii=False)
            print(f"[INFO] {model_name} 已保存 {len(test_set)} 个样本对至: {filename}")
        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")

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