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
        # stats 记录格式: [attacker][victim] = {"total": 0, "fooled": 0}
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        # 1. 初始化 Ranker 和 Optimizer (传入 iterations 参数)
        rankers = {
            m: RNNS_Ranker(self.model_zoo, m, self.attacker_params["rename_fn"])
            for m in self.model_names
        }
        ga_optimizers = {
            m: GeneticAlgorithmOptimizer(
                self.model_zoo,
                m,
                self.attacker_params["rename_fn"],
                iterations=self.iterations  # 传入迭代参数
            )
            for m in self.model_names
        }

        # 2. 外层循环
        for idx, sample in enumerate(dataset):
            code = sample["code"]
            print(f"\n{'=' * 80}\n[Sample {idx + 1}/{len(dataset)}] Extracting Base Features...\n{'=' * 80}")

            raw_variables = self.attacker_params["get_all_vars_fn"](code)
            # 过滤不需要攻击的特殊变量名
            variables = [v for v in raw_variables if not v.isupper() and not v.startswith(("av_", "spapr_", "kvm"))]
            subs_pool = self.attacker_params["get_subs_pool_fn"](code, variables)

            for var in list(subs_pool.keys()):
                if not subs_pool[var]:
                    del subs_pool[var]
                    if var in variables: variables.remove(var)

            if not variables: continue

            # 获取所有模型对原代码的预测
            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            # 阶段二：攻击
            for atk_model in self.model_names:
                print(f"\n  >>> ATTACK: Target={atk_model}, Mode={self.mode}, Iter={self.iterations} <<<")
                orig_pred = orig_predictions[atk_model]["pred"]

                # A. RNNS 排序
                ranked_vars, all_scores = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(), subs_pool=subs_pool, reference_label=orig_pred
                )

                # 【核心修改点3：动态放宽变量选择】
                # 策略：至少选取 self.top_k (如 5) 个，或提取总变量数的前 30%（取二者较大值）
                # 同时确保不会超过当前代码实际拥有的可用变量总数
                dynamic_top_k = max(self.top_k, int(len(ranked_vars) * 0.3))
                dynamic_top_k = min(dynamic_top_k, len(ranked_vars))

                target_vars = ranked_vars[:dynamic_top_k]

                print(f"  * Selected Target Vars ({len(target_vars)}/{len(ranked_vars)}): {target_vars}")
                target_scores = {var: all_scores[var] for var in target_vars}

                # B. GA 优化
                _, adv_code, adv_probs, adv_pred = ga_optimizers[atk_model].run(
                    code=code,
                    original_pred=orig_pred,
                    target_vars=target_vars,
                    subs_pool=subs_pool,
                    variable_scores=target_scores
                )

                # C. 统一判定成功标准：只要预测发生了改变 (不等于原预测)，即为成功
                is_success = (adv_pred != orig_pred)

                print(
                    f"  * Status: {'SUCCESS' if is_success else 'FAILED'} | Orig Pred: {orig_pred} -> Adv Pred: {adv_pred}")

                # 阶段三：迁移性测试
                if is_success:
                    adversarial_test_sets[atk_model].append({"code": adv_code, "label": adv_pred})

                    for vic_model in self.model_names:
                        stats[atk_model][vic_model]["total"] += 1
                        _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                        vic_orig_pred = orig_predictions[vic_model]["pred"]

                        # 迁移性判定：对抗样本在受害者模型上的预测是否改变
                        if vic_adv_pred != vic_orig_pred:
                            stats[atk_model][vic_model]["fooled"] += 1
                            print(f"    - {vic_model:<13}: ✅ FOOLED ({vic_orig_pred} -> {vic_adv_pred})")

        self.print_summary(stats)
        # ... 保存代码省略 ...

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        filename = f"test_set_adv_by_{model_name}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4)
        print(f"\n[INFO] {model_name} 生成的 {len(test_set)} 个样本已保存为测试集: {filename}")

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