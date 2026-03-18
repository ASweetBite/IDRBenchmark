import difflib
from typing import List, Dict

import json

from attacks.optimizers import GeneticAlgorithmOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZoo


class VRTGAttacker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=5):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.get_all_vars_fn = get_all_vars_fn
        self.get_subs_pool_fn = get_subs_pool_fn
        self.rnns_ranker = RNNS_Ranker(model_zoo, target_model, rename_fn)
        self.ga_optimizer = GeneticAlgorithmOptimizer(model_zoo, target_model, rename_fn)
        self.top_k = top_k

    def attack(self, code: str):
        orig_probs, orig_pred = self.model_zoo.predict(code, self.target_model)

        raw_variables = self.get_all_vars_fn(code)
        variables = [v for v in raw_variables if not v.isupper() and not v.startswith(("av_", "spapr_", "kvm"))]
        subs_pool = self.get_subs_pool_fn(code, variables)

        # 过滤
        for var in list(subs_pool.keys()):
            if not subs_pool[var]:
                del subs_pool[var]
                if var in variables:
                    variables.remove(var)

        if not variables:
            return {"success": False, "adv_code": code, "orig_probs": orig_probs, "adv_probs": orig_probs,
                    "orig_pred": orig_pred, "adv_pred": orig_pred}

        ranked_vars = self.rnns_ranker.rank_variables(code, variables, subs_pool, orig_pred)
        target_vars = ranked_vars[:self.top_k]

        success, adv_code, adv_probs, adv_pred = self.ga_optimizer.run(
            code, orig_pred, target_vars, subs_pool
        )

        # 返回所有必要信息
        return {
            "success": success,
            "adv_code": adv_code,
            "orig_probs": orig_probs,
            "adv_probs": adv_probs,
            "orig_pred": orig_pred,
            "adv_pred": adv_pred
        }


class TransferabilityEvaluator:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.attacker_params = {
            "get_all_vars_fn": get_all_vars_fn,
            "get_subs_pool_fn": get_subs_pool_fn,
            "rename_fn": rename_fn
        }

    def evaluate(self, dataset: List[Dict]):
        # stats 记录格式: [attacker][victim] = {"total": 0, "fooled": 0}
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        for atk_model in self.model_names:
            print(f"\n{'#' * 80}\n### ATTACK CAMPAIGN: Target Model = {atk_model} ###\n{'#' * 80}")
            attacker = VRTGAttacker(self.model_zoo, atk_model, **self.attacker_params)

            for idx, sample in enumerate(dataset):
                code = sample["code"]
                orig_preds = {m: self.model_zoo.predict(code, m)[1] for m in self.model_names}

                # 1. 执行攻击
                res = attacker.attack(code)

                # 2. 打印攻击结果 (只看 Target Model)
                print(f"\n[Sample {idx + 1}] Target: {atk_model} | Status: {'SUCCESS' if res['success'] else 'FAILED'}")
                print(f"  * Orig -> Adv Pred: {res['orig_pred']} -> {res['adv_pred']}")

                # 3. 只有在攻击成功时，才测试迁移性
                if res['success']:
                    adv_code = res["adv_code"]
                    print(f"  * Transferability Test (on other models):")
                    adv_entry = {
                        "code": res["adv_code"],
                        "label": res["adv_pred"]
                    }
                    adversarial_test_sets[atk_model].append(adv_entry)

                    for vic_model in self.model_names:
                        # 统计逻辑：只有在这个模型上产生了预测翻转，才算成功迁移
                        stats[atk_model][vic_model]["total"] += 1

                        _, adv_pred = self.model_zoo.predict(adv_code, vic_model)

                        if adv_pred != orig_preds[vic_model]:
                            stats[atk_model][vic_model]["fooled"] += 1
                            print(f"    - {vic_model:<13}: ✅ FOOLED (Orig: {orig_preds[vic_model]} -> Adv: {adv_pred})")
                        else:
                            # 只有输出不同的才打印，如果保持 Robust，可以静默或者简单提示，避免刷屏
                            # print(f"    - {vic_model:<13}: ❌ ROBUST")
                            pass
                else:
                    print(f"  * Attack Failed, skipping transferability test.")
            self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])

        self.print_summary(stats)

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        """将生成的对抗性样本保存为测试集文件"""
        filename = f"test_set_adv_by_{model_name}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            # 存为标准的 JSON 数组格式
            json.dump(test_set, f, indent=4)

        print(f"\n[INFO] {model_name} 生成的 {len(test_set)} 个样本已保存为测试集: {filename}")

    def print_summary(self, stats):
        # (保持原样即可)
        print("\n" + "=" * 90)
        print("📊 FINAL CROSS-MODEL TRANSFERABILITY MATRIX (ASR %)")
        print("=" * 90)

        # 打印表头
        header = f"{'Attacker \\ Victim':<20} |"
        for m in self.model_names:
            header += f" {m:<13} |"
        print(header)
        print("-" * len(header))

        # 打印每一行
        for atk_m in self.model_names:
            row = f"{atk_m:<20} |"
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                row += f" {asr:>11.2f}% |"
            print(row)
        print("=" * 90 + "\n")