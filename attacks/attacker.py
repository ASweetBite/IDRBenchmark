import difflib
from typing import List, Dict
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
        all_existing_vars = set(raw_variables)
        subs_pool = self.get_subs_pool_fn(code, variables)

        # 过滤
        for var in list(subs_pool.keys()):
            subs_pool[var] = [cand for cand in subs_pool[var] if cand not in all_existing_vars or cand == var]
            if not subs_pool[var]:
                del subs_pool[var]
                if var in variables: variables.remove(var)

        if not variables:
            return {"success": False, "adv_code": code, "orig_probs": orig_probs, "adv_probs": orig_probs,
                    "orig_pred": orig_pred, "adv_pred": orig_pred}

        ranked_vars = self.rnns_ranker.rank_variables(code, variables, subs_pool, orig_pred)
        target_vars = ranked_vars[:self.top_k]

        success, adv_code, adv_probs, adv_pred = self.ga_optimizer.run(
            code, orig_pred, target_vars, subs_pool, all_existing_vars
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
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}

        for atk_model in self.model_names:
            print(f"\n{'#' * 80}\n### TARGETING MODEL: {atk_model} ###\n{'#' * 80}")
            attacker = VRTGAttacker(self.model_zoo, atk_model, **self.attacker_params)

            for idx, sample in enumerate(dataset):
                code = sample["code"]
                orig_preds = {m: self.model_zoo.predict(code, m)[1] for m in self.model_names}

                # --- 执行攻击 ---
                res = attacker.attack(code)
                adv_code = res["adv_code"]

                # --- 打印中间结果 ---
                print(f"\n[Sample {idx + 1}] Result: {'SUCCESS' if res['success'] else 'FAILED'}")
                print(f"  * Orig Pred: {res['orig_pred']} | Adv Pred: {res['adv_pred']}")
                print(
                    f"  * Probs:     {list(map(lambda x: round(x, 4), res['orig_probs']))} -> {list(map(lambda x: round(x, 4), res['adv_probs']))}")

                # 打印 Diff
                diff = list(difflib.unified_diff(code.splitlines(), adv_code.splitlines(), lineterm=''))
                if diff:
                    print(f"  * Code Diff (first 10 lines):")
                    for line in diff[:12]:
                        print(f"    {line}")

                # --- 评估迁移性 ---
                print(f"  * Transferability Test:")
                for vic_model in self.model_names:
                    stats[atk_model][vic_model]["total"] += 1
                    _, adv_pred = self.model_zoo.predict(adv_code, vic_model)

                    status = "✅ FOOLED" if adv_pred != orig_preds[vic_model] else "❌ ROBUST"
                    if adv_pred != orig_preds[vic_model]:
                        stats[atk_model][vic_model]["fooled"] += 1

                    print(f"    - {vic_model:<13}: {status} (Orig: {orig_preds[vic_model]} -> Adv: {adv_pred})")

        self.print_summary(stats)

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