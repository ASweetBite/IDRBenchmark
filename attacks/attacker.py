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
        variables = self.get_all_vars_fn(code)
        subs_pool = self.get_subs_pool_fn(code, variables)

        if not variables:
            return {"success": False, "adv_code": code, "orig_pred": orig_pred, "adv_pred": orig_pred}

        ranked_vars = self.rnns_ranker.rank_variables(code, variables, subs_pool, orig_pred)
        target_vars = ranked_vars[:self.top_k]

        success, adv_code, adv_probs, adv_pred = self.ga_optimizer.run(
            code, orig_pred, target_vars, subs_pool
        )
        return {"success": success, "adv_code": adv_code, "orig_pred": orig_pred, "adv_pred": adv_pred}

class TransferabilityEvaluator:
    def __init__(self, attacker: VRTGAttacker, model_zoo: ModelZoo):
        self.attacker = attacker
        self.model_zoo = model_zoo
        self.target_model = attacker.target_model
        self.other_models = [m for m in model_zoo.model_names if m != self.target_model]

    def evaluate(self, dataset: List[Dict]):
        total = 0
        stats = {m: {"correct_orig": 0, "fooled": 0} for m in self.model_zoo.model_names}

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            true_label = sample["label"]

            # 1. 获取原代码的预测结果
            orig_preds = {}
            orig_probs = {}
            for m in self.model_zoo.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_probs[m] = probs
                orig_preds[m] = pred

            if orig_preds[self.target_model] != true_label:
                continue

            total += 1
            print(f"\n{'=' * 80}")
            print(
                f"[Sample {idx + 1}] Target={self.target_model} | TrueLabel={true_label} | OrigPred={orig_preds[self.target_model]}")

            # 2. 执行攻击
            res = self.attacker.attack(code)
            adv_code = res["adv_code"]

            # 3. 获取对抗样本的预测结果 (用于调试)
            adv_probs, adv_pred = self.model_zoo.predict(adv_code, self.target_model)

            # --- 详细日志打印 ---
            print(f"  [*] Attack Status: {'SUCCESS' if res['success'] else 'FAILED'}")
            print(f"  [*] Probs Change:  {orig_probs[self.target_model]} -> {adv_probs}")

            print(f"  [*] Code Diff (Original -> Adversarial):")
            diff = difflib.unified_diff(
                code.splitlines(), adv_code.splitlines(),
                fromfile='Original', tofile='Adversarial', lineterm=''
            )
            # 限制 diff 输出长度，避免刷屏
            diff_lines = list(diff)
            for line in diff_lines[:20]:  # 只打印前20行差异
                print(f"      {line}")
            if len(diff_lines) > 20:
                print("      ... (diff truncated)")

            # 4. 评估跨模型迁移性 (保持原有逻辑)
            print(f"  [*] Transferability Test:")
            for m in self.model_zoo.model_names:
                # 只有原代码预测正确的才计算迁移性
                if orig_preds[m] == true_label:
                    _, m_adv_pred = self.model_zoo.predict(adv_code, m)
                    if m_adv_pred != true_label:
                        stats[m]["fooled"] += 1
                        status = "✅ FOOLED"
                    else:
                        status = "❌ ROBUST"
                    print(f"      - {m:<10}: {status} (Orig: {orig_preds[m]} -> Adv: {m_adv_pred})")
        # 输出统计结果
        print("\n" + "=" * 60)
        print("📊 CROSS-MODEL TRANSFERABILITY REPORT")
        print("=" * 60)
        print(f"Valid attack attempts (Target model correctly predicted orig): {total}\n")

        for m in self.model_zoo.model_names:
            correct = stats[m]["correct_orig"]
            fooled = stats[m]["fooled"]
            asr = (fooled / correct * 100) if correct > 0 else 0

            tag = "[Target]" if m == self.target_model else "[Transfer]"
            print(f"{tag} {m:<10}: Orig Accuracy={correct}/{len(dataset)} | ASR={asr:.2f}% ({fooled}/{correct})")
