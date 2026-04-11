import string
from typing import List, Dict
import random
from utils.ast_tools import is_valid_identifier
from utils.model_zoo import ModelZoo


class RandomAttacker:
    """Implements a randomized identifier renaming attack to evaluate code model robustness."""

    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, mode="binary"):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.get_all_vars_fn = get_all_vars_fn
        self.rename_fn = rename_fn
        self.analyzer = None

    def set_analyzer(self, analyzer):
        """Attaches an AST analyzer to handle scope validation and conflict checks."""
        self.analyzer = analyzer

    def _generate_random_identifier(self, min_len=6, max_len=12) -> str:
        """Generates a syntactically valid C/C++ identifier consisting of random characters."""
        first_char = random.choice(string.ascii_lowercase + '_')
        rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_')
                             for _ in range(random.randint(min_len, max_len)))
        rand_name = first_char + rest_chars

        return rand_name if is_valid_identifier(rand_name) else self._generate_random_identifier()

    def attack_sample(self, code: str, ground_truth: int, target_model: str):
        """Executes the random renaming attack on a single code snippet and returns success metrics."""
        orig_probs, orig_pred = self.model_zoo.predict(code, target_model)

        if orig_pred != ground_truth:
            return False, code, orig_probs, orig_pred

        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)
        if not identifiers:
            return False, code, orig_probs, orig_pred

        renaming_map = {}
        used_names = set(identifiers.keys())

        for var_name in identifiers.keys():
            while True:
                rand_name = self._generate_random_identifier()
                if rand_name not in used_names and self.analyzer.can_rename_to(code_bytes, var_name, rand_name):
                    renaming_map[var_name] = rand_name
                    used_names.add(rand_name)
                    break

        try:
            adv_code = self.rename_fn(code, renaming_map)
        except:
            return False, code, orig_probs, orig_pred

        adv_probs, adv_pred = self.model_zoo.predict(adv_code, target_model)
        attack_success = (adv_pred != orig_pred)

        return attack_success, adv_code, adv_probs, adv_pred

    def attack(self, dataset: List[Dict]):
        """Orchestrates the attack over a dataset and returns the resulting Attack Success Rate (ASR) matrix."""
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}

        adv_samples = {model: [] for model in self.model_names}

        print("\n" + "=" * 80)
        print("🔍 Initializing Random Renaming Attack")
        print("=" * 80)

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample["label"]

            for target_model in self.model_names:
                success, adv_code, adv_probs, adv_pred = self.attack_sample(code, ground_truth, target_model)

                stats[target_model][target_model]["total"] += 1

                if success:
                    stats[target_model][target_model]["fooled"] += 1
                    adv_samples[target_model].append({
                        "original_code": code,
                        "adversarial_code": adv_code,
                        "label": ground_truth
                    })
                    print(f"[Sample {idx + 1}] {target_model} | Random Rename ✅ Attack Success")

        self.print_summary(stats)

        asr_matrix = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        return asr_matrix

    def print_summary(self, stats):
        """Displays a summary table of the self-attack success rates for all targeted models."""
        print("\n" + "=" * 60)
        print("📊 Random Renaming Attack - Success Rate (ASR) [Self-Attack Only]")
        print("=" * 60)
        for model in self.model_names:
            res = stats[model][model]
            asr = (res["fooled"] / res["total"] * 100) if res["total"] > 0 else 0.0
            print(f"{model:<15} | ASR: {asr:.2f}% ({res['fooled']}/{res['total']})")
        print("=" * 60 + "\n")