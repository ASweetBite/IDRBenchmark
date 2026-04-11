import os
import re
import json
from typing import List, Dict, Tuple

from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.model_zoo import ModelZoo


class NormalizationAttacker:
    def __init__(self, model_zoo: ModelZoo, generator: HeavyWeightCandidateGenerator,
                 get_all_vars_with_types_fn, rename_fn, mode="binary"):
        """Initializes the attacker with a model zoo, candidate generator, and AST utility functions."""
        self.model_zoo = model_zoo
        self.generator = generator
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.get_all_vars_fn = get_all_vars_with_types_fn
        self.rename_fn = rename_fn

    def _generate_type_aware_mapping(self, code: str, var_type_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        """Generates a renaming map using type-based categorization combined with MLM-based word prediction."""
        var_info = []
        for var_name, var_type in var_type_pairs:
            pos = code.find(var_name)
            if pos != -1:
                var_info.append({
                    "name": var_name,
                    "type": var_type.strip(),
                    "pos": pos
                })

        sorted_vars = sorted(var_info, key=lambda x: x["pos"])

        mapping = {}
        used_names = set()

        for item in sorted_vars:
            name = item["name"]
            v_type = item["type"]

            if "*" in v_type:
                category = "pointer"
            elif "char" in v_type.lower():
                category = "char"
            elif "int" in v_type.lower() or "long" in v_type.lower() or "short" in v_type.lower():
                category = "int"
            else:
                category = v_type.split()[-1]

            new_name = self.generator.generate_normalized_name(
                code=code,
                target_name=name,
                var_type=category,
                excluded_names=used_names
            )

            mapping[name] = new_name
            used_names.add(new_name)

        return mapping

    def attack(self, dataset: List[Dict]):
        """Executes the normalization attack across the dataset and evaluates success rates for each model."""
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            raw_var_pairs = self.get_all_vars_fn(code)

            filtered_pairs = []
            for item in raw_var_pairs:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    name, v_type = item[0], item[1]
                elif isinstance(item, str):
                    name, v_type = item, "void"
                else:
                    continue

                if not name.isupper() and not name.startswith(("av_", "spapr_", "kvm")):
                    filtered_pairs.append((name, v_type))

            if not filtered_pairs:
                continue

            rename_map = self._generate_type_aware_mapping(code, filtered_pairs)
            adv_code = self.rename_fn(code, rename_map)

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | MLM Type-Aware Normalizing...")
                stats[atk_model][atk_model]["total"] += 1

                _, adv_pred = self.model_zoo.predict(adv_code, atk_model)
                is_success = (adv_pred != orig_pred)

                if is_success:
                    stats[atk_model][atk_model]["fooled"] += 1
                    print(f"  * [Success] {orig_pred} -> {adv_pred}")
                    adversarial_test_sets[atk_model].append({
                        "original_code": code,
                        "adversarial_code": adv_code,
                        "label": ground_truth,
                        "rename_map": rename_map
                    })

        self.print_summary(stats)

        for atk_model in self.model_names:
            if adversarial_test_sets[atk_model]:
                self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])

        return stats

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        """Serializes the generated adversarial test set to a JSON file."""
        result_dir = "./results"
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        filename = f"mlm_type_norm_{model_name}_{self.mode}.json"
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved to: {file_path}")

    def print_summary(self, stats):
        """Prints a summary matrix of the attack success rates for all evaluated models."""
        print("\n" + "=" * 90)
        print("📊 MLM TYPE-AWARE NORMALIZATION ATTACK SUCCESS RATE")
        print("=" * 90)
        header = f"{'Source Model':<20} |"
        for m in self.model_names: header += f" {m:<13} |"
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