import os
import re
import json
from typing import List, Dict, Tuple

from attacks.generators import CodeBasedCandidateGenerator
from utils.model_zoo import ModelZoo


class NormalizationAttacker:
    def __init__(self, model_zoo: ModelZoo, generator: CodeBasedCandidateGenerator,
                 get_all_vars_with_types_fn, rename_fn, mode="binary"):
        self.model_zoo = model_zoo
        self.generator = generator  # 现在持有一个支持 MLM 的生成器实例
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.get_all_vars_fn = get_all_vars_with_types_fn
        self.rename_fn = rename_fn

    def _generate_type_aware_mapping(self, code: str, var_type_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        结合旧版的类型分类逻辑与新版的 MLM 单词生成逻辑
        """
        # 1. 记录每个变量第一次出现的位置并排序
        var_info = []
        for var_name, var_type in var_type_pairs:
            pos = code.find(var_name)
            if pos != -1:
                var_info.append({
                    "name": var_name,
                    "type": var_type.strip(),
                    "pos": pos
                })

        # 按在代码中出现的先后顺序排序，保证 MLM 预测的上下文顺序
        sorted_vars = sorted(var_info, key=lambda x: x["pos"])

        mapping = {}
        used_names = set()

        for item in sorted_vars:
            name = item["name"]
            v_type = item["type"]

            # --- [保留旧版逻辑]：判定重命名分类（作为前缀） ---
            if "*" in v_type:
                category = "pointer"
            elif "char" in v_type.lower():
                category = "char"
            elif "int" in v_type.lower() or "long" in v_type.lower() or "short" in v_type.lower():
                category = "int"
            else:
                # 处理类似 "struct MyClass" 的情况，取最后一部分作为前缀
                category = v_type.split()[-1]

            # --- [核心修改]：调用 Generator，传入完整 code 和识别出的 category ---
            # 这里的 generator 会在 code 中寻找 name 的位置并用 MLM 预测单词
            new_name = self.generator.generate_normalized_name(
                code=code,
                target_name=name,
                var_type=category,  # 传入我们识别出的类型前缀
                excluded_names=used_names
            )

            mapping[name] = new_name
            used_names.add(new_name)

        return mapping

    def attack(self, dataset: List[Dict]):
        # 初始化统计数据
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            # --- [步骤 A]：获取原始预测 ---
            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            # --- [步骤 B]：获取变量及其类型并过滤 ---
            raw_var_pairs = self.get_all_vars_fn(code)

            # 过滤逻辑：去掉全大写和系统特定前缀（保留你之前的过滤规则）
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

            # --- [步骤 C]：生成基于 MLM + 类型的归一化映射 ---
            # 这里现在会传入 code 到映射生成函数中
            rename_map = self._generate_type_aware_mapping(code, filtered_pairs)

            # 使用映射表修改代码
            adv_code = self.rename_fn(code, rename_map)

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | MLM Type-Aware Normalizing...")
                stats[atk_model][atk_model]["total"] += 1

                # 检查重命名后的预测结果
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

        # 保存结果
        for atk_model in self.model_names:
            if adversarial_test_sets[atk_model]:
                self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])

        return stats

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        result_dir = "./results"
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        filename = f"mlm_type_norm_{model_name}_{self.mode}.json"
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4, ensure_ascii=False)
        print(f"[INFO] 已保存至: {file_path}")

    def print_summary(self, stats):
        # ... (保持原有的 print_summary 代码不变)
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