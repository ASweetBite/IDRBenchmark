import os
import re
import json
from typing import List, Dict, Tuple
from utils.model_zoo import ModelZoo


class NormalizationAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_with_types_fn, rename_fn, mode="binary"):
        """
        :param model_zoo: 模型对象
        :param get_all_vars_with_types_fn: 函数应返回 List[Tuple[变量名, 类型字符串]]
        :param rename_fn: 执行替换的底层函数
        :param mode: 任务模式
        """
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        # 注意：此函数现在需要返回 (name, type) 的元组
        self.get_all_vars_fn = get_all_vars_with_types_fn
        self.rename_fn = rename_fn

    def _generate_type_aware_mapping(self, code: str, var_type_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        根据变量类型和出现顺序生成映射表：
        1. 指针 -> pointer_N
        2. int -> int_N
        3. char -> char_N
        4. 类对象 -> 类名_N
        """
        # 记录每个变量第一次出现的位置
        var_info = []
        for var_name, var_type in var_type_pairs:
            pos = code.find(var_name)
            if pos != -1:
                var_info.append({
                    "name": var_name,
                    "type": var_type.strip(),
                    "pos": pos
                })

        # 按在代码中出现的先后顺序排序
        sorted_vars = sorted(var_info, key=lambda x: x["pos"])

        mapping = {}
        counters = {}  # 存储每种类型的计数器，例如 {"int": 1, "pointer": 2, "UserClass": 1}

        for item in sorted_vars:
            name = item["name"]
            v_type = item["type"]

            # --- 判定重命名分类 ---
            # 1. 如果是指针 (包含 * 号)
            if "*" in v_type:
                category = "pointer"
            # 2. 如果是 char 型
            elif "char" in v_type.lower():
                category = "char"
            # 3. 如果是 int 型 (包括 long, short, unsigned int 等)
            elif "int" in v_type.lower() or "long" in v_type.lower() or "short" in v_type.lower():
                category = "int"
            # 4. 其他情况视为类对象或自定义类型
            else:
                # 去掉类型中的空格，处理类似 "struct MyClass" 的情况，取最后一部分
                category = v_type.split()[-1]

            # 更新计数器并生成新名字
            counters[category] = counters.get(category, 0) + 1
            new_name = f"{category}_{counters[category]}"
            mapping[name] = new_name

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
            # 这里的 get_all_vars_fn 预期返回 List[Tuple[name, type]]
            raw_var_pairs = self.get_all_vars_fn(code)

            # 过滤逻辑：去掉全大写和系统特定前缀
            filtered_pairs = [
                (name, v_type) for name, v_type in raw_var_pairs
                if not name.isupper() and not name.startswith(("av_", "spapr_", "kvm"))
            ]

            if not filtered_pairs:
                continue

            # --- [步骤 C]：生成基于类型的归一化映射 ---
            rename_map = self._generate_type_aware_mapping(code, filtered_pairs)
            # 使用映射表修改代码
            adv_code = self.rename_fn(code, rename_map)

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | Type-Aware Normalizing...")
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

                # (可选) 迁移攻击部分可根据需要在此开启...

        self.print_summary(stats)

        # 保存结果
        for atk_model in self.model_names:
            if adversarial_test_sets[atk_model]:
                self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])

        return stats

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        result_dir = "./results"
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        filename = f"type_norm_test_{model_name}_{self.mode}.json"
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4, ensure_ascii=False)
        print(f"[INFO] 已保存至: {file_path}")

    def print_summary(self, stats):
        print("\n" + "=" * 90)
        print("📊 TYPE-AWARE NORMALIZATION ATTACK SUCCESS RATE")
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