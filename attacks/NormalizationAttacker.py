import os
import re
import json
from typing import List, Dict, Tuple
from utils.model_zoo import ModelZoo


class NormalizationAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, rename_fn, mode="binary"):
        """
        :param model_zoo: 模型动物园对象
        :param get_all_vars_fn: 获取变量的函数
        :param rename_fn: 执行替换的底层函数（通常是对接正则表达式或AST）
        :param mode: 任务模式
        """
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.get_all_vars_fn = get_all_vars_fn
        self.rename_fn = rename_fn  # 虽然我们自己生成映射，但实际替换还是建议用原本的 rename_fn 确保语法正确

    def _generate_sequential_mapping(self, code: str, variables: List[str]) -> Dict[str, str]:
        """
        根据变量在代码中出现的先后顺序，生成 {原变量名: VAR_N} 的映射表
        """
        # 记录每个变量第一次出现的位置
        var_positions = {}
        for var in variables:
            pos = code.find(var)
            if pos != -1:
                var_positions[var] = pos

        # 按出现位置排序
        sorted_vars = sorted(var_positions.keys(), key=lambda x: var_positions[x])

        # 生成映射表
        return {var: f"VAR_{i + 1}" for i, var in enumerate(sorted_vars)}

    def attack(self, dataset: List[Dict]):
        # 初始化统计数据 (与原版一致)
        stats = {atk: {vic: {"total": 0, "fooled": 0} for vic in self.model_names}
                 for atk in self.model_names}
        adversarial_test_sets = {m: [] for m in self.model_names}

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample.get("label")

            # --- [步骤 A]：获取所有模型的原始预测 ---
            orig_predictions = {}
            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}

            # 提取变量并过滤（逻辑同原版）
            raw_variables = self.get_all_vars_fn(code)
            variables = [v for v in raw_variables if not v.isupper() and not v.startswith(("av_", "spapr_", "kvm"))]
            if not variables:
                continue

            # --- [步骤 B]：攻击阶段 (顺序重命名) ---
            # 生成规律化的重命名映射表 {"x": "VAR_1", "y": "VAR_2"...}
            rename_map = self._generate_sequential_mapping(code, variables)
            # 使用映射表修改代码
            adv_code = self.rename_fn(code, rename_map)

            for atk_model in self.model_names:
                orig_pred = orig_predictions[atk_model]["pred"]

                # 只有当模型原本预测正确时，才纳入攻击统计
                if orig_pred != ground_truth:
                    continue

                print(f"\n[Sample {idx + 1}] Target={atk_model} | Normalizing Variables to VAR_N...")
                stats[atk_model][atk_model]["total"] += 1

                # 检查重命名后的预测结果
                _, adv_pred = self.model_zoo.predict(adv_code, atk_model)
                is_success = (adv_pred != orig_pred)

                if is_success:
                    stats[atk_model][atk_model]["fooled"] += 1
                    print(f"  * [Normalization] ✅ SUCCESS | {orig_pred} -> {adv_pred}")
                    adversarial_test_sets[atk_model].append({
                        "original_code": code,
                        "adversarial_code": adv_code,
                        "label": ground_truth,
                        "original_label": ground_truth,
                        "rename_map": rename_map
                    })
                else:
                    # print(f"  * [White-Box] ❌ FAILED (Still predicted {adv_pred})")
                    pass

                # # --- [步骤 C]：迁移攻击评估 (黑盒) ---
                # if is_success:
                #     for vic_model in self.model_names:
                #         if vic_model == atk_model: continue
                #
                #         vic_orig_pred = orig_predictions[vic_model]["pred"]
                #         if vic_orig_pred == ground_truth:
                #             stats[atk_model][vic_model]["total"] += 1
                #             _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                #
                #             if vic_adv_pred != vic_orig_pred:
                #                 stats[atk_model][vic_model]["fooled"] += 1
                #                 print(f"    - [Transfer] ✅ {vic_model} FOOLED")
                #             else:
                #                 print(f"    - [Transfer] ❌ {vic_model} resisted")

        # 打印最终矩阵
        self.print_summary(stats)
        # 保存结果
        for atk_model in self.model_names:
            if adversarial_test_sets[atk_model]:
                self.save_as_test_set(atk_model, adversarial_test_sets[atk_model])

        asr_matrix = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        return asr_matrix  # 返回字典结果

    def save_as_test_set(self, model_name: str, test_set: List[Dict]):
        result_dir = "./results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        filename = f"norm_test_set_{model_name}_{self.mode}.json"
        file_path = os.path.join(result_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(test_set, f, indent=4, ensure_ascii=False)
            print(f"[INFO] {model_name} 已保存 {len(test_set)} 个样本至: {file_path}")
        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")

    def print_summary(self, stats):
        print("\n" + "=" * 90)
        print("📊 NORMALIZATION ATTACK SUCCESS RATE (ASR %)")
        print("=" * 90)
        header = f"{'Source Model':<20} |"
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