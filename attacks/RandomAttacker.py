import string
from typing import List, Dict
import random
from utils.ast_tools import is_valid_identifier
from utils.model_zoo import ModelZoo

class RandomAttacker:
    def __init__(self, model_zoo: ModelZoo, get_all_vars_fn, get_subs_pool_fn, rename_fn, mode="binary"):
        self.model_zoo = model_zoo
        self.model_names = model_zoo.model_names
        self.mode = mode
        self.get_all_vars_fn = get_all_vars_fn
        self.rename_fn = rename_fn
        self.analyzer = None  # 外部传入AST分析器

    def set_analyzer(self, analyzer):
        """绑定AST分析器，用于作用域检查"""
        self.analyzer = analyzer

    def _generate_random_identifier(self, min_len=6, max_len=12) -> str:
        """生成【合法、随机】的C/C++标识符：如 tejbsja、_x7s2k"""
        # 首字符：字母/下划线
        first_char = random.choice(string.ascii_lowercase + '_')
        # 剩余字符：字母/数字/下划线
        rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_')
                             for _ in range(random.randint(min_len, max_len)))
        rand_name = first_char + rest_chars
        # 确保是合法标识符
        return rand_name if is_valid_identifier(rand_name) else self._generate_random_identifier()

    def attack_sample(self, code: str, ground_truth: int, target_model: str):
        """对单条代码执行随机改名攻击"""
        # 1. 获取原始预测
        orig_probs, orig_pred = self.model_zoo.predict(code, target_model)
        # 仅对模型预测正确的样本攻击
        if orig_pred != ground_truth:
            return False, code, orig_probs, orig_pred

        # 2. 提取所有可修改变量
        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)
        if not identifiers:
            return False, code, orig_probs, orig_pred

        # 3. 为每个变量生成**唯一、无冲突**的随机名称
        renaming_map = {}
        used_names = set(identifiers.keys())  # 避免重名

        for var_name in identifiers.keys():
            # 循环生成，直到找到无冲突的随机名
            while True:
                rand_name = self._generate_random_identifier()
                # 检查：未使用 + 无作用域冲突
                if rand_name not in used_names and self.analyzer.can_rename_to(code_bytes, var_name, rand_name):
                    renaming_map[var_name] = rand_name
                    used_names.add(rand_name)
                    break

        # 4. 执行随机改名
        try:
            adv_code = self.rename_fn(code, renaming_map)
        except:
            return False, code, orig_probs, orig_pred

        # 5. 预测对抗样本
        adv_probs, adv_pred = self.model_zoo.predict(adv_code, target_model)
        # 攻击成功：预测翻转
        attack_success = (adv_pred != orig_pred)

        return attack_success, adv_code, adv_probs, adv_pred

    def attack(self, dataset: List[Dict]):
        """执行批量随机改名攻击，统计ASR"""
        stats = {model: {"total": 0, "fooled": 0} for model in self.model_names}
        adv_samples = {model: [] for model in self.model_names}

        print("\n" + "="*80)
        print("🔍 开始执行【第一部分：随机改名攻击】")
        print("="*80)

        for idx, sample in enumerate(dataset):
            code = sample["code"]
            ground_truth = sample["label"]

            for target_model in self.model_names:
                # 执行攻击
                success, adv_code, adv_probs, adv_pred = self.attack_sample(code, ground_truth, target_model)
                stats[target_model]["total"] += 1

                if success:
                    stats[target_model]["fooled"] += 1
                    adv_samples[target_model].append({
                        "original_code": code,
                        "adversarial_code": adv_code,
                        "label": ground_truth
                    })
                    print(f"[Sample {idx+1}] {target_model} | 随机改名 ✅ 攻击成功")
                else:
                    print(f"[Sample {idx+1}] {target_model} | 随机改名 ❌ 攻击失败")

        # 打印随机攻击ASR结果
        self.print_summary(stats)

        asr_matrix = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        return asr_matrix  # 返回字典结果

    def print_summary(self, stats):
        print("\n" + "="*60)
        print("📊 随机改名攻击 - 攻击成功率(ASR)")
        print("="*60)
        for model, res in stats.items():
            asr = (res["fooled"] / res["total"] * 100) if res["total"] > 0 else 0.0
            print(f"{model:<15} | ASR: {asr:.2f}% ({res['fooled']}/{res['total']})")
        print("="*60 + "\n")