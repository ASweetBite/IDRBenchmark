import os
import re
import json
import random
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. AST 分析与转换工具 (保持不变)
# ==========================================
from tree_sitter import Language, Parser
import tree_sitter_c


class IdentifierAnalyzer:
    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.keywords = {
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen"
        }

    def extract_identifiers(self, source_code: bytes) -> dict:
        tree = self.parser.parse(source_code)
        identifiers = defaultdict(list)
        scope_stack = [0]
        scope_counter = 0

        def traverse(node):
            nonlocal scope_counter
            if node.type == 'compound_statement':
                scope_counter += 1
                scope_stack.append(scope_counter)

            if node.type == "identifier":
                parent_type = node.parent.type if node.parent else None
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                if name not in self.keywords and \
                        parent_type != 'function_declarator' and \
                        parent_type != 'field_identifier':
                    identifiers[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "scope": scope_stack[-1]
                    })
            for child in node.children:
                traverse(child)

            if node.type == 'compound_statement':
                scope_stack.pop()

        traverse(tree.root_node)
        return dict(identifiers)


def is_valid_identifier(name: str) -> bool:
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict) -> str:
        existing_names = set(identifiers.keys())
        for old_name, new_name in renaming_map.items():
            if not is_valid_identifier(new_name):
                raise ValueError(f"命名不合法: '{new_name}'")
            if new_name in existing_names and new_name != old_name:
                raise ValueError(f"重命名冲突: '{old_name}' -> '{new_name}' 已存在。")

        code = bytearray(source_code)
        replacements = []
        for old_name, new_name in renaming_map.items():
            if old_name in identifiers:
                for pos in identifiers[old_name]:
                    replacements.append((pos['start'], pos['end'], new_name))

        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, new_name in replacements:
            code[start:end] = new_name.encode("utf-8")

        return code.decode("utf-8")


# ==========================================
# 2. 多模型管理中心 (Model Zoo) - 【新增】
# ==========================================
class ModelZoo:
    """
    统一管理多个模型，支持 CodeBERT, CodeT5, VulBERTa 等。
    """

    def __init__(self, model_configs: Dict[str, str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_names = list(model_configs.keys())

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path} on {self.device}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
                model.eval()
                self.models[name] = {"tokenizer": tokenizer, "model": model}
            except Exception as e:
                print(f"[!] Warning: Failed to load {name} from {path}. Error: {e}")
                print(f"    Fallback to random predictions for {name} (Debugging Mode).")
                self.models[name] = None  # 用于本地无模型时占位

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        m = self.models.get(target_model)

        # 本地调试占位逻辑（当你没下载模型时）
        if m is None:
            probs = [0.1, 0.9] if "strcpy" in code else [0.9, 0.1]
            return probs, int(np.argmax(probs))

        inputs = m["tokenizer"](
            code, return_tensors="pt", truncation=True, max_length=512, padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = m["model"](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
            pred_label = int(np.argmax(probs))
        return probs, pred_label

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        return probs[label]


# ==========================================
# 3. RNNS 排序 (适配目标模型)
# ==========================================
class RNNS_Ranker:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label):
        orig_prob = self.model_zoo.predict_label_conf(code, reference_label, self.target_model)
        scores = []

        for var in variables:
            candidates = subs_pool.get(var, [])
            if not candidates: continue

            test_sub = random.choice(candidates)
            renamed_code = self.rename_fn(code, {var: test_sub})
            new_prob = self.model_zoo.predict_label_conf(renamed_code, reference_label, self.target_model)

            prob_drop = orig_prob - new_prob
            scores.append((var, prob_drop))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [var for var, score in scores]


# ==========================================
# 4. GA 优化 (适配目标模型)
# ==========================================
class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, pop_size=10, max_generations=10):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = max_generations

    def run(self, code, original_pred, target_vars, subs_pool):
        population = [{var: var for var in target_vars}]
        for _ in range(self.pop_size - 1):
            population.append({v: random.choice(subs_pool.get(v, [v]) + [v]) for v in target_vars})

        best_code = code
        best_fitness = -1

        for gen in range(self.max_generations):
            evaluated = []
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                mutated_code = self.rename_fn(code, rename_map)
                probs, pred = self.model_zoo.predict(mutated_code, self.target_model)

                fitness = 1.0 - probs[original_pred]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_code = mutated_code

                if pred != original_pred:
                    print(f"    [+] Target Model ({self.target_model}) Attack Succeeded at Gen {gen}!")
                    return True, mutated_code, probs, pred

            evaluated.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in evaluated[:max(2, self.pop_size // 2)]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = {v: p1[v] if random.random() > 0.5 else p2[v] for v in target_vars}
                for v in child:
                    if random.random() < 0.2:
                        child[v] = random.choice(subs_pool.get(v, [v]) + [v])
                new_pop.append(child)
            population = new_pop

        final_probs, final_pred = self.model_zoo.predict(best_code, self.target_model)
        return False, best_code, final_probs, final_pred


# ==========================================
# 5. IRTG 攻击器
# ==========================================
class IRTG_Attacker:
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


# ==========================================
# 6. 数据集加载器 (适配 MegaVul, Juliet) - 【新增】
# ==========================================
class DatasetLoader:
    @staticmethod
    def load_json(filepath: str, max_samples=100) -> List[Dict]:
        """
        假设数据集是 list of dict: [{"code": "...", "label": 1}, ...]
        如果文件不存在，返回一个演示用的默认数据集。
        """
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[:max_samples]
        else:
            print(f"[!] Dataset {filepath} not found. Using default dummy dataset.")
            return [
                {"code": "void test(char *str) { char buf[50]; strcpy(buf, str); }", "label": 1},
                {"code": "void safe(char *str) { char buf[50]; strncpy(buf, str, 49); }", "label": 0}
            ]


# ==========================================
# 7. 多模型迁移性评估器 - 【重构核心】
# ==========================================
class TransferabilityEvaluator:
    def __init__(self, attacker: IRTG_Attacker, model_zoo: ModelZoo):
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

            print(f"\n{'=' * 60}\n[Sample {idx + 1}/{len(dataset)}] Label: {true_label}")

            # 1. 记录所有模型对原代码的预测
            orig_preds = {}
            for m in self.model_zoo.model_names:
                _, pred = self.model_zoo.predict(code, m)
                orig_preds[m] = pred
                if pred == true_label:
                    stats[m]["correct_orig"] += 1

            # 如果白盒目标模型就预测错了，直接跳过攻击（不纳入基数）
            if orig_preds[self.target_model] != true_label:
                print(f"  [-] Target model ({self.target_model}) misclassified originally. Skip.")
                continue

            total += 1

            # 2. 对 Target Model 执行攻击
            print(f"  [*] Attacking Target Model: {self.target_model} ...")
            res = self.attacker.attack(code)
            adv_code = res["adv_code"]

            # 3. 评估跨模型迁移性
            print(f"  [*] Transferability Test:")
            for m in self.model_zoo.model_names:
                if orig_preds[m] != true_label:
                    continue  # 原本就预测错的模型不计入被欺骗次数

                _, adv_pred = self.model_zoo.predict(adv_code, m)

                if adv_pred != true_label:
                    stats[m]["fooled"] += 1
                    status = "✅ FOOLED"
                else:
                    status = "❌ ROBUST"

                model_type = "Target" if m == self.target_model else "Transfer"
                print(f"      - [{model_type}] {m}: Orig={orig_preds[m]} -> Adv={adv_pred} ({status})")

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


# ==========================================
# 8. Main 函数适配
# ==========================================
def main():
    analyzer = IdentifierAnalyzer()
    transformer = CodeTransformer()

    def get_all_vars_fn(code_str: str) -> List[str]:
        return list(analyzer.extract_identifiers(code_str.encode("utf-8")).keys())

    def get_subs_pool_fn(code_str: str, variables: List[str]) -> Dict[str, List[str]]:
        pool = {}
        common = ["idx", "tmp", "data", "val", "ptr", "buffer_obj"]
        for var in variables:
            cands = [f"{var}_tmp", f"tmp_{var}", random.choice(common)]
            pool[var] = [c for c in set(cands) if is_valid_identifier(c) and c != var]
        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        try:
            safe_map = {k: v for k, v in renaming_map.items() if k in ids and k != v and is_valid_identifier(v)}
            return transformer.validate_and_apply(code_bytes, ids, safe_map) if safe_map else code_str
        except ValueError:
            return code_str

    # 1. 配置你的模型路径 (这里换成你本地实际下载微调好的模型路径)
    model_configs = {
        "CodeBERT": "./models/codebert_finetuned",
        "CodeT5": "./models/codet5_finetuned",
        "VulBERTa": "./models/vulberta_finetuned"
    }

    # 2. 初始化 ModelZoo
    model_zoo = ModelZoo(model_configs)

    # 3. 指定攻击目标 (白盒攻击哪个模型？)
    TARGET_MODEL = "CodeBERT"

    # 4. 初始化攻击器与评估器
    attacker = IRTG_Attacker(model_zoo, TARGET_MODEL, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=3)
    evaluator = TransferabilityEvaluator(attacker, model_zoo)

    # 5. 读取数据集 (MegaVul / Juliet)
    dataset = DatasetLoader.load_json("./data/megavul_test.json", max_samples=5)

    # 6. 开始多模型迁移评估
    evaluator.evaluate(dataset)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()