import logging
import os
import random
from collections import Counter
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.bert_loader import CodeBERTModelLoader

logger = logging.getLogger(__name__)


# ==========================================
# 1. 独立抽离的平滑器组件 (作为 ModelZoo 的内部模块)
# ==========================================
class CodeSmoother:
    def __init__(self, config: Dict, candidate_generator):
        self.num_samples = config.get("num_samples", 50)
        self.variance_threshold = config.get("variance_threshold", 0.05)
        self.replace_prob = config.get("replace_prob", 0.5)
        self.batch_size = config.get("batch_size", 32)

        self.candidate_generator = candidate_generator
        self.analyzer = IdentifierAnalyzer()

    def generate_smoothed_samples(self, code: str, candidate_dict: dict = None, sensitive_vars: list = None) -> List[
        str]:
        """批量生成蒙特卡洛变体样本"""
        code_bytes = code.encode("utf-8")
        try:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        except Exception as e:
            logger.warning(f"AST 解析失败，返回原样: {e}")
            return [code] * self.num_samples

        if not identifiers:
            return [code] * self.num_samples

        samples = []
        for _ in range(self.num_samples):
            # 确定要替换的标识符 (靶向 or 随机)
            if sensitive_vars:
                targets = [v for v in identifiers if v in sensitive_vars and random.random() < self.replace_prob]
            else:
                targets = [v for v in identifiers if random.random() < self.replace_prob]

            if not targets:
                samples.append(code)
                continue

            rename_map = {}
            for t in targets:
                if candidate_dict and t in candidate_dict and candidate_dict[t]:
                    rename_map[t] = random.choice(candidate_dict[t])
                else:
                    cands = self.candidate_generator.get_random_replacement(code, [t])
                    if cands and t in cands:
                        rename_map[t] = cands[t]

            if not rename_map:
                samples.append(code)
            else:
                transformed = CodeTransformer.validate_and_apply(code_bytes, identifiers, rename_map, self.analyzer)
                samples.append(transformed if transformed else code)

        return samples


# ==========================================
# 2. 整合后的 ModelZoo
# ==========================================
class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict, smoother=None):
        # 1. 基础环境设置
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.num_classes = run_cfg.get('num_classes', 16)
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        self.models = {}
        self.model_names = list(model_configs.keys())
        self.smoother = smoother  # 已注入的平滑器

        # 2. 批量加载被攻击的模型
        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                continue

            try:
                # 检查是否是双头模型
                if os.path.exists(os.path.join(path, "dual_heads.pt")):
                    print(f"[*] 检测到双头模型，正在初始化加载器...")
                    # 动态构建 loader 配置
                    loader_cfg = {
                        "model": {
                            "model_name": path,
                            "max_seq_len": self.max_seq_len,
                            "device": str(self.device)
                        },
                        "data": {"num_classes": self.num_classes}
                    }
                    loader = CodeBERTModelLoader(loader_cfg)
                    model_obj, _ = loader.load_model()
                    self.models[name] = {"type": "dual_head", "model_obj": model_obj}
                else:
                    print(f"[*] 加载标准 HF 分类器...")
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(
                        self.device)
                    model.eval()
                    self.models[name] = {"type": "standard", "tokenizer": tokenizer, "model": model}
            except Exception as e:
                print(f"[!] Error loading {name}: {e}")

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        m = self.models.get(target_model)
        if m is None:
            return [0.5, 0.5], 0

        # ========== 处理双头模型 ==========
        if m["type"] == "dual_head":
            res = m["model_obj"].predict(code)

            if self.eval_mode == "binary":
                # f_det 是漏洞类别(1类)的概率。我们需要补齐 Safe(0类) 的概率以对齐接口
                p_vuln = res["f_det"]
                probs = [1.0 - p_vuln, p_vuln]
                pred_label = 1 if p_vuln > 0.5 else 0
            else:
                # 多分类模式，直接返回所有的类别概率
                probs = res["f_cls"]
                pred_label = int(np.argmax(probs))

            return probs, pred_label

        # ========== 处理标准模型 ==========
        else:
            inputs = m["tokenizer"](
                code, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = m["model"](**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
                pred_label = int(np.argmax(probs))
            return probs, pred_label

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        # 防止因模型错误导致数组越界
        if label < len(probs):
            return probs[label]
        return 0.0

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        m = self.models.get(target_model)
        if m is None:
            return [[0.5, 0.5] for _ in codes], [0] * len(codes)

        # ========== 处理双头模型 ==========
        if m["type"] == "dual_head":
            # 直接调用你的 batch_predict
            res = m["model_obj"].batch_predict(codes, batch_size=batch_size)

            if self.eval_mode == "binary":
                f_det = res["f_det"]  # shape: (N,)
                probs = [[1.0 - float(p), float(p)] for p in f_det]
                preds = [1 if p > 0.5 else 0 for p in f_det]
            else:
                f_cls = res["f_cls"]  # shape: (N, num_classes)
                probs = f_cls.tolist()
                preds = [int(np.argmax(p)) for p in f_cls]

            return probs, preds

        # ========== 处理标准模型 ==========
        else:
            all_probs = []
            all_preds = []

            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]
                inputs = m["tokenizer"](
                    batch_codes, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
                ).to(self.device)

                with torch.no_grad():
                    outputs = m["model"](**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                    if probs.ndim == 1:
                        probs = [probs.tolist()]
                    else:
                        probs = probs.tolist()

                    preds = [int(np.argmax(p)) for p in probs]

                    all_probs.extend(probs)
                    all_preds.extend(preds)

            return all_probs, all_preds

    def predict_with_rejection(self, code: str, target_model: str,
                               candidate_dict: dict = None, sensitive_vars: list = None) -> Tuple[
        Union[int, str], float, float]:
        """
        供鲁棒性评测脚本调用的核心防御接口。
        返回: (预测标签/拒识标志, 置信度, 预测方差)
        """
        if not self.smoother:
            raise ValueError("[!] Smoother 未初始化，请在实例化 ModelZoo 时提供 smoother_config 和 candidate_generator。")

        m = self.models.get(target_model)
        if m is None:
            return 0, 0.0, 0.0

        # 1. 生成 N 个平滑变体
        samples = self.smoother.generate_smoothed_samples(code, candidate_dict, sensitive_vars)
        N = len(samples)

        # 2. 批量获取底层推断结果并融合概率
        if m["type"] == "dual_head":
            # 获取底层的原始 f_det 和 f_cls
            res = m["model_obj"].batch_predict(samples, batch_size=self.smoother.batch_size)
            f_det = res["f_det"]  # shape: [N]
            f_cls = res["f_cls"]  # shape: [N, C]

            C = f_cls.shape[1]
            raw_probs = np.zeros((N, C + 1))
            raw_probs[:, 0] = 1.0 - f_det  # 第 0 类：P(Safe)
            raw_probs[:, 1:] = f_det[:, np.newaxis] * f_cls  # 第 1~C 类：P(Vuln 且是类型 K)

        else:
            # 如果是标准单头模型，直接调用 ModelZoo 的 batch_predict 获取统一的 probs
            batch_probs, _ = self.batch_predict(samples, target_model, batch_size=self.smoother.batch_size)
            raw_probs = np.array(batch_probs)  # 此时对于二分类 shape为 [N, 2]，多分类为 [N, C]

        # 3. 多数表决 (Majority Voting)
        predictions = np.argmax(raw_probs, axis=1)
        vote_counter = Counter(predictions.tolist())
        majority_class, count = vote_counter.most_common(1)[0]
        confidence = count / N

        # 4. 置信方差计算 (Variance Analysis)
        majority_probs = raw_probs[:, majority_class]
        variance = float(np.var(majority_probs, ddof=1)) if N > 1 else 0.0

        # 5. 拒识决策 (Rejection)
        if variance > self.smoother.variance_threshold:
            return "Reject_Adversarial", confidence, variance

        # 返回的是 C+1 融合空间下的类别。如果外部是二分类评测，1~C 统称为 1 (Vuln)
        if self.eval_mode == "binary" and majority_class > 0:
            majority_class = 1

        return int(majority_class), confidence, variance