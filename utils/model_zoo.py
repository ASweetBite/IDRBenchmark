import logging
import os
import random
from collections import Counter
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.bert_loader import CodeBERTModelLoader

logger = logging.getLogger(__name__)


class CodeSmoother:
    def __init__(self, config: Dict, candidate_generator):
        """Initializes the smoother with Monte Carlo sampling parameters."""
        self.num_samples = config.get("num_samples", 50)
        self.variance_threshold = config.get("variance_threshold", 0.05)
        self.replace_prob = config.get("replace_prob", 0.5)
        self.batch_size = config.get("batch_size", 32)
        self.candidate_generator = candidate_generator
        self.analyzer = IdentifierAnalyzer()

    def generate_smoothed_samples(self, code: str, candidate_dict: dict = None, sensitive_vars: list = None) -> List[
        str]:
        """Generates batch Monte Carlo variants of the input code for randomized smoothing."""
        code_bytes = code.encode("utf-8")
        try:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        except Exception as e:
            logger.warning(f"AST parsing failed, returning original code: {e}")
            return [code] * self.num_samples

        if not identifiers:
            return [code] * self.num_samples

        samples = []
        for _ in range(self.num_samples):
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


class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict, smoother=None):
        """Initializes the ModelZoo by batch loading target models and injecting the smoother."""
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.num_classes = run_cfg.get('num_classes', 16)
        self.max_seq_len = run_cfg.get('max_seq_len', 512)
        self.use_majority_voting = run_cfg.get('use_majority_voting', False)

        self.models = {}
        self.model_names = list(model_configs.keys())
        self.smoother = smoother

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                continue

            try:
                if os.path.exists(os.path.join(path, "dual_heads.pt")):
                    print(f"[*] Dual-head model detected, initializing loader...")
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
                    print(f"[*] Loading standard HF classifier...")
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(
                        self.device)
                    model.eval()
                    self.models[name] = {"type": "standard", "tokenizer": tokenizer, "model": model}
            except Exception as e:
                print(f"[!] Error loading {name}: {e}")

    def _base_predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        """Performs raw single-inference logic without smoothing or voting."""
        m = self.models.get(target_model)
        if m is None:
            return [0.5, 0.5], 0

        if m["type"] == "dual_head":
            res = m["model_obj"].predict(code)
            if self.eval_mode == "binary":
                p_vuln = res["f_det"]
                probs = [1.0 - p_vuln, p_vuln]
                pred_label = 1 if p_vuln > 0.5 else 0
            else:
                probs = res["f_cls"]
                pred_label = int(np.argmax(probs))
            return probs, pred_label
        else:
            inputs = m["tokenizer"](
                code, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = m["model"](**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
                pred_label = int(np.argmax(probs))
            return probs, pred_label

    def _base_batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        """Performs raw batch-inference logic without smoothing or voting."""
        m = self.models.get(target_model)
        if m is None:
            return [[0.5, 0.5] for _ in codes], [0] * len(codes)

        if m["type"] == "dual_head":
            res = m["model_obj"].batch_predict(codes, batch_size=batch_size)
            if self.eval_mode == "binary":
                f_det = res["f_det"]
                probs = [[1.0 - float(p), float(p)] for p in f_det]
                preds = [1 if p > 0.5 else 0 for p in f_det]
            else:
                f_cls = res["f_cls"]
                probs = f_cls.tolist()
                preds = [int(np.argmax(p)) for p in f_cls]
            return probs, preds
        else:
            all_probs, all_preds = [], []
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]
                inputs = m["tokenizer"](
                    batch_codes, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
                ).to(self.device)

                with torch.no_grad():
                    outputs = m["model"](**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                    probs = [probs.tolist()] if probs.ndim == 1 else probs.tolist()
                    preds = [int(np.argmax(p)) for p in probs]
                    all_probs.extend(probs)
                    all_preds.extend(preds)
            return all_probs, all_preds

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        """Predicts the label for a single code snippet, applying majority voting if enabled."""
        if self.use_majority_voting and self.smoother:
            samples = self.smoother.generate_smoothed_samples(code)
            probs_list, preds_list = self._base_batch_predict(samples, target_model,
                                                              batch_size=self.smoother.batch_size)
            majority_class = Counter(preds_list).most_common(1)[0][0]
            avg_probs = np.mean(probs_list, axis=0).tolist()
            return avg_probs, majority_class
        return self._base_predict(code, target_model)

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        """Predicts labels for a batch of code snippets using efficient flattened inference."""
        if self.use_majority_voting and self.smoother:
            all_samples = []
            for code in codes:
                all_samples.extend(self.smoother.generate_smoothed_samples(code))

            all_probs, all_preds = self._base_batch_predict(all_samples, target_model, batch_size=batch_size)

            final_probs, final_preds = [], []
            num_samples = self.smoother.num_samples

            for i in range(len(codes)):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                group_preds = all_preds[start_idx:end_idx]
                group_probs = all_probs[start_idx:end_idx]

                final_preds.append(Counter(group_preds).most_common(1)[0][0])
                final_probs.append(np.mean(group_probs, axis=0).tolist())

            return final_probs, final_preds
        return self._base_batch_predict(codes, target_model, batch_size)

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        """Retrieves confidence for a specific label, inheriting the current prediction logic."""
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0

    def predict_with_rejection(self, code: str, target_model: str, candidate_dict: dict = None,
                               sensitive_vars: list = None) -> Tuple[Union[int, str], float, float]:
        """Predicts a label with a rejection mechanism for high-variance adversarial samples."""
        if not self.smoother:
            raise ValueError("[!] Smoother not initialized. Provide smoother_config and generator in ModelZoo.")

        m = self.models.get(target_model)
        if m is None:
            return 0, 0.0, 0.0

        samples = self.smoother.generate_smoothed_samples(code, candidate_dict, sensitive_vars)
        N = len(samples)

        if m["type"] == "dual_head":
            res = m["model_obj"].batch_predict(samples, batch_size=self.smoother.batch_size)
            f_det, f_cls = res["f_det"], res["f_cls"]
            raw_probs = np.zeros((N, f_cls.shape[1] + 1))
            raw_probs[:, 0] = 1.0 - f_det
            raw_probs[:, 1:] = f_det[:, np.newaxis] * f_cls
        else:
            batch_probs, _ = self._base_batch_predict(samples, target_model, batch_size=self.smoother.batch_size)
            raw_probs = np.array(batch_probs)

        predictions = np.argmax(raw_probs, axis=1)
        majority_class, count = Counter(predictions.tolist()).most_common(1)[0]
        confidence = count / N
        variance = float(np.var(raw_probs[:, majority_class], ddof=1)) if N > 1 else 0.0

        if variance > self.smoother.variance_threshold:
            return "Reject_Adversarial", confidence, variance

        if self.eval_mode == "binary" and majority_class > 0:
            majority_class = 1

        return int(majority_class), confidence, variance