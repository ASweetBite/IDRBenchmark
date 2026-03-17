import os
import json
from typing import List, Dict
from datasets import load_dataset


class DatasetLoader:
    @staticmethod
    def load_json(filepath: str, max_samples=1000) -> List[Dict]:
        """
        如果文件存在则加载；如果不存在，则从 HuggingFace 自动下载并转换为 list of dict 格式。
        """
        # 1. 如果有本地文件，优先加载
        if os.path.exists(filepath):
            print(f"[*] Loading dataset from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[:max_samples]

        # 2. 如果没有文件，自动下载 CodeXGLUE 漏洞检测数据集 (Devign)
        print(f"[!] Dataset {filepath} not found.")
        print(f"[*] Downloading CodeXGLUE Defect Detection (Devign) from HuggingFace...")

        # 加载测试集 (split="test")，通常用于评估和攻击
        ds = load_dataset("code_x_glue_cc_defect_detection", split="test", trust_remote_code=True)

        # 3. 数据标准化
        processed_data = []
        # 只取前 max_samples
        for i in range(min(len(ds), max_samples)):
            item = ds[i]
            # 统一字段名：code, label
            processed_data.append({
                "code": item["func"],  # 原数据集中的字段是 'func'
                "label": int(item["target"])  # 原数据集中的字段是 'target'
            })

        print(f"[*] Successfully loaded {len(processed_data)} samples.")
        return processed_data