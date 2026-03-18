import os
import json
from typing import List, Dict
from datasets import load_dataset


class DatasetLoader:
    @staticmethod
    def load_json(filepath: str, max_samples=1000) -> List[Dict]:
        """
        优先加载本地数据集；若不存在，则下载、处理并保存到本地。
        """
        # 1. 尝试从本地加载
        if os.path.exists(filepath):
            print(f"[*] Loading dataset from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)[:max_samples]

        # 2. 如果文件不存在，从 HuggingFace 下载
        print(f"[!] Dataset {filepath} not found.")
        print(f"[*] Downloading CodeXGLUE Defect Detection (Devign) from HuggingFace...")

        ds = load_dataset("code_x_glue_cc_defect_detection", split="test", trust_remote_code=True)

        # 3. 数据标准化
        processed_data = []
        for i in range(min(len(ds), max_samples)):
            item = ds[i]
            processed_data.append({
                "code": item["func"],
                "label": int(item["target"])
            })

        # 4. 保存到本地 (确保目录存在)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)

        print(f"[*] Successfully saved {len(processed_data)} samples to {filepath}")
        return processed_data

