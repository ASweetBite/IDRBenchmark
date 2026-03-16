import os
import json
from typing import List, Dict

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

