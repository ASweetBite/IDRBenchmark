import pandas as pd
import os
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.mode = "binary"  # 默认为二分类

    def load_parquet_dataset(self, filepath: str, mode: str = "binary", max_samples: int = None, random_seed: int = 42) -> List[Dict]:
        """
        加载数据，支持随机采样。
        - max_samples: 如果为 None，则加载全部；否则随机抽取指定数量样本。
        - random_seed: 随机种子，确保多次运行实验时抽取的样本一致。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.mode = mode
        print(f"[*] Loading dataset in '{mode}' mode from {filepath}...")

        # 1. 读取 Parquet
        df = pd.read_parquet(filepath)
        if 'func' not in df.columns or 'cwe' not in df.columns:
            raise ValueError("Parquet file must contain 'func' and 'cwe' columns.")

        # 2. 随机采样逻辑 (修改点)
        if max_samples and max_samples < len(df):
            print(f"[*] 正在从 {len(df)} 条数据中随机采样 {max_samples} 条 (seed={random_seed})...")
            df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
        else:
            print(f"[*] 加载全部 {len(df)} 条数据。")

        processed_data = []

        # 3. 标签处理逻辑 (保持原样)
        if self.mode == "binary":
            df['label'] = df['cwe'].apply(lambda x: 0 if (not x or x == "") else 1)
            self.label_map = {0: "Safe", 1: "Vulnerable"}
        elif self.mode == "multi":
            df['label_raw'] = df['cwe'].apply(lambda x: "Safe" if (not x or x == "") else x)
            encoded_labels = self.label_encoder.fit_transform(df['label_raw'])
            df['label'] = encoded_labels
            self.label_map = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
        else:
            raise ValueError("Mode must be 'binary' or 'multi'")

        # 4. 构建统一数据列表
        for _, row in df.iterrows():
            processed_data.append({
                "code": row["func"],
                "label": int(row["label"]),
                "raw_cwe": row["cwe"]
            })

        print(f"[*] Successfully processed {len(processed_data)} samples.")
        return processed_data

    def get_label_map(self) -> Dict:
        return self.label_map