import pandas as pd
import os
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.mode = "binary"  # 默认为二分类

    def load_parquet_dataset(self, filepath: str, mode: str = "binary", max_samples: int = None,
                             random_seed: int = 50) -> List[Dict]:
        """
        加载数据，支持随机采样。
        - 针对 binary 模式：自动执行 1:2 (Safe:Vulnerable) 的降采样平衡。
        - max_samples: 如果为 None，则加载全部平衡后的数据；否则按 1:2 比例随机抽取指定数量样本。
        - random_seed: 随机种子，确保多次运行实验时抽取的样本一致。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.mode = mode
        print(f"\n[*] Loading dataset in '{mode}' mode from {filepath}...")

        # 1. 读取 Parquet
        df = pd.read_parquet(filepath)
        if 'func' not in df.columns or 'cwe' not in df.columns:
            raise ValueError("Parquet file must contain 'func' and 'cwe' columns.")

        processed_data = []

        # 2. 核心修改：标签处理与 1:2 数据平衡策略
        if self.mode == "binary":
            df['label'] = df['cwe'].apply(lambda x: 0 if (not x or x == "") else 1)
            safe_df = df[df['label'] == 0]
            vuln_df = df[df['label'] == 1]

            print(f"[*] 原始数据分布: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

            # 确定最终需要的 Safe 和 Vuln 数量
            if max_samples and max_samples < len(df):
                # 如果指定了上限，按照 1:2 的比例分配
                safe_needed = max_samples // 2
                vuln_needed = max_samples - safe_needed
            else:
                # 如果没有指定上限，按照整体 1:2 的比例最大化利用 Safe 数据
                safe_needed = len(safe_df)
                vuln_needed = safe_needed * 2

            # 防御性编程：确保我们要求的数量不会超过实际拥有的数据量
            safe_needed = min(safe_needed, len(safe_df))
            vuln_needed = min(vuln_needed, len(vuln_df), safe_needed * 2)

            print(f"[*] 平衡后抽取 (1:2): Safe={safe_needed}, Vuln={vuln_needed} (Seed={random_seed})")

            # 分别进行随机采样
            safe_sampled = safe_df.sample(n=safe_needed, random_state=random_seed)
            vuln_sampled = vuln_df.sample(n=vuln_needed, random_state=random_seed)

            # 合并两部分数据，并再次打乱顺序 (frac=1)
            df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
            self.label_map = {0: "Safe", 1: "Vulnerable"}

        elif self.mode == "multi":
            # 多分类逻辑保持原样
            df['label_raw'] = df['cwe'].apply(lambda x: "Safe" if (not x or x == "") else x)
            encoded_labels = self.label_encoder.fit_transform(df['label_raw'])
            df['label'] = encoded_labels
            self.label_map = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}

            # 多分类的随机采样
            if max_samples and max_samples < len(df):
                print(f"[*] 正在从 {len(df)} 条多分类数据中随机采样 {max_samples} 条 (seed={random_seed})...")
                df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
        else:
            raise ValueError("Mode must be 'binary' or 'multi'")

        # 3. 构建统一数据列表
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