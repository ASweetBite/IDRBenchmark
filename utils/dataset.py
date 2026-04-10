import pandas as pd
import os
import json # [新增]
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.mode = "binary"  # 默认为二分类

    def load_parquet_dataset(self, filepath: str, mode: str = "binary", max_samples: int = None,
                             random_seed: int = 50, label_map_path: Optional[str] = None) -> List[Dict]:
        """
        加载数据，支持随机采样和标签映射。
        - label_map_path: [新增] json 文件路径。如果存在则读取并强制映射；如果不存在则基于当前数据生成并保存。
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

        # 2. 标签处理
        if self.mode == "binary":
            df['label'] = df['cwe'].apply(lambda x: 0 if (not x or x == "") else 1)
            safe_df = df[df['label'] == 0]
            vuln_df = df[df['label'] == 1]

            print(f"[*] 原始数据分布: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

            if max_samples and max_samples < len(df):
                safe_needed = max_samples // 3
                vuln_needed = max_samples - safe_needed
            else:
                safe_needed = len(safe_df)
                vuln_needed = safe_needed * 2

            safe_needed = min(safe_needed, len(safe_df))
            vuln_needed = min(vuln_needed, len(vuln_df), safe_needed * 2)

            print(f"[*] 平衡后抽取 (1:2): Safe={safe_needed}, Vuln={vuln_needed} (Seed={random_seed})")

            safe_sampled = safe_df.sample(n=safe_needed, random_state=random_seed)
            vuln_sampled = vuln_df.sample(n=vuln_needed, random_state=random_seed)

            df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
            self.label_map = {0: "Safe", 1: "Vulnerable"}

        elif self.mode == "multi":
            df['label_raw'] = df['cwe'].apply(lambda x: "Safe" if (not x or x == "") else x)

            # ========================================================
            # [核心修改]：Multi 模式的 JSON 读取与保存逻辑
            # ========================================================
            if label_map_path and os.path.exists(label_map_path):
                # A. 存在预定义映射表：读取并强制对齐
                print(f"[*] Loading existing label map from {label_map_path}...")
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    loaded_map = json.load(f)
                    # 注意：JSON 序列化时 key 都会变成 string，加载时需要转回 int
                    self.label_map = {int(k): v for k, v in loaded_map.items()}

                # 构建反向映射表 {CWE: ID}
                cwe_to_id = {v: k for k, v in self.label_map.items()}

                # 过滤掉模型未见过 (不在映射表中) 的脏数据
                valid_mask = df['label_raw'].isin(cwe_to_id.keys())
                dropped_count = len(df) - valid_mask.sum()
                if dropped_count > 0:
                    print(f"[!] Warning: Dropped {dropped_count} samples due to unknown CWEs not in the label map.")
                    df = df[valid_mask].reset_index(drop=True)

                # 执行硬性映射
                df['label'] = df['label_raw'].map(cwe_to_id)

            else:
                # B. 不存在映射表：基于当前数据集重新生成
                print("[*] Generating new label mapping from current dataset...")
                encoded_labels = self.label_encoder.fit_transform(df['label_raw'])
                df['label'] = encoded_labels
                self.label_map = {int(i): cls for i, cls in enumerate(self.label_encoder.classes_)}

                # 如果传入了路径，将其保存为 JSON
                if label_map_path:
                    # 确保保存的目录存在
                    os.makedirs(os.path.dirname(label_map_path) or '.', exist_ok=True)
                    with open(label_map_path, 'w', encoding='utf-8') as f:
                        json.dump(self.label_map, f, indent=4, ensure_ascii=False)
                    print(f"[*] Saved new label map to {label_map_path}")
            # ========================================================

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