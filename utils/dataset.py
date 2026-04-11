import pandas as pd
import os
import json
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.mode = "binary"  # 默认为二分类

        self.safe_flags = [
            "", "none", "0", "safe", "nan", "null", "false",
            "<null>", "<na>"
        ]

    def load_parquet_dataset(self, filepath: str, mode: str = "binary", max_samples: int = None,
                             random_seed: int = 50, label_map_path: Optional[str] = None) -> List[Dict]:
        """
        加载数据，支持随机采样、标签映射与数据清洗。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.mode = mode
        print(f"\n[*] Loading dataset in '{mode}' mode from {filepath}...")

        # 1. 读取 Parquet
        df = pd.read_parquet(filepath)
        if 'func' not in df.columns or 'cwe' not in df.columns:
            raise ValueError("Parquet file must contain 'func' and 'cwe' columns.")

        # ========================================================
        # [新增核心过滤 1]：基础代码过滤 (过滤掉单行或空代码)
        # ========================================================
        def _line_count(s):
            return len([l for l in str(s).splitlines() if l.strip()])

        initial_count = len(df)
        df = df[df["func"].apply(_line_count) > 1].copy()

        # 统一转为字符串并去除首尾空格
        df["cwe"] = df["cwe"].fillna("").astype(str).str.strip()

        print(f"[*] 数据清洗: 过滤了 {initial_count - len(df)} 条单行或空代码，剩余 {len(df)} 条有效样本。")
        # ========================================================

        processed_data = []

        # 2. 标签处理
        if self.mode == "binary":
            # ========================================================
            # [新增核心过滤 2]：使用严谨的 safe_flags 判断
            # ========================================================
            df['label'] = df['cwe'].apply(lambda x: 0 if str(x).lower() in self.safe_flags else 1)

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
            # [新增核心过滤 2]：同样应用于 Multi 模式
            df['label_raw'] = df['cwe'].apply(lambda x: "Safe" if str(x).lower() in self.safe_flags else x)

            if label_map_path and os.path.exists(label_map_path):
                # A. 存在预定义映射表：读取并强制对齐
                print(f"[*] Loading existing label map from {label_map_path}...")
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                    # 兼容 Hugging Face 的标准格式 (带有 "id2label" 嵌套)
                    if "id2label" in loaded_data:
                        raw_map = loaded_data["id2label"]
                    else:
                        # 兼容普通的扁平 JSON 格式
                        raw_map = loaded_data

                    self.label_map = {int(k): v for k, v in raw_map.items()}

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
                    os.makedirs(os.path.dirname(label_map_path) or '.', exist_ok=True)
                    with open(label_map_path, 'w', encoding='utf-8') as f:
                        # 统一保存为标准的 Hugging Face 嵌套格式，保持生态一致性
                        save_data = {
                            "id2label": self.label_map,
                            "label2id": {v: k for k, v in self.label_map.items()}
                        }
                        json.dump(save_data, f, indent=4, ensure_ascii=False)
                    print(f"[*] Saved new label map to {label_map_path}")

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
                "raw_cwe": row["cwe"]  # 这里的 cwe 是清理过后的字符串
            })

        print(f"[*] Successfully processed {len(processed_data)} samples.")
        return processed_data

    def get_label_map(self) -> Dict:
        return self.label_map