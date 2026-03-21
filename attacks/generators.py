import re
import torch
from typing import List

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier, CodeTransformer


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo, analyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer

    def generate_candidates(self, code: str, var_name: str, top_k_mlm=100, top_n_keep=20) -> List[str]:
        keywords = self.analyzer.keywords
        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        if var_name not in identifiers:
            print(f"[DEBUG-MLM] 变量 '{var_name}' 在 AST 解析出的 identifiers 中不存在。跳过。")
            return []

        # 预测掩码
        mask_token = self.model_zoo.mlm_tokenizer.mask_token
        masked_code = re.sub(r'\b' + re.escape(var_name) + r'\b', mask_token, code, count=1)

        inputs = self.model_zoo.mlm_tokenizer(
            masked_code, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model_zoo.device)

        mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
        mask_token_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        # 【重点排查点】是否因为超出 512 长度被截断？
        if len(mask_token_indices) == 0:
            print(
                f"    [DEBUG-MLM] 变量 '{var_name}' 找不到 Mask Token! (极大概率是代码太长，超出 max_length=512 被截断)。")
            return []

        with torch.no_grad():
            logits = self.model_zoo.mlm_model(**inputs).logits
            mask_logits = logits[0, mask_token_indices[0], :]
            _, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)

            raw_candidates = [
                self.model_zoo.mlm_tokenizer.decode([idx]).strip()
                for idx in top_k_indices
            ]

        valid_candidates = []

        # --- 数据统计字典 ---
        stats = {
            "invalid_format": 0,
            "is_keyword": 0,
            "is_self": 0,
            "ast_conflict": 0,
            "transform_error": 0
        }

        # 原始候选词去重
        unique_raw_cands = list(set(raw_candidates))

        for raw_cand in unique_raw_cands:
            cand = raw_cand.replace('Ġ', '').replace('##', '').strip()
            cand = re.sub(r'[^a-zA-Z0-9_]', '', cand)

            # 1. 合法性检查
            if not cand or not cand[0].isalpha() and cand[0] != '_':
                stats["invalid_format"] += 1
                continue
            # 2. 关键字黑名单
            if cand in keywords:
                stats["is_keyword"] += 1
                continue
            # 3. 避免重命名为自身
            if cand == var_name:
                stats["is_self"] += 1
                continue

            # 4. AST 作用域检查
            if not self.analyzer.can_rename_to(code_bytes, var_name, cand):
                stats["ast_conflict"] += 1
                continue

            # 5. 代码变换验证
            try:
                _ = CodeTransformer.validate_and_apply(
                    code_bytes, identifiers, {var_name: cand}, analyzer=self.analyzer
                )
                valid_candidates.append(cand)
            except Exception:
                stats["transform_error"] += 1
                continue

            # 达到数量即可退出
            if len(valid_candidates) >= top_n_keep:
                break

        # # ==========================================
        # # 打印非常详细的过滤结果日志
        # # ==========================================
        # print(f"    [DEBUG-MLM] Var: '{var_name:<10}' | 最终通过: {len(valid_candidates):>2}/{top_n_keep} "
        #       f"| 原始去重: {len(unique_raw_cands):>2} | 过滤 -> 格式非法:{stats['invalid_format']}, "
        #       f"关键字:{stats['is_keyword']}, AST冲突:{stats['ast_conflict']}, 转换报错:{stats['transform_error']}")
        #
        # # 如果通过的数量极其惨淡，把 MLM 生成的原始词打印出来看看它到底生成了些什么鬼东西
        # if len(valid_candidates) < 2:
        #     print(f"      -> [Warning] 候选词匮乏! MLM 吐出的部分生词样本: {unique_raw_cands[:10]}")

        return valid_candidates