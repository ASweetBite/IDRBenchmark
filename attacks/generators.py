import re
import torch
from typing import List

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier, CodeTransformer


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo, analyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer

    def generate_candidates(self, code: str, target_name: str, identifiers=None, top_k_mlm=100, top_n_keep=20) -> List[
        str]:
        keywords = self.analyzer.keywords
        code_bytes = code.encode("utf-8")

        if identifiers is None:
            identifiers = self.analyzer.extract_identifiers(code_bytes)

        if target_name not in identifiers:
            return []

        target_info = identifiers[target_name][0]
        start_byte = target_info['start']
        end_byte = target_info['end']

        mask_token = self.model_zoo.mlm_tokenizer.mask_token
        protected_mask = f" {mask_token} "
        mask_token_bytes = protected_mask.encode("utf-8")

        masked_code_bytes = code_bytes[:start_byte] + mask_token_bytes + code_bytes[end_byte:]

        context_half_size = 700
        mask_start_in_masked = start_byte
        mask_end_in_masked = start_byte + len(mask_token_bytes)

        crop_start = max(0, mask_start_in_masked - context_half_size)
        crop_end = min(len(masked_code_bytes), mask_end_in_masked + context_half_size)

        cropped_code_bytes = masked_code_bytes[crop_start:crop_end]
        cropped_code = cropped_code_bytes.decode("utf-8", errors="replace")
        inputs = self.model_zoo.mlm_tokenizer(
            cropped_code, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model_zoo.device)

        mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
        mask_token_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_token_indices) == 0:
            full_inputs = self.model_zoo.mlm_tokenizer(cropped_code, return_tensors="pt").to(self.model_zoo.device)
            full_mask_indices = (full_inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

            if len(full_mask_indices) > 0:
                print(f"    [DEBUG-MLM] 确认: '{target_name}' 被截断了。Token 位置在 {full_mask_indices[0]}，超过了 512。")
            else:
                print(f"    [DEBUG-MLM] 异常: '{target_name}' 即使不截断也找不到 Mask。可能是 Tokenizer 行为异常。")
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

        stats = {
            "invalid_format": 0,
            "is_keyword": 0,
            "is_self": 0,
            "ast_conflict": 0,
            "transform_error": 0
        }

        unique_raw_cands = list(set(raw_candidates))

        for raw_cand in unique_raw_cands:
            cand = raw_cand.replace('Ġ', '').replace('##', '').strip()
            cand = re.sub(r'[^a-zA-Z0-9_]', '', cand)

            if not cand or not cand[0].isalpha() and cand[0] != '_':
                stats["invalid_format"] += 1
                continue
            if cand in keywords:
                stats["is_keyword"] += 1
                continue
            if cand == target_name:
                stats["is_self"] += 1
                continue

            if not self.analyzer.can_rename_to(code_bytes, target_name, cand):
                stats["ast_conflict"] += 1
                continue

            try:
                _ = CodeTransformer.validate_and_apply(
                    code_bytes, identifiers, {target_name: cand}, analyzer=self.analyzer
                )
                valid_candidates.append(cand)
            except Exception:
                stats["transform_error"] += 1
                continue

            if len(valid_candidates) >= top_n_keep:
                break

        # print(f"    [DEBUG-MLM] Target: '{target_name:<10}' ({entity_type}) | Valid: {len(valid_candidates):>2}/{top_n_keep} "
        #       f"| Raw: {len(unique_raw_cands):>2} | 过滤 -> 格式:{stats['invalid_format']}, "
        #       f"关键字:{stats['is_keyword']}, AST冲突:{stats['ast_conflict']}, 报错:{stats['transform_error']}")

        return valid_candidates