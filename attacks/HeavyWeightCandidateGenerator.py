import itertools
import os
import random
import re
from collections import defaultdict
from typing import List, Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.ast_tools import CodeTransformer


class HeavyWeightCandidateGenerator:
    """Generates context-aware identifier candidates using Masked Language Modeling and AST validation."""

    def __init__(self, mlm_engine, analyzer, config):
        """Initializes the generator with a language model engine, AST analyzer, and configuration."""
        self.mlm_engine = mlm_engine
        self.analyzer = analyzer
        self._embedding_cache = {}
        self.config = config

    def _detect_naming_style(self, name: str) -> str:
        """Determines the naming convention of a given identifier string."""
        if '_' in name:
            return 'SCREAMING_SNAKE' if name.isupper() else 'snake_case'
        elif name.islower():
            return 'single_lower'
        elif name.isupper():
            return 'single_upper'
        elif name[0].islower() and any(c.isupper() for c in name):
            return 'camelCase'
        elif name[0].isupper() and any(c.islower() for c in name):
            return 'PascalCase'
        return 'unknown'

    def _matches_style(self, original_style: str, candidate: str) -> bool:
        """Checks if a candidate identifier matches the required naming style."""
        cand_style = self._detect_naming_style(candidate)
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower':
            return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'):
            return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE':
            return True

        return cand_style == original_style

    def _split_identifier(self, name: str):
        """Deconstructs an identifier into its constituent word parts based on naming style."""
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name):
                return [name], ''
            return parts, 'camel'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
        """Constructs a string where specific identifier parts are replaced by mask tokens."""
        mask_list = [mask_token] * num_masks
        new_parts = parts[:start] + mask_list + parts[end:]

        if style == '_':
            return "_".join(new_parts)
        elif style == 'camel':
            res = []
            for j, p in enumerate(new_parts):
                if p == mask_token:
                    res.append(p)
                else:
                    res.append(p.lower() if j == 0 and target_name[0].islower() else p.capitalize())
            return "".join(res).replace(mask_token.capitalize(), mask_token)
        else:
            return mask_token

    def _assemble_multi_candidate(self, parts: List[str], start: int, end: int, predicted_words: tuple, style: str,
                                  target_name: str) -> str:
        """Reassembles an identifier using words predicted by the model at masked positions."""
        new_parts = parts[:start] + list(predicted_words) + parts[end:]

        if style == '_':
            return "_".join(new_parts)
        elif style == 'camel':
            res = []
            for j, p in enumerate(new_parts):
                res.append(p.lower() if j == 0 and target_name[0].islower() else p.capitalize())
            return "".join(res)
        else:
            return "".join(predicted_words)

    def _get_code_embedding_batched(self, code_snippets: List[str], batch_size: int = 32) -> torch.Tensor:
        """Performs batch inference to get sequence-level embeddings for code snippets using Mean Pooling."""
        all_embeddings = []
        tokenizer = self.mlm_engine.tokenizer

        for i in range(0, len(code_snippets), batch_size):
            batch_texts = code_snippets[i: i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.mlm_engine.device)

            with torch.no_grad():
                # Request hidden states to extract the sequence representations
                outputs = self.mlm_engine.model(**inputs, output_hidden_states=True)
                # Usually, hidden_states[-1] is the last layer's hidden states: [batch, seq_len, hidden_dim]
                last_hidden = outputs.hidden_states[-1]

                # Mean Pooling: Ignore padding tokens
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                all_embeddings.append(mean_pooled.cpu().detach())  # Offload to CPU to save VRAM

        return torch.cat(all_embeddings, dim=0)

    def _get_model_logits_batched(self, cropped_codes: List[str]):
        """Performs batch inference to obtain model logits and mask indices for multiple code snippets."""
        if not cropped_codes:
            return None, []

        inputs = self.mlm_engine.tokenizer(
            cropped_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.mlm_engine.device)

        mask_token_id = self.mlm_engine.tokenizer.mask_token_id

        with torch.no_grad():
            batch_logits = self.mlm_engine.model(**inputs).logits

        batch_mask_indices = []
        for i in range(batch_logits.size(0)):
            indices = (inputs.input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
            batch_mask_indices.append(indices)

        return batch_logits, batch_mask_indices

    def _decode_words(self, mask_logits, top_k, allow_underscore=False, required_length=None):
        """Decodes model logits into a list of candidate words filtered by style and length constraints."""
        _, top_indices = torch.topk(mask_logits, top_k, dim=-1)
        words = []
        for idx in top_indices:
            w = self.mlm_engine.tokenizer.decode([idx]).strip().replace('Ġ', '').replace('##', '')
            if allow_underscore:
                w = re.sub(r'[^a-zA-Z0-9_]', '', w)
                if not w or (not w[0].isalpha() and w[0] != '_'): continue
            else:
                w = re.sub(r'[^a-zA-Z0-9]', '', w)
                if not w: continue

            if required_length is not None and len(w) != required_length:
                continue
            words.append(w)
        return words

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        """Validates if a specific candidate renaming is syntactically correct and doesn't conflict in the AST."""
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        try:
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'],
                                               {ctx['target_name']: cand}, analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 32) -> torch.Tensor:
        """
        核心科技：计算带有上下文的 Token 级变量语义向量，抛弃整句噪音。
        精准定位 BPE 分词后的变量边界，彻底解决分数拥挤和缩写识别问题。
        """
        all_embeddings = []
        tokenizer = self.mlm_engine.tokenizer

        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.mlm_engine.device)

            with torch.no_grad():
                outputs = self.mlm_engine.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

            for b_idx in range(len(batch_texts)):
                # 利用 BPE 序列匹配法，精准找到变量在句子中的 Token 索引
                p_tokens = tokenizer.encode(batch_prefixes[b_idx], add_special_tokens=False)
                pv_tokens = tokenizer.encode(batch_prefixes[b_idx] + batch_vars[b_idx], add_special_tokens=False)

                shared_len = 0
                for pt, pvt in zip(p_tokens, pv_tokens):
                    if pt == pvt:
                        shared_len += 1
                    else:
                        break

                # 定位变量 Token 的起止位置 (考虑 [CLS] 占据的 index 0)
                start_idx = shared_len + 1
                end_idx = len(pv_tokens) + 1

                # 边界安全拦截
                start_idx = min(start_idx, 511)
                end_idx = min(max(start_idx + 1, end_idx), 512)
                var_tokens_str = tokenizer.convert_ids_to_tokens(pv_tokens[shared_len:])
                # print(f"[Align Debug] Variable '{batch_vars[b_idx]}' aligned to tokens: {var_tokens_str}")
                # 精准抠出属于这个变量的所有 sub-token 向量，并做局部池化
                target_hiddens = last_hidden[b_idx, start_idx:end_idx, :]
                pooled = target_hiddens.mean(dim=0)

                all_embeddings.append(pooled.cpu().detach())

        return torch.stack(all_embeddings)

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx, is_full_context=False):
        # 获取基础语义阈值（假设配置中设为 0.85）
        base_threshold = ctx.get('semantic_threshold', 0.85)

        # 预过滤：基础检查
        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == ctx['target_name']: continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand): continue
            base_cands.append(cand)

        # 核心环节：Token-Level 语义过滤
        semantically_valid = []
        if base_cands and base_threshold > 0:
            # 1. 提取原变量 Token 向量
            orig_emb = self._get_variable_token_embeddings(
                [ctx['local_prefix']], [ctx['target_name']], [ctx['local_suffix']]
            ).to(self.mlm_engine.device)

            # 2. 批量提取候选词 Token 向量
            prefixes = [ctx['local_prefix']] * len(base_cands)
            suffixes = [ctx['local_suffix']] * len(base_cands)
            cand_embs = self._get_variable_token_embeddings(prefixes, base_cands, suffixes).to(self.mlm_engine.device)

            # 3. 计算相似度
            sims = F.cosine_similarity(orig_emb, cand_embs)

            # print(f"\n[*] 语义验证阶段 (Target: '{ctx['target_name']}', Base Threshold: {base_threshold}):")
            # print(f"{'Candidate':<20} | {'Token Sim':<10} | {'Dyn Thresh':<10} | {'Status'}")
            # print("-" * 65)

            target_name = ctx['target_name']
            target_parts, _ = self._split_identifier(target_name)

            for cand, sim in zip(base_cands, sims):
                score = sim.item()

                # ==================== 新增：动态阈值调整引擎 ====================
                current_threshold = base_threshold
                cand_parts, _ = self._split_identifier(cand)

                # 仅针对目标变量是单字的情况进行动态适应
                if len(target_parts) == 1:
                    if len(cand_parts) > 1:
                        # 发生了扩充 (生成了两个及以上的词汇)
                        if target_name.lower() in [p.lower() for p in cand_parts]:
                            # 模式 1: 保留了原词的扩充 (如 vma_cache) -> 天生分高，严格卡死 (+0.10)
                            current_threshold = min(0.98, base_threshold + 0.10)
                        else:
                            # 模式 2: 未保留原词的扩充 (如 mem_area) -> 分数稍高，适当收紧 (+0.05)
                            current_threshold = min(0.95, base_threshold + 0.05)
                    else:
                        # 模式 3: 非扩充，依然是单字替换 (如 area)
                        # 如果单词特别短（长度 <= 3），为了防止被误杀，甚至可以考虑微调降低阈值
                        if len(cand) <= 3:
                            current_threshold = max(0.50, base_threshold - 0.15)
                        else:
                            current_threshold = base_threshold
                # ================================================================

                is_pass = score >= current_threshold
                status = "[PASS]" if is_pass else "[FILTERED]"

                # 格式化输出以供调试
                # print(f"{cand:<20} | {score:.4f}{' ':<4} | {current_threshold:.4f}{' ':<4} | {status}")

                if is_pass:
                    semantically_valid.append(cand)
            # print("-" * 65 + "\n")
        else:
            semantically_valid = base_cands

        # 最终的 AST 验证逻辑
        added = 0
        for cand in semantically_valid:
            if added >= quota:
                break
            valid_cand = self._verify_ast_single(cand, ctx)
            if valid_cand and valid_cand not in final_candidates:
                final_candidates.append(valid_cand)
                added += 1

        return added

    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
        """
        利用 Tree-sitter AST 精准提取变量所在的最小逻辑语句（Statement）。
        向上回溯节点，直到其父节点是一个块作用域或全局作用域。
        """
        # 使用 analyzer 中已配置好的 language 创建临时 parser
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language
        tree = parser.parse(code_bytes)

        # 1. 找到精准匹配目标字节范围的 AST 节点
        node = tree.root_node.descendant_for_byte_range(target_start, target_end)

        if not node:
            # 极端异常情况下的降级：退回物理行
            line_start = code_bytes.rfind(b'\n', 0, target_start) + 1
            line_end = code_bytes.find(b'\n', target_end)
            if line_end == -1: line_end = len(code_bytes)
            return (code_bytes[line_start:target_start].decode("utf-8", errors="replace"),
                    code_bytes[target_end:line_end].decode("utf-8", errors="replace"))

        # 2. 向上回溯，寻找包含该变量的完整语句层级
        statement_node = node
        # 遇到这些父节点类型时停止回溯，意味着当前 node 已经是一个完整的 Statement/Declaration
        stop_parent_types = {
            'compound_statement',  # { ... } 内部的语句
            'translation_unit',  # 全局文件的顶层语句
            'function_definition',  # 防止把整个函数体都切进去
            'for_statement',  # 防止提取出整个 for 循环块，只保留初始化或条件部分
            'while_statement',
            'if_statement'
        }

        while statement_node.parent:
            if statement_node.parent.type in stop_parent_types:
                break
            statement_node = statement_node.parent

        stmt_start = statement_node.start_byte
        stmt_end = statement_node.end_byte

        # 3. 截取该 Statement 内部，变量前和变量后的字符串作为 prefix 和 suffix
        local_prefix = code_bytes[stmt_start:target_start].decode("utf-8", errors="replace")
        local_suffix = code_bytes[target_end:stmt_end].decode("utf-8", errors="replace")

        return local_prefix, local_suffix

    def _generate_core(self, code: str, target_name: str, identifiers: dict,
                       top_k_mlm: int, top_n_keep: int, semantic_threshold: float,
                       context_ratio: float, preserve_style: bool, strict_structure: bool) -> List[str]:
        """Orchestrates the candidate generation process including masking, batch inference, and filtering."""
        stripped_name = target_name.strip('_')
        if len(stripped_name) == 0: return []
        if target_name.startswith('__') and target_name.endswith('__'): return []

        code_bytes = code.encode("utf-8")
        if identifiers is None:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        if target_name not in identifiers: return []

        if len(stripped_name) == 1:
            min_freq = self.config.get('min_freq_for_single_char', 3)
            occurrence_count = len(identifiers[target_name])

            if occurrence_count < min_freq:
                return []

        original_style = self._detect_naming_style(target_name)

        # 获取目标变量在字节码中的位置信息
        target_info = identifiers[target_name][0]
        prefix = code_bytes[:target_info['start']]
        suffix = code_bytes[target_info['end']:]

        # 获取用于计算相似度的精确语句级上下文
        local_prefix, local_suffix = self._extract_local_context_ast(
            code_bytes, target_info['start'], target_info['end']
        )

        original_code_emb = None
        if semantic_threshold > 0:
            original_local_str = local_prefix + target_name + local_suffix
            original_code_emb = self._get_code_embedding_batched([original_local_str])
            if original_code_emb is not None:
                original_code_emb = original_code_emb.to(self.mlm_engine.device)

        parts, style = self._split_identifier(target_name)
        n_parts = len(parts)

        if n_parts > 10: return []

        target_lengths = [len(p) for p in parts]
        mask_token = self.mlm_engine.tokenizer.mask_token

        dynamic_top_k = max(3, top_k_mlm // max(1, (n_parts - 1)))

        variations = []

        # ==================== 单词变量的扩充机制 ====================
        if n_parts == 1 and not strict_structure:
            variations.append({'type': 'full', 'start': 0, 'end': 1, 'num_masks': 1, 'expand_mode': 'none'})
            variations.append({'type': 'full', 'start': 0, 'end': 1, 'num_masks': 1, 'expand_mode': 'prefix'})
            variations.append({'type': 'full', 'start': 0, 'end': 1, 'num_masks': 1, 'expand_mode': 'suffix'})
            variations.append({'type': 'full', 'start': 0, 'end': 1, 'num_masks': 2, 'expand_mode': 'both'})

        # ==================== 复合词变体生成 ====================
        elif strict_structure:
            variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': n_parts, 'expand_mode': 'none'})
            if n_parts > 1:
                max_sub_length = n_parts if n_parts <= 3 else 2
                for s in range(n_parts):
                    for e in range(s + 1, min(s + 1 + max_sub_length, n_parts + 1)):
                        if s == 0 and e == n_parts: continue
                        variations.append(
                            {'type': 'sub', 'start': s, 'end': e, 'num_masks': e - s, 'expand_mode': 'none'})
        else:
            variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': 1, 'expand_mode': 'none'})
            if n_parts > 1:
                variations.append(
                    {'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': n_parts, 'expand_mode': 'none'})
                max_sub_length = n_parts if n_parts <= 3 else 2
                for s in range(n_parts):
                    for e in range(s + 1, min(s + 1 + max_sub_length, n_parts + 1)):
                        if s == 0 and e == n_parts: continue
                        variations.append({'type': 'sub', 'start': s, 'end': e, 'num_masks': 1, 'expand_mode': 'none'})
                        if (e - s) > 1:
                            variations.append(
                                {'type': 'sub', 'start': s, 'end': e, 'num_masks': e - s, 'expand_mode': 'none'})

        cropped_codes = []
        context_half = 700
        mask_start = len(prefix)

        prefix_str = prefix[max(0, mask_start - context_half):].decode("utf-8", errors="replace")
        suffix_str = suffix[:context_half].decode("utf-8", errors="replace")

        for var in variations:
            expand_mode = var.get('expand_mode', 'none')

            # 根据模式构造带下划线的特殊 Mask
            if expand_mode == 'none':
                masked_var = self._build_masked_string(parts, var['start'], var['end'], var['num_masks'], style,
                                                       mask_token, target_name)
            elif expand_mode == 'prefix':
                masked_var = f"{mask_token}_{target_name}"
            elif expand_mode == 'suffix':
                masked_var = f"{target_name}_{mask_token}"
            elif expand_mode == 'both':
                masked_var = f"{mask_token}_{mask_token}"

            cropped_code = prefix_str + masked_var + suffix_str
            cropped_codes.append(cropped_code)

        batch_logits, batch_mask_indices = self._get_model_logits_batched(cropped_codes)

        raw_full_cands, raw_sub_cands_lists = [], []

        if batch_logits is not None:
            for i, var in enumerate(variations):
                logits = batch_logits[i:i + 1]
                mask_indices = batch_mask_indices[i]
                expand_mode = var.get('expand_mode', 'none')

                if len(mask_indices) < var['num_masks']: continue

                current_cands = []

                # ==================== 处理扩充模式的候选词组装 ====================
                if expand_mode != 'none':
                    if expand_mode == 'prefix':
                        words = self._decode_words(logits[0, mask_indices[0], :], dynamic_top_k, allow_underscore=False)
                        # print(f"[Debug] 前缀模式生成词汇: {words[:5]}...")
                        for w in words: current_cands.append(f"{w}_{target_name}")

                    elif expand_mode == 'suffix':
                        words = self._decode_words(logits[0, mask_indices[0], :], dynamic_top_k, allow_underscore=False)
                        # print(f"[Debug] 后缀模式生成词汇: {words[:5]}...")
                        for w in words: current_cands.append(f"{target_name}_{w}")

                    elif expand_mode == 'both':
                        per_mask_top_k = max(2, dynamic_top_k // 2)
                        words1 = self._decode_words(logits[0, mask_indices[0], :], per_mask_top_k,
                                                    allow_underscore=False)
                        words2 = self._decode_words(logits[0, mask_indices[1], :], per_mask_top_k,
                                                    allow_underscore=False)

                        MAX_EXPAND_COMBOS = 50
                        combo_count = 0
                        for w1, w2 in itertools.product(words1, words2):
                            if combo_count >= MAX_EXPAND_COMBOS: break
                            current_cands.append(f"{w1}_{w2}")
                            combo_count += 1

                # ==================== 原有的非扩充模式组装 ====================
                elif var['num_masks'] == 1 and not strict_structure:
                    words = self._decode_words(logits[0, mask_indices[0], :], dynamic_top_k, allow_underscore=True)
                    for w in words:
                        current_cands.append(
                            self._assemble_multi_candidate(parts, var['start'], var['end'], (w,), style, target_name))
                else:
                    per_mask_top_k = min(10, max(2, dynamic_top_k // var['num_masks']))
                    expanded_top_k = dynamic_top_k * 3 if strict_structure else per_mask_top_k
                    mask_preds = []

                    for m_idx in range(var['num_masks']):
                        part_idx = var['start'] + m_idx if strict_structure else None
                        req_len = target_lengths[part_idx] if strict_structure else None
                        words = self._decode_words(logits[0, mask_indices[m_idx], :], expanded_top_k,
                                                   allow_underscore=False, required_length=req_len)
                        words = words[:per_mask_top_k]

                        if strict_structure and not words:
                            words = [parts[part_idx]]
                        elif not strict_structure and not words:
                            words = ['temp']
                        mask_preds.append(words)

                    MAX_COMBINATIONS = 150
                    combo_count = 0
                    for combo in itertools.product(*mask_preds):
                        if combo_count >= MAX_COMBINATIONS: break
                        combo_count += 1
                        cand = self._assemble_multi_candidate(parts, var['start'], var['end'], combo, style,
                                                              target_name)
                        if strict_structure:
                            if [len(p) for p in self._split_identifier(cand)[0]] == target_lengths:
                                current_cands.append(cand)
                        else:
                            current_cands.append(cand)

                if var['type'] == 'full':
                    raw_full_cands.extend(current_cands)
                else:
                    raw_sub_cands_lists.append(current_cands)

        unique_full = list(dict.fromkeys(raw_full_cands))
        unique_sub = []
        seen_sub = set()
        if raw_sub_cands_lists:
            max_len = max(len(lst) for lst in raw_sub_cands_lists)
            for j in range(max_len):
                for lst in raw_sub_cands_lists:
                    if j < len(lst) and lst[j] not in seen_sub:
                        seen_sub.add(lst[j])
                        unique_sub.append(lst[j])

        ctx = {
            'code_bytes': code_bytes, 'target_name': target_name, 'identifiers': identifiers,
            'keywords': self.analyzer.keywords, 'original_style': original_style,
            'original_code_emb': original_code_emb,
            'local_prefix': local_prefix,
            'local_suffix': local_suffix,
            'semantic_threshold': semantic_threshold, 'preserve_style': preserve_style
        }

        final_candidates = []
        target_full_quota = int(top_n_keep * context_ratio)

        actual_full = self._verify_and_filter(unique_full, target_full_quota, final_candidates, ctx,
                                              is_full_context=True)
        self._verify_and_filter(unique_sub, top_n_keep - actual_full, final_candidates, ctx, is_full_context=False)

        if strict_structure and len(final_candidates) < top_n_keep:
            if hasattr(self, '_generate_structural_fallback'):
                local_cands = self._generate_structural_fallback(code_bytes, target_name, identifiers,
                                                                 top_n_keep - len(final_candidates))
                for lc in local_cands:
                    if lc not in final_candidates and self.analyzer.can_rename_to(code_bytes, target_name, lc):
                        final_candidates.append(lc)

        if len(final_candidates) > top_n_keep:
            return random.sample(final_candidates, top_n_keep)
        return final_candidates

    def generate_candidates(self, code: str, target_name: str, identifiers=None) -> List[str]:
        """Performs standard context-aware candidate generation."""
        return self._generate_core(
            code=code,
            target_name=target_name,
            identifiers=identifiers,
            top_k_mlm=self.config['top_k_mlm'],
            top_n_keep=self.config['top_n_keep'],
            semantic_threshold=self.config['semantic_threshold'],
            context_ratio=self.config['context_ratio'],
            preserve_style=self.config['preserve_style'],
            strict_structure=False
        )

    def generate_structural_candidates(self, code: str, target_name: str, identifiers=None) -> List[str]:
        """Generates isomorphic candidates by enforcing strict naming style and structural consistency."""
        return self._generate_core(
            code=code,
            target_name=target_name,
            identifiers=identifiers,
            top_k_mlm=self.config['top_k_mlm'],
            top_n_keep=self.config['top_n_keep'],
            semantic_threshold=self.config['structural_semantic_threshold'],
            context_ratio=self.config['context_ratio'],
            preserve_style=True,
            strict_structure=True
        )

    def _generate_structural_fallback(self, code_bytes: bytes, target_name: str, identifiers: dict, num_needed: int) -> \
    List[str]:
        """Provides a fallback mechanism by assembling candidates from local identifier fragments when MLM fails."""
        format_info = self.analyzer.analyze_format(target_name)
        target_lengths = format_info['lengths']
        local_names = list(identifiers.keys())

        pool = defaultdict(set)
        for name in local_names:
            parts = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)|[a-z0-9]+', name)
            for part in parts:
                if len(part) > 0: pool[len(part)].add(part.lower())

        length_pool = {length: list(words) for length, words in pool.items()}

        fallback_candidates = []
        for _ in range(50):
            sampled_words = []
            for length in target_lengths:
                if length in length_pool and length_pool[length]:
                    sampled_words.append(random.choice(length_pool[length]))
                else:
                    break

            if len(sampled_words) == len(target_lengths):
                if format_info['style'] == "snake_case":
                    assembled = "_".join(sampled_words)
                elif format_info['style'] == "camelCase":
                    assembled = sampled_words[0] + "".join(w.capitalize() for w in sampled_words[1:])
                else:
                    assembled = "".join(w.capitalize() for w in sampled_words)
                assembled = format_info['prefix'] + assembled

                if assembled != target_name and assembled not in self.analyzer.keywords:
                    fallback_candidates.append(assembled)

        return list(dict.fromkeys(fallback_candidates))[:num_needed]

    def get_top_mlm_words(self, code_bytes: bytes, target_name: str, top_k=20) -> List[str]:
        """Predicts the most suitable raw words for a masked identifier position using the MLM model."""
        mask_token = self.mlm_engine.tokenizer.mask_token
        pattern = rf'\b{re.escape(target_name)}\b'.encode()
        masked_code_bytes = re.sub(pattern, mask_token.encode(), code_bytes, count=1)

        inputs = self.mlm_engine.tokenizer(
            masked_code_bytes.decode(errors='replace'),
            return_tensors="pt", truncation=True, max_length=512
        ).to(self.mlm_engine.device)

        mask_token_id = self.mlm_engine.tokenizer.mask_token_id
        mask_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) == 0: return []

        with torch.no_grad():
            logits = self.mlm_engine.model(**inputs).logits
            mask_logits = logits[0, mask_indices[0], :]
            _, top_k_indices = torch.topk(mask_logits, top_k, dim=-1)

        words = []
        for idx in top_k_indices:
            word = self.mlm_engine.tokenizer.decode([idx]).strip().lower()
            word = re.sub(r'[^a-z]', '', word)
            if len(word) > 1:
                words.append(word)
        return words

    def _infer_type_from_code(self, code: str, target_name: str) -> str:
        """Heuristically infers the data type of an identifier from its surrounding C/C++ source code."""
        pattern = r'([a-zA-Z_][\w\s\*,&:]*?)\s*\b' + re.escape(target_name) + r'\b\s*[\[=;,)]'
        match = re.search(pattern, code)

        if match:
            type_part = match.group(1).strip()
            if ',' in type_part:
                type_part = type_part.split(',')[0].strip()
                type_part = ' '.join([word for word in type_part.split() if word in
                                      ["int", "long", "short", "char", "float", "double", "unsigned", "signed",
                                       "struct", "class"]])

            type_part = re.sub(r'\b(static|const|inline|extern|volatile|register)\b', '', type_part).strip()
            if type_part:
                return type_part

        param_pattern = r'([a-zA-Z_][\w\s\*,&:]*?)\s*\b' + re.escape(target_name) + r'\b\s*[,)]'
        match = re.search(param_pattern, code)
        if match:
            type_part = match.group(1).strip()
            type_part = re.sub(r'\b(const)\b', '', type_part).strip()
            if type_part:
                return type_part

        return "void"

    def generate_normalized_name(self, code: str, target_name: str, var_type: str, excluded_names: set) -> str:
        """Generates a normalized identifier name based on inferred type and context-aware MLM predictions."""
        if re.search(r'\b' + re.escape(target_name) + r'\s*\(', code):
            category = "fun"
        else:
            inferred_type = self._infer_type_from_code(code, target_name)
            clean_type = re.sub(r'\b(struct|class|enum|union)\b', '', inferred_type).strip()
            is_pointer = "*" in inferred_type or "*" in clean_type
            core_type = clean_type.replace("*", "").replace("&", "").strip()
            core_type = core_type.split()[-1] if core_type else "void"

            primitives_int = ["int", "long", "short", "size_t", "float", "double", "unsigned", "signed", "uint32_t",
                              "uint64_t", "int32_t", "uint8_t"]
            primitives_char = ["char"]

            if core_type == "void":
                category = "var"
            elif core_type.lower() not in (primitives_int + primitives_char + ["bool"]):
                if "::" in core_type:
                    category = core_type.split("::")[-1]
                else:
                    category = core_type
                category = re.sub(r'\W+', '', category)
                if not category or category.isdigit():
                    category = "obj"
                elif category[0].isdigit():
                    category = "v" + category
            else:
                if is_pointer:
                    category = "pointer"
                elif core_type.lower() in primitives_char:
                    category = "char"
                else:
                    category = "int"

        code_str = code if isinstance(code, str) else code.decode('utf-8')
        candidate_words = self.get_top_mlm_words(code_str.encode(), target_name)

        for w in candidate_words:
            w = re.sub(r'\W+', '', w)
            if not w: continue
            potential_name = f"{category}_{w}"
            if potential_name not in excluded_names:
                return potential_name

        return f"{category}_{len(excluded_names)}"