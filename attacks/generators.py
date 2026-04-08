import re
import random
import torch
import itertools
from typing import List, Dict
import torch.nn.functional as F
from collections import defaultdict

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier, CodeTransformer


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo, analyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer
        self.common_affixes = ['ptr', 'buf', 'val', 'idx', 'msg', 'data', 'tmp', 'ref', 'res', 'list', 'obj', 'item', 'ctx']

    def _detect_naming_style(self, name: str) -> str:
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
        cand_style = self._detect_naming_style(candidate)
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower':
            return True
        return cand_style == original_style

    def _get_word_embedding(self, word: str) -> torch.Tensor:
        tokenizer = self.model_zoo.mlm_tokenizer
        tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(tokens) == 0:
            return None
        tokens = tokens.to(self.model_zoo.device)
        with torch.no_grad():
            embeddings = self.model_zoo.mlm_model.get_input_embeddings()(tokens)
        return embeddings.mean(dim=0)

    def _split_identifier(self, name: str):
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name):
                return [name], ''
            return parts, 'camel'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
        """根据跨度和指定的 Mask 数量构造带 Mask 的变量名"""
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
        """将预测出的多个词元与原词的剩余部分重新组装"""
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

    def generate_candidates(self, code: str, target_name: str, identifiers=None,
                            top_k_mlm=80, top_n_keep=50,
                            preserve_style: bool = True,
                            semantic_threshold: float = 0.2,
                            context_ratio: float = 0.3) -> List[str]:
        """
        新增参数:
        context_ratio (float): [0.0, 1.0] 之间，指定有多少比例的候选词是完全脱离原变量名（即全掩码）生成的。
        """
        keywords = self.analyzer.keywords
        code_bytes = code.encode("utf-8")

        if identifiers is None:
            identifiers = self.analyzer.extract_identifiers(code_bytes)

        if target_name not in identifiers:
            return []

        original_style = self._detect_naming_style(target_name)
        original_emb = None
        if semantic_threshold > 0:
            original_emb = self._get_word_embedding(target_name)

        target_info = identifiers[target_name][0]
        start_byte = target_info['start']
        end_byte = target_info['end']
        prefix = code_bytes[:start_byte]
        suffix = code_bytes[end_byte:]

        parts, style = self._split_identifier(target_name)
        mask_token = self.model_zoo.mlm_tokenizer.mask_token

        # 1. 构建所有掩码变体 (Topology Variations)
        variations = []
        n_parts = len(parts)

        # A. 全掩码 (完全依赖上下文)
        variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': 1})
        if n_parts > 1:
            # 用同等数量的 Mask 替换全身，联合预测全新的长变量
            variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': n_parts})

        # B. 局部跨度掩码 (Sub-span Masking)
        if n_parts > 1:
            for start in range(n_parts):
                for end in range(start + 1, n_parts + 1):
                    if start == 0 and end == n_parts:
                        continue  # 已经是全掩码了

                    span_len = end - start
                    # 变体 1: 将这段跨度压缩为 1 个 Mask (例如 3词 变 1词)
                    variations.append({'type': 'sub', 'start': start, 'end': end, 'num_masks': 1})

                    # 变体 2: 等长替换 (例如 3词 换 3词)
                    if span_len > 1:
                        variations.append({'type': 'sub', 'start': start, 'end': end, 'num_masks': span_len})

        raw_full_cands = []
        raw_sub_cands_lists = []  # 用于子集变体的交替收集

        # 2. 执行模型预测
        for var in variations:
            masked_var_name = self._build_masked_string(parts, var['start'], var['end'], var['num_masks'], style,
                                                        mask_token, target_name)
            masked_code_bytes = prefix + masked_var_name.encode("utf-8") + suffix

            mask_start_in_masked = len(prefix)
            mask_end_in_masked = mask_start_in_masked + len(masked_var_name.encode("utf-8"))
            context_half_size = 700

            crop_start = max(0, mask_start_in_masked - context_half_size)
            crop_end = min(len(masked_code_bytes), mask_end_in_masked + context_half_size)
            cropped_code = masked_code_bytes[crop_start:crop_end].decode("utf-8", errors="replace")

            inputs = self.model_zoo.mlm_tokenizer(
                cropped_code, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model_zoo.device)

            mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
            mask_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_indices) < var['num_masks']:
                continue

            with torch.no_grad():
                logits = self.model_zoo.mlm_model(**inputs).logits

                # 如果只有一个 MASK
                if var['num_masks'] == 1:
                    mask_logits = logits[0, mask_indices[0], :]
                    _, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)

                    current_cands = []
                    for idx in top_k_indices:
                        pred_word = self.model_zoo.mlm_tokenizer.decode([idx]).strip().replace('Ġ', '').replace('##',
                                                                                                                '')
                        pred_word = re.sub(r'[^a-zA-Z0-9_]', '', pred_word)
                        if not pred_word or (not pred_word[0].isalpha() and pred_word[0] != '_'):
                            continue

                        full_cand = self._assemble_multi_candidate(parts, var['start'], var['end'], (pred_word,), style,
                                                                   target_name)
                        current_cands.append(full_cand)

                # 如果有多个 MASK (联合预测)
                else:
                    per_mask_top_k = min(10, max(3, top_k_mlm // (var['num_masks'] * 2)))  # 限制多词组合防爆炸
                    mask_preds = []

                    for m_idx in range(var['num_masks']):
                        m_logits = logits[0, mask_indices[m_idx], :]
                        _, top_indices = torch.topk(m_logits, per_mask_top_k, dim=-1)

                        words = []
                        for idx in top_indices:
                            w = self.model_zoo.mlm_tokenizer.decode([idx]).strip().replace('Ġ', '').replace('##', '')
                            w = re.sub(r'[^a-zA-Z0-9]', '', w)  # 多词联合中间不能带下划线
                            if w: words.append(w)
                        mask_preds.append(words if words else ['temp'])

                    # 笛卡尔积产生联合组合 (例如 top 5 x top 5 x top 5 = 125 种可能)
                    current_cands = []
                    for combo in itertools.product(*mask_preds):
                        full_cand = self._assemble_multi_candidate(parts, var['start'], var['end'], combo, style,
                                                                   target_name)
                        current_cands.append(full_cand)

            if var['type'] == 'full':
                raw_full_cands.extend(current_cands)
            else:
                raw_sub_cands_lists.append(current_cands)

        # 3. 收集与整理
        # (1) 去重纯上下文生成词
        unique_full = list(dict.fromkeys(raw_full_cands))

        # (2) 交替收集局部替换词，保证多样性
        unique_sub = []
        seen_sub = set()
        if raw_sub_cands_lists:
            max_len = max(len(lst) for lst in raw_sub_cands_lists)
            for j in range(max_len):
                for lst in raw_sub_cands_lists:
                    if j < len(lst) and lst[j] not in seen_sub:
                        seen_sub.add(lst[j])
                        unique_sub.append(lst[j])

        # 4. 配额分配与最终过滤
        target_full_quota = int(top_n_keep * context_ratio)
        target_sub_quota = top_n_keep - target_full_quota

        final_candidates = []

        # Modified _filter_and_add
        def _filter_and_add(candidate_list, quota, is_full_context=False):
            added_count = 0
            for cand in candidate_list:
                if added_count >= quota:
                    break
                if cand in keywords or cand == target_name: continue
                if preserve_style and not self._matches_style(original_style, cand): continue

                # CRITICAL FIX 1: Bypass semantic check if it's a full context guess
                if not is_full_context and semantic_threshold > 0 and original_emb is not None:
                    cand_emb = self._get_word_embedding(cand)
                    if cand_emb is not None:
                        sim = F.cosine_similarity(original_emb.unsqueeze(0), cand_emb.unsqueeze(0)).item()
                        if sim < semantic_threshold: continue

                if not self.analyzer.can_rename_to(code_bytes, target_name, cand): continue
                try:
                    _ = CodeTransformer.validate_and_apply(code_bytes, identifiers, {target_name: cand},
                                                           analyzer=self.analyzer)
                    final_candidates.append(cand)
                    added_count += 1
                except Exception:
                    continue
            return added_count

        # 优先填满 Full Context 词汇
        actual_full_added = _filter_and_add(unique_full, target_full_quota, is_full_context=True)

        # 如果 Full Context 没填满配额，把名额让给 Sub Mask
        remaining_quota = top_n_keep - actual_full_added
        _filter_and_add(unique_sub, remaining_quota, is_full_context=False)
        # 最后去重返回
        unique_final = list(dict.fromkeys(final_candidates))

        if len(unique_final) > top_n_keep:
            return random.sample(unique_final, top_n_keep)
        return unique_final

    def _build_length_pool(self, mlm_seeds: List[str], local_identifiers: List[str]) -> Dict[int, List[str]]:
        """
        优化 A：构建长度索引词库
        将所有可用的单词（来自模型预测和本地代码）按长度分类
        """
        pool = defaultdict(set)

        # 1. 处理 MLM 种子词和本地标识符
        # 我们需要把复合词拆开，获取最基础的单词块
        all_raw_names = mlm_seeds + local_identifiers + self.common_affixes

        for name in all_raw_names:
            # 统一拆分为纯单词块
            parts = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)|[a-z0-9]+', name)
            for part in parts:
                clean_part = part.lower()
                if len(clean_part) > 0:
                    pool[len(clean_part)].add(clean_part)

        # 转回列表方便 random.sample
        return {length: list(words) for length, words in pool.items()}

    def _assemble_by_style(self, words: List[str], format_info: dict) -> str:
        style = format_info['style']
        prefix = format_info['prefix']

        # 只要是 snake_case，就用下划线连接
        if style == "snake_case":
            # words 里面已经是对应长度的候选词了
            # 这里直接用 '_' 连接即可
            assembled = "_".join(words)
        elif style == "PascalCase":
            assembled = "".join(w.capitalize() for w in words)
        elif style == "camelCase":
            assembled = words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else ""
        else:
            assembled = "".join(words)

        return prefix + assembled

    def generate_structural_candidates(self, code: str, target_name: str, top_n_keep=20) -> List[str]:
        """
        核心方法：生成与原变量名格式完全相同的候选词
        """
        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        if target_name not in identifiers:
            return []

        # 1. 解析原始格式 (使用你在 ast_tools 中新增的方法)
        format_info = self.analyzer.analyze_format(target_name)
        target_lengths = format_info['lengths']

        # 2. 获取原材料
        # 为了保证语义，这里可以调用你原来的 MLM 逻辑获取 base_seeds
        # 假设这里我们已经拿到了基础词 pool
        # 同时也拿到了当前代码中定义的本地变量名作为补充
        local_names = list(identifiers.keys())

        # 注意：这里需要你原有的 MLM 预测逻辑获取 seeds，此处简化处理
        mlm_seeds = []  # 实际运行时应调用模型预测

        length_pool = self._build_length_pool(mlm_seeds, local_names)

        # 3. 尝试组装候选词
        structural_candidates = []
        max_attempts = 100  # 防止死循环

        for _ in range(max_attempts):
            sampled_words = []
            possible = True

            for length in target_lengths:
                if length in length_pool and len(length_pool[length]) > 0:
                    sampled_words.append(random.choice(length_pool[length]))
                else:
                    possible = False
                    break

            if possible and sampled_words:
                new_name = self._assemble_by_style(sampled_words, format_info)
                structural_candidates.append(new_name)

            # 去重
            structural_candidates = list(dict.fromkeys(structural_candidates))
            if len(structural_candidates) >= top_n_keep * 2:  # 多生成一些用于过滤
                break

        # 4. 验证过滤 (合法性、作用域冲突等)
        valid_candidates = []
        for cand in structural_candidates:
            if cand == target_name or cand in self.analyzer.keywords:
                continue

            # 静态检查
            if self.analyzer.can_rename_to(code_bytes, target_name, cand):
                valid_candidates.append(cand)

            if len(valid_candidates) >= top_n_keep:
                break

        return valid_candidates