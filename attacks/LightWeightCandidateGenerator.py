import re
import itertools
import torch
import torch.nn.functional as F
from typing import List, Dict, Any


class LightweightCandidateGenerator:
    def __init__(self, mlm_engine, analyzer, config):
        """
        初始化轻量级生成器（仅依赖 MLM，用作快速探针与初步攻击空间构建）
        :param mlm_engine: 用于掩码预测、提取 Token Embedding 和计算相似度的模型
        :param analyzer: 语法树与上下文分析器
        :param config: 全局配置字典
        """
        self.mlm_engine = mlm_engine
        self.analyzer = analyzer
        self.config = config
        stats_path = config.get('naming_stats_path', 'naming_stats.json')

        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    # =========================================================================
    # 基础工具与命名规范检测 (原样保留)
    # =========================================================================
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
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower': return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'): return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE': return True
        return cand_style == original_style

    def _split_identifier(self, name: str):
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name): return [name], ''
            return parts, 'camel'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
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

    # =========================================================================
    # 上下文提取与模型推理交互
    # =========================================================================
    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language
        tree = parser.parse(code_bytes)
        node = tree.root_node.descendant_for_byte_range(target_start, target_end)

        if not node:
            line_start = code_bytes.rfind(b'\n', 0, target_start) + 1
            line_end = code_bytes.find(b'\n', target_end)
            if line_end == -1: line_end = len(code_bytes)
            return (code_bytes[line_start:target_start].decode("utf-8", errors="replace"),
                    code_bytes[target_end:line_end].decode("utf-8", errors="replace"))

        statement_node = node
        stop_parent_types = {'compound_statement', 'translation_unit', 'function_definition', 'for_statement',
                             'while_statement', 'if_statement'}

        while statement_node.parent and statement_node.parent.type not in stop_parent_types:
            statement_node = statement_node.parent

        stmt_start = statement_node.start_byte
        stmt_end = statement_node.end_byte
        local_prefix = code_bytes[stmt_start:target_start].decode("utf-8", errors="replace")
        local_suffix = code_bytes[target_end:stmt_end].decode("utf-8", errors="replace")
        return local_prefix, local_suffix

    def _find_best_context_occurrence(self, code_bytes: bytes, occurrences: List[dict]) -> int:
        if len(occurrences) <= 1: return 0
        best_idx, max_score = 0, -1.0
        search_limit = min(len(occurrences), 10)

        for i in range(search_limit):
            occ = occurrences[i]
            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, occ['start'], occ['end'])
            score = len(local_prefix) + len(local_suffix)
            if '(' in local_suffix or ',' in local_suffix: score += 100
            if any(k in local_prefix for k in ['if ', 'while ', 'for ', 'return ']): score += 80
            if re.search(r'=\s*(0|NULL|nullptr|false|true|\{\})\s*;', local_suffix): score -= 150
            if score > max_score:
                max_score = score
                best_idx = i
        return best_idx

    def _get_model_logits_batched(self, cropped_codes: List[str]):
        if not cropped_codes: return None, []
        inputs = self.mlm_engine.tokenizer(
            cropped_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.mlm_engine.device)
        mask_token_id = self.mlm_engine.tokenizer.mask_token_id

        with torch.no_grad():
            batch_logits = self.mlm_engine.model(**inputs).logits

        batch_mask_indices = [(inputs.input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0] for i in
                              range(batch_logits.size(0))]
        return batch_logits, batch_mask_indices

    def _decode_words(self, mask_logits, top_k, allow_underscore=False, required_length=None):
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
            if required_length is not None and len(w) != required_length: continue
            words.append(w)
        return words

    # =========================================================================
    # 复杂度过滤系统 (特征提取与验证)
    # =========================================================================
    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 64) -> torch.Tensor:
        all_embeddings = []
        tokenizer = self.mlm_engine.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]
        device = self.mlm_engine.device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[
            0] >= 8 else torch.float16
        self.mlm_engine.model.to(dtype)

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = self.mlm_engine.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

            cached_p_tokens = {}
            for b_idx in range(len(batch_texts)):
                p_text = batch_prefixes[b_idx]
                if p_text not in cached_p_tokens:
                    cached_p_tokens[p_text] = tokenizer.encode(p_text, add_special_tokens=False)

                p_tokens = cached_p_tokens[p_text]
                pv_tokens = tokenizer.encode(p_text + batch_vars[b_idx], add_special_tokens=False)

                shared_len = sum(1 for pt, pvt in zip(p_tokens, pv_tokens) if pt == pvt)
                start_idx = min(shared_len + 1, 255)
                end_idx = min(max(start_idx + 1, len(pv_tokens) + 1), 256)

                pooled = last_hidden[b_idx, start_idx:end_idx, :].mean(dim=0)
                all_embeddings.append(pooled.to(torch.float32).cpu())

        return torch.stack(all_embeddings)

    def _is_trivial_change(self, target_name: str, cand: str) -> bool:
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        # 使用你外部配置的 CodeTransformer 进行校验
        try:
            from utils.ast_tools import CodeTransformer  # 根据实际路径调整
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'], {ctx['target_name']: cand},
                                               analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx):
        """完全还原：纯粹的余弦相似度 + Heuristic Bonus 过滤逻辑，带全链路监控日志"""
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        # 1. 预过滤：基础检查
        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == ctx['target_name']:
                print(f"        🚫 [Filter | Keyword/Self] '{cand}'")
                continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand):
                print(f"        🚫 [Filter | Style Clash] '{cand}' (Expected: {ctx['original_style']})")
                continue
            base_cands.append(cand)

        if not base_cands:
            return 0

        # 2. 提取原始向量
        orig_emb = None
        if base_threshold > 0:
            orig_emb = self._get_variable_token_embeddings(
                [ctx['local_prefix']], [ctx['target_name']], [ctx['local_suffix']]
            ).to(self.mlm_engine.device)

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        for i in range(0, len(base_cands), CHUNK_SIZE):
            if added >= quota: break

            chunk = base_cands[i: i + CHUNK_SIZE]
            filtered_chunk = []
            heuristic_bonuses = []

            for cand in chunk:
                bonus = 0.0
                if hasattr(self, 'scorer'):
                    cand_parts, _ = self._split_identifier(cand)
                    bonus = self.scorer.calculate_heuristic_score(
                        cand_parts, entity_type, target_parts=target_parts, return_type=return_type
                    )

                if bonus <= -100:
                    print(f"        🚫 [Filter | NLP Rules] '{cand}' (Score: {bonus})")
                    continue
                if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
                    print(f"        🚫 [Filter | AST Conflict] '{cand}' (Scope collision or Syntax error)")
                    continue

                filtered_chunk.append(cand)
                heuristic_bonuses.append(bonus)

            if not filtered_chunk: continue
            semantically_valid = []

            # 3. 核心打分与拦截
            if base_threshold > 0:
                prefixes = [ctx['local_prefix']] * len(filtered_chunk)
                suffixes = [ctx['local_suffix']] * len(filtered_chunk)

                cand_embs = self._get_variable_token_embeddings(prefixes, filtered_chunk, suffixes).to(
                    self.mlm_engine.device)
                sims = F.cosine_similarity(orig_emb, cand_embs)

                for cand, sim, bonus in zip(filtered_chunk, sims, heuristic_bonuses):
                    final_score = sim.item() + bonus

                    if final_score >= base_threshold:
                        # 记录符合语义的词汇，留给下一步 AST 验证
                        semantically_valid.append((cand, final_score, sim.item(), bonus))
                    else:
                        print(
                            f"        🚫 [Filter | Semantic] '{cand}' (Sim: {sim.item():.3f} + Bonus: {bonus:.3f} = {final_score:.3f} < {base_threshold})")
            else:
                # 如果不需要语义打分，直接赋默认分过关
                semantically_valid = [(cand, 1.0, 1.0, 0.0) for cand in filtered_chunk]

            # 4. 最终 AST 验证并加入池
            for cand_data in semantically_valid:
                if added >= quota: break
                cand, final_score, sim_val, bonus_val = cand_data

                valid_cand = self._verify_ast_single(cand, ctx)
                if valid_cand and valid_cand not in final_candidates:
                    final_candidates.append(valid_cand)
                    added += 1
                    # ==========================================
                    # 【核心修改】：打印成功通过的所有考核指标
                    # ==========================================
                    if base_threshold > 0:
                        print(
                            f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {final_score:.3f} = Sim {sim_val:.3f} + Bonus {bonus_val:.3f})")
                    else:
                        print(f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Semantic check disabled)")
                else:
                    if not valid_cand:
                        print(f"        🚫 [Filter | Final AST] '{cand}' (Failed context insertion verify)")
                    else:
                        print(f"        🚫 [Filter | Duplicate] '{cand}' (Already in final pool)")

        return added

    # =========================================================================
    # 核心入口: 仅执行 MLM 生成与过滤
    # =========================================================================
    def generate_candidates(self, batch_tasks: List[Dict[str, Any]], top_k_mlm: int = 40, top_n_keep: int = 20) -> Dict[
        str, List[str]]:
        """
        批量快速生成 MLM 候选词。
        返回的数据将直接送入 RNNS 进行显著性排序。
        """
        results = {task["target_name"]: [] for task in batch_tasks}
        mlm_tracking = []
        task_metadata = {}
        mask_token = self.mlm_engine.tokenizer.mask_token

        # 1. 任务解析与 MLM 变体构建
        for task_idx, task in enumerate(batch_tasks):
            target_name = task["target_name"]
            code_str = task["code_str"]
            code_bytes = code_str.encode("utf-8")

            identifiers = self.analyzer.extract_identifiers(code_bytes)
            if target_name not in identifiers: continue

            best_occ_idx = self._find_best_context_occurrence(code_bytes, identifiers[target_name])
            target_info = identifiers[target_name][best_occ_idx]

            entity_type = 'BOOLEAN_VAR' if target_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if target_info.get('entity_type') == 'function' else 'VARIABLE')
            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(target_name)
            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, target_info['start'],
                                                                         target_info['end'])

            task_metadata[task_idx] = {
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "identifiers": identifiers, "entity_type": entity_type, "original_style": original_style,
                "code_bytes": code_bytes, "local_prefix": local_prefix, "local_suffix": local_suffix,
                "raw_mlm_cands": []
            }

            prefix_str = local_prefix[max(0, len(local_prefix) - 700):]
            suffix_str = local_suffix[:700]

            variations = []
            if len(parts) == 1:
                variations.extend([
                    {'expand_mode': 'none', 'num_masks': 1, 'masked_str': mask_token},
                    {'expand_mode': 'prefix', 'num_masks': 1, 'masked_str': f"{mask_token}_{target_name}"},
                    {'expand_mode': 'suffix', 'num_masks': 1, 'masked_str': f"{target_name}_{mask_token}"}
                ])
            else:
                for i in range(len(parts)):
                    variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 1, 'num_masks': 1,
                                       'masked_str': self._build_masked_string(parts, i, i + 1, 1, style, mask_token,
                                                                               target_name)})
                if len(parts) >= 2:
                    for i in range(len(parts) - 1):
                        variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 2, 'num_masks': 2,
                                           'masked_str': self._build_masked_string(parts, i, i + 2, 2, style,
                                                                                   mask_token, target_name)})

            for var in variations:
                mlm_tracking.append({"task_idx": task_idx, "cropped_code": prefix_str + var['masked_str'] + suffix_str,
                                     "variation_info": var})

        if not task_metadata: return results

        # 2. 批量 MLM 推理
        all_cropped_codes = [item["cropped_code"] for item in mlm_tracking]
        batch_logits, batch_mask_indices = self._get_model_logits_batched(all_cropped_codes)

        # 3. 解析 MLM 输出
        def _join_parts(new_parts, orig_name, st):
            if st == '_':
                return "_".join(new_parts)
            elif st == 'camel':
                return "".join(
                    p.lower() if j == 0 and orig_name[0].islower() else p.capitalize() for j, p in enumerate(new_parts))
            return "".join(new_parts)

        if batch_logits is not None:
            for i, track_info in enumerate(mlm_tracking):
                meta = task_metadata[track_info["task_idx"]]
                var_info = track_info["variation_info"]
                logits = batch_logits[i:i + 1]
                mask_indices = batch_mask_indices[i]
                num_masks = var_info.get('num_masks', 1)

                if len(mask_indices) < num_masks: continue

                if num_masks == 1:
                    words = self._decode_words(logits[0, mask_indices[0], :], top_k_mlm)
                    for w in words:
                        if var_info.get('expand_mode') == 'prefix':
                            meta["raw_mlm_cands"].append(f"{w}_{meta['target_name']}")
                        elif var_info.get('expand_mode') == 'suffix':
                            meta["raw_mlm_cands"].append(f"{meta['target_name']}_{w}")
                        else:
                            if meta["n_parts"] == 1:
                                meta["raw_mlm_cands"].append(w)
                            else:
                                meta["raw_mlm_cands"].append(_join_parts(
                                    meta["parts"][:var_info['start']] + [w] + meta["parts"][var_info['end']:],
                                    meta["target_name"], meta["style"]))
                elif num_masks == 2:
                    top_k_2holes = min(4, max(2, top_k_mlm // 4))
                    words1 = self._decode_words(logits[0, mask_indices[0], :], top_k_2holes)
                    words2 = self._decode_words(logits[0, mask_indices[1], :], top_k_2holes)
                    for w1, w2 in itertools.product(words1, words2):
                        meta["raw_mlm_cands"].append(
                            _join_parts(meta["parts"][:var_info['start']] + [w1, w2] + meta["parts"][var_info['end']:],
                                        meta["target_name"], meta["style"]))

        # 4. 执行复杂度过滤系统
        for t_idx, meta in task_metadata.items():
            unique_mlm_cands = list(dict.fromkeys(meta["raw_mlm_cands"]))
            ctx = {
                'code_bytes': meta["code_bytes"], 'target_name': meta["target_name"],
                'identifiers': meta["identifiers"],
                'keywords': self.analyzer.keywords, 'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"], 'semantic_threshold': self.config.get('semantic_threshold', 0.85),
                'preserve_style': self.config.get('preserve_style', True), 'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["identifiers"][meta["target_name"]] if u.get('return_type')), None)
            }

            final_candidates = []
            self._verify_and_filter(unique_mlm_cands, top_n_keep, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates

        return results