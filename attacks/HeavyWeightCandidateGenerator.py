import itertools
import json
import random
import re
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F

from utils.ast_tools import CodeTransformer


class HeavyWeightCandidateGenerator:
    def __init__(self, mlm_engine, llm_client, analyzer, config):
        """
        初始化生成器。
        :param mlm_engine: 用于提取 Token Embedding 和计算相似度的模型 (如 CodeBERT)
        :param llm_client: 用于生成候选词的轻量级本地大模型客户端
        """
        self.mlm_engine = mlm_engine
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.config = config
        stats_path = config.get('naming_stats_path', 'naming_stats.json')

        # 假设你已经导入了 StatisticalNamingScorer
        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

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
                outputs = self.mlm_engine.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                all_embeddings.append(mean_pooled.cpu().detach())

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

    def _find_best_context_occurrence(self, code_bytes: bytes, occurrences: List[dict]) -> int:
        """在变量的所有出现位置中，评估并选择一个语义信息最丰富的上下文位置。"""
        if len(occurrences) <= 1: return 0

        best_idx = 0
        max_score = -1.0
        search_limit = min(len(occurrences), 10)

        for i in range(search_limit):
            occ = occurrences[i]
            local_prefix, local_suffix = self._extract_local_context_ast(
                code_bytes, occ['start'], occ['end']
            )

            score = 0.0
            score += len(local_prefix) + len(local_suffix)

            if '(' in local_suffix or ',' in local_suffix:
                score += 100

            if any(k in local_prefix for k in ['if ', 'while ', 'for ', 'return ']):
                score += 80

            if re.search(r'=\s*(0|NULL|nullptr|false|true|\{\})\s*;', local_suffix):
                score -= 150

            if score > max_score:
                max_score = score
                best_idx = i

        return best_idx

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
        # 假设 CodeTransformer 已经在外部导入或可用
        try:
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'],
                                               {ctx['target_name']: cand}, analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 32) -> torch.Tensor:
        """精准定位 BPE 分词后的变量边界，提取Token级变量语义向量"""
        all_embeddings = []
        tokenizer = self.mlm_engine.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.mlm_engine.device)

            with torch.no_grad():
                outputs = self.mlm_engine.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

            cached_p_tokens = {}

            for b_idx in range(len(batch_texts)):
                p_text = batch_prefixes[b_idx]
                if p_text not in cached_p_tokens:
                    cached_p_tokens[p_text] = tokenizer.encode(p_text, add_special_tokens=False)

                p_tokens = cached_p_tokens[p_text]
                pv_tokens = tokenizer.encode(p_text + batch_vars[b_idx], add_special_tokens=False)

                shared_len = 0
                for pt, pvt in zip(p_tokens, pv_tokens):
                    if pt == pvt:
                        shared_len += 1
                    else:
                        break

                start_idx = shared_len + 1
                end_idx = len(pv_tokens) + 1
                start_idx = min(start_idx, 255)
                end_idx = min(max(start_idx + 1, end_idx), 256)

                target_hiddens = last_hidden[b_idx, start_idx:end_idx, :]
                pooled = target_hiddens.mean(dim=0)
                all_embeddings.append(pooled.cpu().detach())

        return torch.stack(all_embeddings)

    def _is_trivial_change(self, target_name: str, cand: str) -> bool:
        """【新增】纯文本判断是否为微小改动（前置短路拦截使用）"""
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)

        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx, is_full_context=False,
                           generator_source="UNKNOWN"):
        """执行语义校验与生态位配额过滤 (支持区分 LLM 与 MLM 动态阈值)"""
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        # 预过滤：基础检查
        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == ctx['target_name']: continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand): continue
            base_cands.append(cand)

        if not base_cands:
            return 0

        orig_emb = None
        if base_threshold > 0:
            orig_emb = self._get_variable_token_embeddings(
                [ctx['local_prefix']], [ctx['target_name']], [ctx['local_suffix']]
            ).to(self.mlm_engine.device)

            # print(
            #     f"\n[*] 语义验证阶段 (Target: '{ctx['target_name']}', Type: {entity_type}, Source: {generator_source}, Base: {base_threshold}):")
            # print(
            #     f"{'Candidate':<25} | {'Base Sim':<9} | {'NLP Bonus':<10} | {'Final':<8} | {'Thresh':<8} | {'Status'}")
            # print("-" * 85)

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        trivial_added = 0
        max_trivial_allowed = max(1, int(quota * 0.4))
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)

        return_type = ctx.get('return_type', None)

        for i in range(0, len(base_cands), CHUNK_SIZE):
            if added >= quota:
                break

            chunk = base_cands[i: i + CHUNK_SIZE]

            filtered_chunk = []
            is_trivial_flags = []
            heuristic_bonuses = []

            for cand in chunk:
                bonus = 0.0
                if hasattr(self, 'scorer'):
                    cand_parts, _ = self._split_identifier(cand)
                    bonus = self.scorer.calculate_heuristic_score(
                        cand_parts,
                        entity_type,
                        target_parts=target_parts,
                        return_type=return_type
                    )

                if bonus <= -100:
                    continue

                is_triv = self._is_trivial_change(target_name, cand)
                if is_triv and trivial_added >= max_trivial_allowed:
                    continue

                if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
                    continue

                filtered_chunk.append(cand)
                is_trivial_flags.append(is_triv)
                heuristic_bonuses.append(bonus)

            if not filtered_chunk:
                continue

            semantically_valid = []

            if base_threshold > 0:
                prefixes = [ctx['local_prefix']] * len(filtered_chunk)
                suffixes = [ctx['local_suffix']] * len(filtered_chunk)

                cand_embs = self._get_variable_token_embeddings(prefixes, filtered_chunk, suffixes).to(
                    self.mlm_engine.device)
                sims = F.cosine_similarity(orig_emb, cand_embs)

                for cand, sim, is_trivial_change, bonus in zip(filtered_chunk, sims, is_trivial_flags,
                                                               heuristic_bonuses):
                    base_sim_score = sim.item()
                    final_score = base_sim_score + bonus

                    target_parts, _ = self._split_identifier(target_name)
                    cand_parts, _ = self._split_identifier(cand)

                    target_lower_parts = [p.lower() for p in target_parts]
                    cand_lower_parts = [p.lower() for p in cand_parts]

                    # 计算词根重合度 (是否保留了某个词语)
                    overlap_count = sum(1 for p in cand_lower_parts if p in target_lower_parts)
                    has_partial_overlap = overlap_count > 0

                    # ================= 动态阈值引擎 (路由分发) =================
                    current_threshold = base_threshold

                    if generator_source == 'LLM':
                        # LLM 生成的长变量 (多词)：容易发散，略微拉高阈值进行收束
                        if len(target_parts) > 1 or len(cand_parts) > 1:
                            current_threshold = max(0.75, base_threshold - 0.03)
                        # 防御：如果 LLM 生成的词极度雷同，说明在偷懒，稍微拉高要求
                        # if has_partial_overlap:
                        #     current_threshold = min(0.96, current_threshold + 0.02)

                    elif generator_source == 'MLM':
                        # MLM 局部微调：保留了部分词语的多词变量
                        if has_partial_overlap:
                            if len(target_parts) <= 3 :
                                # 只改了一部分，如果相似度还不高说明改坏了，较大幅度拉高阈值
                                current_threshold = min(0.98, base_threshold + 0.08)
                            # MLM 短变量替换：(如 pvma -> p_vma)，字面全换，适度降低阈值包容多样性
                            else:
                                current_threshold = min(0.99, base_threshold + 0.13)
                        elif len(target_parts) == 1:
                            current_threshold = base_threshold

                    else:
                        # Fallback (兜底原本的逻辑)
                        if is_trivial_change:
                            current_threshold = min(0.96, base_threshold + 0.08)

                    # =========================================================

                    is_pass = final_score >= current_threshold
                    status = "[PASS]" if is_pass else "[FILTERED]"

                    bonus_str = f"{bonus:>+.4f}" if hasattr(self, 'scorer') else "N/A"
                    # print(
                    #     f"{cand:<25} | {base_sim_score:.4f}  | {bonus_str:<10} | {final_score:.4f} | {current_threshold:.4f} | {status}")

                    if is_pass:
                        if is_trivial_change:
                            if trivial_added < max_trivial_allowed:
                                semantically_valid.append(cand)
                                trivial_added += 1
                        else:
                            semantically_valid.append(cand)
            else:
                semantically_valid = filtered_chunk

            # 最终的 AST 验证逻辑
            for cand in semantically_valid:
                if added >= quota:
                    break
                valid_cand = self._verify_ast_single(cand, ctx)
                if valid_cand and valid_cand not in final_candidates:
                    final_candidates.append(valid_cand)
                    added += 1

        if base_threshold > 0:
            # print("-" * 85 + "\n")
            pass

        return added

    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
        """利用 Tree-sitter AST 精准提取变量所在的最小逻辑语句。"""
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
        stop_parent_types = {
            'compound_statement', 'translation_unit', 'function_definition',
            'for_statement', 'while_statement', 'if_statement'
        }

        while statement_node.parent:
            if statement_node.parent.type in stop_parent_types:
                break
            statement_node = statement_node.parent

        stmt_start = statement_node.start_byte
        stmt_end = statement_node.end_byte

        local_prefix = code_bytes[stmt_start:target_start].decode("utf-8", errors="replace")
        local_suffix = code_bytes[target_end:stmt_end].decode("utf-8", errors="replace")

        return local_prefix, local_suffix

    def _build_llm_prompt(self, context_code: str, target_name: str, style: str, top_n: int, entity_type: str, n_parts: int) -> str:
        """
        统一的 LLM 提示词构建器：针对 1.5B 小模型优化。
        """
        # ================= 1. 实体词性约束 (简化语言，小模型更容易懂) =================
        if entity_type == 'VARIABLE':
            entity_rule = "Use NOUNS only (e.g., 'data_buffer'). NO verbs."
        elif entity_type == 'BOOLEAN_VAR':
            entity_rule = "Use BOOLEAN prefixes (e.g., 'is_', 'has_', 'can_')."
        else:
            entity_rule = "Use ACTION VERBS (e.g., 'get_data', 'update_state')."

        # ================= 2. 策略约束 (降低认知负担) =================
        if n_parts <= 2:
            max_allowed_parts = n_parts + 1
            strategy_instruction = f"""[Strategy: Short & Concise]
- MAX WORDS: {max_allowed_parts} words per name.
- EXAMPLES: "shm_info", "mem_data", "idx"
- Use common C/C++ abbreviations (ptr, buf, mem, val)."""
        else:
            strategy_instruction = f"""[Strategy: Semantic Refactoring]
- Provide professional synonyms matching the exact system logic.
- Keep the length similar to the original name."""

        # ================= 3. 组装最终 Prompt (修复代码块，增加示例与引导) =================
        # 注意：在末尾直接输出 ```json\n[ ，这叫做 Output Priming，能强制小模型开始写数组。
        return f"""You are an expert C/C++ developer. Suggest exactly {top_n} alternative names for `{target_name}`.

[Context Code]
```cpp
{context_code}
{strategy_instruction}

[Strict Rules]

{entity_rule}

STYLE: Use {style} naming convention.

NO generic names ("new_var", "temp").

[Task]
Output ONLY a JSON array containing EXACTLY {top_n} strings. Do not explain.
Example format for {top_n} items: ["name1", "name2", "name3", ...]

JSON
["""

    from typing import List, Dict, Any

    def generate_candidates(self, batch_tasks: List[Dict[str, Any]], top_k_mlm: int = 40, top_n_keep: int = 50) -> Dict[
        str, List[str]]:
        """
        批量候选词生成架构：极大提升 GPU 并行利用率
        :param batch_tasks: 列表，每个元素形如 {"target_name": "var_name", "code_str": "sliced_context"}
        :return: 字典，{ "var_name": ["cand1", "cand2", ...] }
        """
        results = {task["target_name"]: [] for task in batch_tasks}

        mlm_tracking = []
        llm_prompts = []
        llm_task_mapping = []
        task_metadata = {}

        mask_token = self.mlm_engine.tokenizer.mask_token

        # =========================================================
        # Phase 1: 请求收集与预处理
        # =========================================================
        for task_idx, task in enumerate(batch_tasks):
            target_name = task["target_name"]
            code_str = task["code_str"]
            code_bytes = code_str.encode("utf-8")

            identifiers = self.analyzer.extract_identifiers(code_bytes)
            if target_name not in identifiers:
                continue

            best_occ_idx = self._find_best_context_occurrence(code_bytes, identifiers[target_name])
            target_info = identifiers[target_name][best_occ_idx]

            raw_entity_type = target_info.get('entity_type', 'variable')
            entity_type = 'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE'
            if entity_type == 'VARIABLE' and target_name.startswith(('is_', 'has_', 'can_', 'should_')):
                entity_type = 'BOOLEAN_VAR'

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(target_name)
            n_parts = len(parts)

            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, target_info['start'],
                                                                         target_info['end'])

            task_metadata[task_idx] = {
                "target_name": target_name,
                "parts": parts,
                "style": style,
                "n_parts": n_parts,
                "identifiers": identifiers,
                "entity_type": entity_type,
                "original_style": original_style,
                "code_bytes": code_bytes,
                "local_prefix": local_prefix,
                "local_suffix": local_suffix,
                "raw_mlm_cands": [],
                "raw_llm_cands": []
            }

            # ---------------- 收集 MLM 任务 ----------------
            context_half = 700
            prefix_str = local_prefix[max(0, len(local_prefix) - context_half):]
            suffix_str = local_suffix[:context_half]

            variations = []
            if n_parts == 1:
                variations.append({'expand_mode': 'none', 'num_masks': 1, 'masked_str': mask_token})
                variations.append(
                    {'expand_mode': 'prefix', 'num_masks': 1, 'masked_str': f"{mask_token}_{target_name}"})
                variations.append(
                    {'expand_mode': 'suffix', 'num_masks': 1, 'masked_str': f"{target_name}_{mask_token}"})
            else:
                for i in range(n_parts):
                    masked_var = self._build_masked_string(parts, i, i + 1, 1, style, mask_token, target_name)
                    variations.append(
                        {'expand_mode': 'sub', 'start': i, 'end': i + 1, 'num_masks': 1, 'masked_str': masked_var})
                if n_parts >= 2:
                    for i in range(n_parts - 1):
                        masked_var = self._build_masked_string(parts, i, i + 2, 2, style, mask_token, target_name)
                        variations.append(
                            {'expand_mode': 'sub', 'start': i, 'end': i + 2, 'num_masks': 2, 'masked_str': masked_var})

            for var in variations:
                cropped_code = prefix_str + var['masked_str'] + suffix_str
                mlm_tracking.append({
                    "task_idx": task_idx,
                    "cropped_code": cropped_code,
                    "variation_info": var
                })

            # ---------------- 收集 LLM 任务 ----------------
            prompt = self._build_llm_prompt(
                context_code=code_str,
                target_name=target_name,
                style=original_style,
                top_n=int(top_n_keep * 1.5),
                entity_type=entity_type,
                n_parts=n_parts
            )
            llm_prompts.append(prompt)
            llm_task_mapping.append(task_idx)

        if not task_metadata:
            return results

        # =========================================================
        # Phase 2: GPU 并行推理
        # =========================================================
        all_cropped_codes = [item["cropped_code"] for item in mlm_tracking]
        batch_logits = None
        batch_mask_indices = None
        if all_cropped_codes:
            batch_logits, batch_mask_indices = self._get_model_logits_batched(all_cropped_codes)

        llm_responses = []
        if llm_prompts:
            try:
                llm_responses = self.llm_client.batch_chat(llm_prompts)
            except Exception as e:
                print(f"[!] LLM Batch Chat Failed: {e}")
                llm_responses = [""] * len(llm_prompts)

        # =========================================================
        # Phase 3: 结果分发与解析
        # =========================================================
        def _join_parts(new_parts, orig_name, st):
            if st == '_':
                return "_".join(new_parts)
            elif st == 'camel':
                return "".join(
                    p.lower() if j == 0 and orig_name[0].islower() else p.capitalize() for j, p in enumerate(new_parts))
            return "".join(new_parts)

        # 1. 分发 MLM 结果
        if batch_logits is not None:
            for i, track_info in enumerate(mlm_tracking):
                t_idx = track_info["task_idx"]
                var_info = track_info["variation_info"]
                meta = task_metadata[t_idx]

                logits = batch_logits[i:i + 1]
                mask_indices = batch_mask_indices[i]
                expand_mode = var_info.get('expand_mode', 'none')
                num_masks = var_info.get('num_masks', 1)

                if len(mask_indices) < num_masks: continue

                if num_masks == 1:
                    words = self._decode_words(logits[0, mask_indices[0], :], top_k_mlm, allow_underscore=False)
                    for w in words:
                        if expand_mode == 'prefix':
                            meta["raw_mlm_cands"].append(f"{w}_{meta['target_name']}")
                        elif expand_mode == 'suffix':
                            meta["raw_mlm_cands"].append(f"{meta['target_name']}_{w}")
                        else:
                            if meta["n_parts"] == 1:
                                meta["raw_mlm_cands"].append(w)
                            else:
                                new_parts = meta["parts"][:var_info['start']] + [w] + meta["parts"][var_info['end']:]
                                meta["raw_mlm_cands"].append(_join_parts(new_parts, meta["target_name"], meta["style"]))

                elif num_masks == 2:
                    top_k_2holes = min(4, max(2, top_k_mlm // 4))
                    words1 = self._decode_words(logits[0, mask_indices[0], :], top_k_2holes, allow_underscore=False)
                    words2 = self._decode_words(logits[0, mask_indices[1], :], top_k_2holes, allow_underscore=False)

                    for w1, w2 in itertools.product(words1, words2):
                        new_parts = meta["parts"][:var_info['start']] + [w1, w2] + meta["parts"][var_info['end']:]
                        meta["raw_mlm_cands"].append(_join_parts(new_parts, meta["target_name"], meta["style"]))

        # 2. 分发 LLM 结果
        for resp_idx, response in enumerate(llm_responses):
            t_idx = llm_task_mapping[resp_idx]
            meta = task_metadata[t_idx]

            parsed_cands = []
            if response and isinstance(response, str):
                # 1. 剔除常见的 Markdown 干扰代码块
                clean_text = response.replace("```json", "").replace("```", "").strip()

                # 2. 智能截取与补全：寻找第一个和最后一个引号
                first_quote = clean_text.find('"')
                last_quote = clean_text.rfind('"')

                # 确保找到了至少一对引号
                if first_quote != -1 and last_quote != -1 and first_quote != last_quote:
                    # 提取核心部分，例如截取出的结果是: "name1", "name2"
                    core_text = clean_text[first_quote:last_quote + 1]
                    # 强行包裹为标准 JSON 数组
                    patched_json = f"[{core_text}]"

                    try:
                        parsed_cands = json.loads(patched_json)
                        if not isinstance(parsed_cands, list):
                            parsed_cands = [str(parsed_cands)]
                    except Exception:
                        pass  # 解析失败则静默进入兜底逻辑

                # 3. 终极兜底策略：如果上面的 JSON 解析依然失败（或者根本没双引号），直接用正则硬抠
                if not parsed_cands:
                    # 兼容单双引号包裹的，且符合 C/C++ 标识符规范的字符串
                    parsed_cands = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)

            min_threshold = int(top_n_keep * 0.4)

            # print(f"LLM 原始生成结果 (Target: {meta['target_name']}):")

            # 双桶分流：严格符合长度的放 valid_cands，超长的放 oversized_cands
            valid_cands = []
            oversized_cands = []

            for c in parsed_cands:
                if isinstance(c, str) and c.strip():
                    clean_cand = c.strip()
                    # 假设你之前有个去重逻辑，防止模型输出重复的词
                    if clean_cand in valid_cands or clean_cand in oversized_cands:
                        continue

                    cand_parts_list, _ = self._split_identifier(clean_cand)

                    # 判断长度是否合法
                    if meta["n_parts"] <= 2:
                        is_valid_length = len(cand_parts_list) <= (meta["n_parts"] + 1)
                    else:
                        is_valid_length = len(cand_parts_list) <= (meta["n_parts"] + 2)

                    if is_valid_length:
                        valid_cands.append(clean_cand)
                    else:
                        oversized_cands.append(clean_cand)

            # 核心折中方案：检查合格数量是否达标
            if len(valid_cands) < min_threshold and oversized_cands:
                shortfall = min_threshold - len(valid_cands)
                # print(f"  ⚠️ 警告: 合格候选词数量 ({len(valid_cands)}) 低于最低阈值 ({min_threshold})。")
                # print(f"  🔄 正在从超长备用列表中捞取最多 {shortfall} 个候选词...")

                # 将超长的部分补充进有效列表，直到满足阈值（或者备用桶被掏空）
                fallback_cands = oversized_cands[:shortfall]
                valid_cands.extend(fallback_cands)

                for fc in fallback_cands:
                    # print(f"    -> 已打捞备用词: {fc}")
                    pass

            # 将最终存活的候选词存入 meta
            meta["raw_llm_cands"].extend(valid_cands)
            # print(f"  🎯 最终入库候选词数量: {len(valid_cands)}")
        # =========================================================
        # Phase 4: 应用配额漏斗过滤
        # =========================================================
        for t_idx, meta in task_metadata.items():
            unique_llm_cands = list(dict.fromkeys(meta["raw_llm_cands"]))
            unique_mlm_cands = list(dict.fromkeys(meta["raw_mlm_cands"]))

            quota_llm = max(1, int(top_n_keep * 0.40))
            quota_mlm = top_n_keep - quota_llm

            base_thresh = self.config.get('semantic_threshold', 0.85)
            current_base_thresh = max(0.75, base_thresh - 0.05) if meta["n_parts"] > 1 else base_thresh

            all_usages = meta["identifiers"][meta["target_name"]]
            found_return_type = next((u['return_type'] for u in all_usages if u.get('return_type')), None)

            ctx = {
                'code_bytes': meta["code_bytes"],
                'target_name': meta["target_name"],
                'identifiers': meta["identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],
                'semantic_threshold': current_base_thresh,
                'preserve_style': self.config.get('preserve_style', True),
                'entity_type': meta["entity_type"],
                'return_type': found_return_type
            }

            final_candidates = []

            self._verify_and_filter(unique_llm_cands, quota_llm, final_candidates, ctx, is_full_context=True,
                                    generator_source='LLM')

            mlm_actual_quota = top_n_keep - len(final_candidates)
            self._verify_and_filter(unique_mlm_cands, mlm_actual_quota, final_candidates, ctx, is_full_context=True,
                                    generator_source='MLM')

            if len(final_candidates) < top_n_keep:
                remaining_llm = [c for c in unique_llm_cands if c not in final_candidates]
                self._verify_and_filter(remaining_llm, top_n_keep - len(final_candidates), final_candidates, ctx,
                                        is_full_context=True, generator_source='LLM')

            results[meta["target_name"]] = final_candidates

        return results

    # def generate_candidates(self, code: str, target_name: str, identifiers=None) -> List[str]:
    #     """Performs standard context-aware candidate generation."""
    #     return self._generate_core(
    #         code=code,
    #         target_name=target_name,
    #         identifiers=identifiers,
    #         top_k_mlm=self.config['top_k_mlm'],
    #         top_n_keep=self.config['top_n_keep'],
    #         semantic_threshold=self.config['semantic_threshold'],
    #         context_ratio=self.config['context_ratio'],
    #         preserve_style=self.config['preserve_style'],
    #         strict_structure=False
    #     )

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