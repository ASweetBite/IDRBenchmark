import itertools
import json
import random
import re
from collections import defaultdict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.ast_tools import CodeTransformer


class HeavyWeightCandidateGenerator:
    def __init__(self, embedder, llm_client, analyzer, config):
        """
        初始化重量级候选词生成器 (仅依赖 LLM 进行深度语义改写)
        :param embedder: 用于提取 Token Embedding 和计算相似度的模型 (原 mlm_engine)
        :param llm_client: 用于生成候选词的大模型客户端
        :param analyzer: AST 语法树与上下文分析器
        :param config: 全局配置
        """
        self.embedder = embedder
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.config = config
        stats_path = config.get('naming_stats_path', 'naming_stats.json')

        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    # =========================================================================
    # 基础工具与命名规范检测
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

    # =========================================================================
    # AST 上下文与向量相似度验证
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

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 64) -> torch.Tensor:
        """精准提取Token级变量语义向量（仅用于计算 LLM 候选词的相似度）"""
        all_embeddings = []
        tokenizer = self.embedder.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]

        device = self.embedder.device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[
            0] >= 8 else torch.float16
        self.embedder.model.to(dtype)

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = self.embedder.model(**inputs, output_hidden_states=True)
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

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        try:
            from utils.ast_tools import CodeTransformer  # 确保导入路径正确
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'], {ctx['target_name']: cand},
                                               analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _is_trivial_change(self, target_name: str, cand: str) -> bool:
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

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
            ).to(self.embedder.device)

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
                    self.embedder.device)
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
    # LLM 核心交互逻辑
    # =========================================================================
    def _build_llm_prompt(self, context_code: str, target_name: str, style: str, top_n: int, entity_type: str,
                          n_parts: int) -> str:
        if entity_type == 'VARIABLE':
            entity_rule = "Use NOUNS only (e.g., 'data_buffer'). NO verbs."
        elif entity_type == 'BOOLEAN_VAR':
            entity_rule = "Use BOOLEAN prefixes (e.g., 'is_', 'has_', 'can_')."
        else:
            entity_rule = "Use ACTION VERBS (e.g., 'get_data', 'update_state')."

        if n_parts <= 2:
            max_allowed_parts = n_parts + 1
            strategy_instruction = f"""[Strategy: Short & Concise]
- MAX WORDS: {max_allowed_parts} words per name.
- EXAMPLES: "shm_info", "mem_data", "idx"
- Use common C/C++ abbreviations (ptr, buf, mem, val)."""
        else:
            strategy_instruction = """[Strategy: Semantic Refactoring]
- Provide professional synonyms matching the exact system logic.
- Keep the length similar to the original name."""

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


    def generate_candidates(self, vulnerable_tasks: List[Dict[str, Any]], target_quota: int = 20) -> Dict[str, List[str]]:
        """
        专门为 RNNS 选出的薄弱点调用 LLM 深度生成。
        :param vulnerable_tasks: 仅包含需要重度打击的 Target 节点
        :param target_quota: 需要 LLM 生成的目标合法词汇数量
        """
        results = {task["target_name"]: [] for task in vulnerable_tasks}

        llm_prompts = []
        task_metadata = {}

        # 1. 解析任务并构建 Prompt
        for task_idx, task in enumerate(vulnerable_tasks):
            target_name = task["target_name"]
            code_str = task["code_str"]
            code_bytes = code_str.encode("utf-8")

            identifiers = self.analyzer.extract_identifiers(code_bytes)
            if target_name not in identifiers: continue

            best_occ_idx = self._find_best_context_occurrence(code_bytes, identifiers[target_name])
            target_info = identifiers[target_name][best_occ_idx]

            raw_entity_type = target_info.get('entity_type', 'variable')
            entity_type = 'BOOLEAN_VAR' if target_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE')

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(target_name)
            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, target_info['start'],
                                                                         target_info['end'])

            task_metadata[task_idx] = {
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "identifiers": identifiers, "entity_type": entity_type, "original_style": original_style,
                "code_bytes": code_bytes, "local_prefix": local_prefix, "local_suffix": local_suffix
            }

            # 生成两倍于 target_quota 的词以备过滤消耗
            prompt = self._build_llm_prompt(code_str, target_name, original_style, target_quota * 2, entity_type,
                                            len(parts))
            llm_prompts.append(prompt)

        if not llm_prompts: return results

        # 2. 批量请求 LLM
        try:
            llm_responses = self.llm_client.batch_chat(llm_prompts)
        except Exception as e:
            print(f"[!] LLM Batch Chat Failed: {e}")
            llm_responses = [""] * len(llm_prompts)

        # 3. 解析与过滤
        for resp_idx, response in enumerate(llm_responses):
            meta = task_metadata[resp_idx]
            parsed_cands = []

            if response and isinstance(response, str):
                clean_text = response.replace("```json", "").replace("```", "").strip()
                first_quote, last_quote = clean_text.find('"'), clean_text.rfind('"')

                if first_quote != -1 and last_quote != -1 and first_quote != last_quote:
                    patched_json = f"[{clean_text[first_quote:last_quote + 1]}]"
                    try:
                        parsed_cands = json.loads(patched_json)
                        if not isinstance(parsed_cands, list): parsed_cands = [str(parsed_cands)]
                    except Exception:
                        pass

                if not parsed_cands:
                    parsed_cands = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)

            valid_cands, oversized_cands = [], []
            for c in parsed_cands:
                if isinstance(c, str) and c.strip():
                    clean_cand = c.strip()
                    if clean_cand in valid_cands or clean_cand in oversized_cands: continue

                    cand_parts_list, _ = self._split_identifier(clean_cand)
                    limit = meta["n_parts"] + 1 if meta["n_parts"] <= 2 else meta["n_parts"] + 2

                    if len(cand_parts_list) <= limit:
                        valid_cands.append(clean_cand)
                    else:
                        oversized_cands.append(clean_cand)

            min_threshold = int(target_quota * 0.8)
            if len(valid_cands) < min_threshold and oversized_cands:
                valid_cands.extend(oversized_cands[:min_threshold - len(valid_cands)])

            # 应用基于 Embedding 的过滤系统
            ctx = {
                'code_bytes': meta["code_bytes"], 'target_name': meta["target_name"], 'identifiers': meta["identifiers"],
                'keywords': self.analyzer.keywords, 'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"], 'semantic_threshold': self.config.get('semantic_threshold', 0.85),
                'preserve_style': self.config.get('preserve_style', True), 'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["identifiers"][meta["target_name"]] if u.get('return_type')), None)
            }

            final_candidates = []
            self._verify_and_filter(valid_cands, target_quota, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates
            # print(
            #     f"      ↳ [LLM Gen] Var: '{meta['target_name']}' | Parsed: {len(valid_cands)} | Final Kept: {len(final_candidates)}/{target_quota}")
        return results