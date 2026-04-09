import itertools
import random
import re
from collections import defaultdict
from typing import List, Any

import torch
import torch.nn.functional as F

from utils.ast_tools import CodeTransformer


class HeavyWeightCandidateGenerator:
    def __init__(self, mlm_engine, analyzer):
        self.mlm_engine = mlm_engine
        self.analyzer = analyzer

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

    def _get_word_embedding(self, word: str) -> Any | None:
        tokenizer = self.mlm_engine.tokenizer
        tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(tokens) == 0:
            return None
        tokens = tokens.to(self.mlm_engine.device)
        with torch.no_grad():
            embeddings = self.mlm_engine.model.get_input_embeddings()(tokens)
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

    def _get_model_logits(self, prefix: bytes, masked_var_name_bytes: bytes, suffix: bytes):
        """提取的公共组件 1：处理截断、分词和模型推理"""
        masked_code_bytes = prefix + masked_var_name_bytes + suffix
        mask_start = len(prefix)
        mask_end = mask_start + len(masked_var_name_bytes)
        context_half = 700

        crop_start = max(0, mask_start - context_half)
        crop_end = min(len(masked_code_bytes), mask_end + context_half)
        cropped_code = masked_code_bytes[crop_start:crop_end].decode("utf-8", errors="replace")

        inputs = self.mlm_engine.tokenizer(
            cropped_code, return_tensors="pt", truncation=True, max_length=512
        ).to(self.mlm_engine.device)

        mask_token_id = self.mlm_engine.tokenizer.mask_token_id
        mask_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) == 0:
            return None, []

        with torch.no_grad():
            logits = self.mlm_engine.model(**inputs).logits

        return logits, mask_indices

    def _decode_words(self, mask_logits, top_k, allow_underscore=False, required_length=None):
        """提取的公共组件 2：解码 Token 并执行硬性长度与字符过滤"""
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

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx, is_full_context=False):
        """提取的公共组件 3：统一的语义和 AST 验证通道"""
        added = 0
        for cand in candidate_list:
            if added >= quota: break
            if cand in ctx['keywords'] or cand == ctx['target_name']: continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand): continue

            # 语义校验 (全掩码猜测可跳过)
            if not is_full_context and ctx['semantic_threshold'] > 0 and ctx['original_emb'] is not None:
                cand_emb = self._get_word_embedding(cand)
                if cand_emb is not None:
                    sim = F.cosine_similarity(ctx['original_emb'].unsqueeze(0), cand_emb.unsqueeze(0)).item()
                    if sim < ctx['semantic_threshold']: continue

            # AST 物理冲突校验
            if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand): continue
            try:
                CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'],
                                                   {ctx['target_name']: cand}, analyzer=self.analyzer)
                if cand not in final_candidates:
                    final_candidates.append(cand)
                    added += 1
            except Exception:
                continue
        return added

    def _generate_core(self, code: str, target_name: str, identifiers: dict,
                       top_k_mlm: int, top_n_keep: int, semantic_threshold: float,
                       context_ratio: float, preserve_style: bool, strict_structure: bool) -> List[str]:
        """合并后的核心生成流水线"""

        # ==========================================
        # 新增优化：前置拦截低价值/不合法的变量名
        # ==========================================
        # 1. 过滤单字母变量 (忽略前后下划线后的核心长度，例如 '_x' 也会被过滤)
        if len(target_name.strip('_')) <= 1:
            return []

        # 2. 过滤 Python 内置的魔法方法/系统级变量 (如 __init__, __dict__)
        if target_name.startswith('__') and target_name.endswith('__'):
            return []

        code_bytes = code.encode("utf-8")
        if identifiers is None:
            identifiers = self.analyzer.extract_identifiers(code_bytes)

        if target_name not in identifiers: return []

        original_style = self._detect_naming_style(target_name)
        original_emb = self._get_word_embedding(target_name) if semantic_threshold > 0 else None

        target_info = identifiers[target_name][0]
        prefix = code_bytes[:target_info['start']]
        suffix = code_bytes[target_info['end']:]

        parts, style = self._split_identifier(target_name)
        n_parts = len(parts)

        # 3. 过滤超长变量名：如果词块数量过多（比如超过10个词拼接），MLM联合猜测极易引发组合爆炸，直接拦截
        if n_parts > 10:
            return []

        target_lengths = [len(p) for p in parts]
        mask_token = self.mlm_engine.tokenizer.mask_token

        # 1. 动态构建掩码变体 (Strategy Pattern)
        variations = []
        if strict_structure:
            variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': n_parts})
            if n_parts > 1:
                for s in range(n_parts):
                    for e in range(s + 1, n_parts + 1):
                        if s == 0 and e == n_parts: continue
                        variations.append({'type': 'sub', 'start': s, 'end': e, 'num_masks': e - s})
        else:
            variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': 1})
            if n_parts > 1:
                variations.append({'type': 'full', 'start': 0, 'end': n_parts, 'num_masks': n_parts})
                for s in range(n_parts):
                    for e in range(s + 1, n_parts + 1):
                        if s == 0 and e == n_parts: continue
                        variations.append({'type': 'sub', 'start': s, 'end': e, 'num_masks': 1})
                        if (e - s) > 1:
                            variations.append({'type': 'sub', 'start': s, 'end': e, 'num_masks': e - s})

        # 2. 执行模型预测与解码
        raw_full_cands, raw_sub_cands_lists = [], []
        for var in variations:
            masked_var = self._build_masked_string(parts, var['start'], var['end'], var['num_masks'], style, mask_token,
                                                   target_name)
            logits, mask_indices = self._get_model_logits(prefix, masked_var.encode("utf-8"), suffix)

            if logits is None or len(mask_indices) < var['num_masks']: continue

            current_cands = []
            if var['num_masks'] == 1 and not strict_structure:
                words = self._decode_words(logits[0, mask_indices[0], :], top_k_mlm, allow_underscore=True)
                for w in words:
                    current_cands.append(
                        self._assemble_multi_candidate(parts, var['start'], var['end'], (w,), style, target_name))
            else:
                per_mask_top_k = min(10, max(3, top_k_mlm // (var['num_masks'] * 2)))
                expanded_top_k = top_k_mlm * 5 if strict_structure else per_mask_top_k
                mask_preds = []

                for m_idx in range(var['num_masks']):
                    part_idx = var['start'] + m_idx if strict_structure else None
                    req_len = target_lengths[part_idx] if strict_structure else None

                    words = self._decode_words(logits[0, mask_indices[m_idx], :], expanded_top_k,
                                               allow_underscore=False, required_length=req_len)
                    words = words[:per_mask_top_k]  # 截断回正常阈值

                    # 退化保护机制
                    if strict_structure and not words:
                        words = [parts[part_idx]]
                    elif not strict_structure and not words:
                        words = ['temp']
                    mask_preds.append(words)

                # 笛卡尔积产生联合组合
                for combo in itertools.product(*mask_preds):
                    cand = self._assemble_multi_candidate(parts, var['start'], var['end'], combo, style, target_name)
                    if strict_structure:
                        if [len(p) for p in self._split_identifier(cand)[0]] == target_lengths:
                            current_cands.append(cand)
                    else:
                        current_cands.append(cand)

            if var['type'] == 'full':
                raw_full_cands.extend(current_cands)
            else:
                raw_sub_cands_lists.append(current_cands)

        # 3. 收集与整理去重
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

        # 4. AST与语意过滤
        ctx = {
            'code_bytes': code_bytes, 'target_name': target_name, 'identifiers': identifiers,
            'keywords': self.analyzer.keywords, 'original_style': original_style,
            'original_emb': original_emb, 'semantic_threshold': semantic_threshold, 'preserve_style': preserve_style
        }
        final_candidates = []
        target_full_quota = int(top_n_keep * context_ratio)

        actual_full = self._verify_and_filter(unique_full, target_full_quota, final_candidates, ctx,
                                              is_full_context=True)
        self._verify_and_filter(unique_sub, top_n_keep - actual_full, final_candidates, ctx, is_full_context=False)

        # 5. 模式特有兜底 (降级预案)
        if strict_structure and len(final_candidates) < top_n_keep:
            local_cands = self._generate_structural_fallback(code_bytes, target_name, identifiers,
                                                             top_n_keep - len(final_candidates))
            for lc in local_cands:
                if lc not in final_candidates and self.analyzer.can_rename_to(code_bytes, target_name, lc):
                    final_candidates.append(lc)

        if len(final_candidates) > top_n_keep:
            return random.sample(final_candidates, top_n_keep)
        return final_candidates

    # ==========================================
    # 极度精简后的公共 API 接口
    # ==========================================
    def generate_candidates(self, code: str, target_name: str, identifiers=None,
                            top_k_mlm=60, top_n_keep=50, preserve_style: bool = True,
                            semantic_threshold: float = 0.2, context_ratio: float = 0.3) -> List[str]:
        """模式一：普通生成"""
        return self._generate_core(
            code, target_name, identifiers, top_k_mlm, top_n_keep,
            semantic_threshold, context_ratio, preserve_style, strict_structure=False
        )

    def generate_structural_candidates(self, code: str, target_name: str, identifiers=None,
                                       top_k_mlm=60, top_n_keep=50,
                                       semantic_threshold: float = 0.5, context_ratio: float = 0.3) -> List[str]:
        """模式二：同构生成（强制一致性风格与严格结构）"""
        return self._generate_core(
            code, target_name, identifiers, top_k_mlm, top_n_keep,
            semantic_threshold, context_ratio, preserve_style=True, strict_structure=True
        )

    # ==========================================
    # 模式二：降级保护兜底 (原始结构拼接逻辑)
    # ==========================================
    def _generate_structural_fallback(self, code_bytes: bytes, target_name: str, identifiers: dict, num_needed: int) -> \
    List[str]:
        """由于长度和语义校验过于严苛，在候选项不足时，使用本地标识符词库进行无脑盲拼兜底"""
        format_info = self.analyzer.analyze_format(target_name)
        target_lengths = format_info['lengths']
        local_names = list(identifiers.keys())

        # 直接使用本地文件内的变量词碎块构建长度池
        pool = defaultdict(set)
        for name in local_names:
            parts = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)|[a-z0-9]+', name)
            for part in parts:
                if len(part) > 0: pool[len(part)].add(part.lower())

        length_pool = {length: list(words) for length, words in pool.items()}

        fallback_candidates = []
        for _ in range(50):  # 试拼 50 次
            sampled_words = []
            for length in target_lengths:
                if length in length_pool and length_pool[length]:
                    sampled_words.append(random.choice(length_pool[length]))
                else:
                    break

            if len(sampled_words) == len(target_lengths):
                # 按原风格拼接
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
        """
        通用的 MLM 单词预测逻辑：返回最适合 target_name 位置的原始单词列表
        """
        mask_token = self.mlm_engine.tokenizer.mask_token
        # 1. 掩码替换
        pattern = rf'\b{re.escape(target_name)}\b'.encode()
        masked_code_bytes = re.sub(pattern, mask_token.encode(), code_bytes, count=1)

        # 2. 裁剪上下文 (参考你原有的逻辑)
        # ... (此处省略部分裁剪逻辑，建议直接复用你 generate_candidates 里的截断代码)
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
            word = re.sub(r'[^a-z]', '', word)  # 只保留纯字母
            if len(word) > 1:
                words.append(word)
        return words

    def _infer_type_from_code(self, code: str, target_name: str) -> str:
        """
        增强版正则回溯：支持多变量声明、无空格指针等复杂 C/C++ 语法
        """
        # 匹配模式升级：
        # 1. [\w\s\*,&:]*? 允许类型名中包含逗号(多变量)、&号(引用)、::(命名空间)
        # 2. \s* 允许变量名前没有空格 (例如 int *var)
        pattern = r'([a-zA-Z_][\w\s\*,&:]*?)\s*\b' + re.escape(target_name) + r'\b\s*[\[=;,)]'
        match = re.search(pattern, code)

        if match:
            type_part = match.group(1).strip()

            # --- 脏数据清洗 ---
            # 如果匹配到了 "int a, b, "，我们只取逗号最前面的基础类型 "int a"
            if ',' in type_part:
                type_part = type_part.split(',')[0].strip()
                # 去掉多余的变量名，比如 "int a" -> 变成 "int"
                type_part = ' '.join([word for word in type_part.split() if word in
                                      ["int", "long", "short", "char", "float", "double", "unsigned", "signed",
                                       "struct", "class"]])

            # 清理掉 static, const, inline 等修饰符
            type_part = re.sub(r'\b(static|const|inline|extern|volatile|register)\b', '', type_part).strip()

            # 如果清洗完不为空，返回类型
            if type_part:
                return type_part

        # 函数参数定义兜底匹配 (int a, char *b)
        param_pattern = r'([a-zA-Z_][\w\s\*,&:]*?)\s*\b' + re.escape(target_name) + r'\b\s*[,)]'
        match = re.search(param_pattern, code)
        if match:
            type_part = match.group(1).strip()
            type_part = re.sub(r'\b(const)\b', '', type_part).strip()
            if type_part:
                return type_part

        # 如果实在找不到，返回 void
        return "void"

    def generate_normalized_name(self, code: str, target_name: str, var_type: str, excluded_names: set) -> str:
        """
        完全从 code 中推断类型并生成名字
        支持规则：
        1. 函数 -> fun_mask
        2. 类实例 -> 类名_mask
        3. 基础变量 -> int_mask / pointer_mask / char_mask
        """

        # --- 【新增规则】：判断是否为函数 ---
        # 如果目标名字在代码中以 "名字(" 或 "名字 (" 的形式出现，则判定为函数
        if re.search(r'\b' + re.escape(target_name) + r'\s*\(', code):
            category = "fun"

        else:
            # --- 原有的变量推断逻辑 ---
            inferred_type = self._infer_type_from_code(code, target_name)

            # 1. 提取核心类型并清理修饰符
            clean_type = re.sub(r'\b(struct|class|enum|union)\b', '', inferred_type).strip()
            is_pointer = "*" in inferred_type or "*" in clean_type

            # 提取最后一个词作为核心词
            core_type = clean_type.replace("*", "").replace("&", "").strip()
            core_type = core_type.split()[-1] if core_type else "void"

            primitives_int = ["int", "long", "short", "size_t", "float", "double", "unsigned", "signed", "uint32_t",
                              "uint64_t", "int32_t", "uint8_t"]
            primitives_char = ["char"]

            # 2. 确定类别前缀 (category)
            if core_type == "void":
                category = "var"
            elif core_type.lower() not in (primitives_int + primitives_char + ["bool"]):
                # 处理 C++ 命名空间
                if "::" in core_type:
                    category = core_type.split("::")[-1]
                else:
                    category = core_type

                # 清洗非合法字符
                category = re.sub(r'\W+', '', category)

                # 数字防崩溃拦截
                if not category or category.isdigit():
                    category = "obj"
                elif category[0].isdigit():
                    category = "v" + category
            else:
                # 基础类型逻辑
                if is_pointer:
                    category = "pointer"
                elif core_type.lower() in primitives_char:
                    category = "char"
                else:
                    category = "int"

        # --- 3. 获取 MLM 预测词并拼接 ---
        code_str = code if isinstance(code, str) else code.decode('utf-8')
        candidate_words = self.get_top_mlm_words(code_str.encode(), target_name)

        # 4. 筛选一个未被使用的词
        for w in candidate_words:
            # 过滤掉非法的变量名字符
            w = re.sub(r'\W+', '', w)
            if not w: continue

            potential_name = f"{category}_{w}"
            if potential_name not in excluded_names:
                return potential_name

        # 5. 如果都冲突了，加数字后缀兜底
        return f"{category}_{len(excluded_names)}"