import re
import random
import torch
from typing import List, Dict
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from collections import defaultdict


class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo, analyzer, difficulty=1):
        """
        :param model_zoo: 包含 MLM 模型和 Tokenizer 的对象
        :param analyzer: IdentifierAnalyzer 实例，用于 AST 分析
        :param difficulty: 难度等级 (1: 仅MLM单单词, 2: MLM+驼峰, 3: MLM+驼峰+下划线)
        """
        self.model_zoo = model_zoo
        self.analyzer = analyzer
        # 常用的编程后缀，用于构建复合词空间
        self.common_affixes = ['ptr', 'buf', 'val', 'idx', 'msg', 'data', 'tmp', 'ref', 'res', 'list', 'obj', 'item', 'ctx']

    def _to_pascal_case(self, words: List[str]) -> str:
        """PascalCase: ['my', 'data', 'ptr'] -> MyDataPtr"""
        return "".join(word.capitalize() for word in words)

    def _to_snake_case(self, words: List[str]) -> str:
        """snake_case: ['my', 'data', 'ptr'] -> my_data_ptr"""
        return "_".join(word.lower() for word in words)

    def generate_candidates(self, code: str, target_name: str, identifiers=None, top_k_mlm=100, top_n_keep=20) -> List[
        str]:
        keywords = self.analyzer.keywords
        code_bytes = code.encode("utf-8")

        if identifiers is None:
            identifiers = self.analyzer.extract_identifiers(code_bytes)

        if target_name not in identifiers:
            return []

        # --- 1. MLM 模型预测 (获取语义种子词) ---
        target_info = identifiers[target_name][0]
        start_byte, end_byte = target_info['start'], target_info['end']

        mask_token = self.model_zoo.mlm_tokenizer.mask_token
        protected_mask = f" {mask_token} "
        mask_token_bytes = protected_mask.encode("utf-8")
        masked_code_bytes = code_bytes[:start_byte] + mask_token_bytes + code_bytes[end_byte:]

        # 上下文截断处理
        context_half_size = 700
        crop_start = max(0, start_byte - context_half_size)
        crop_end = min(len(masked_code_bytes), start_byte + len(mask_token_bytes) + context_half_size)
        cropped_code = masked_code_bytes[crop_start:crop_end].decode("utf-8", errors="replace")

        inputs = self.model_zoo.mlm_tokenizer(
            cropped_code, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model_zoo.device)

        mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
        mask_token_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_token_indices) == 0:
            return []

        with torch.no_grad():
            logits = self.model_zoo.mlm_model(**inputs).logits
            mask_logits = logits[0, mask_token_indices[0], :]
            _, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)
            raw_candidates = [self.model_zoo.mlm_tokenizer.decode([idx]).strip() for idx in top_k_indices]

        base_seeds = []
        for p in raw_candidates:
            cand = p.replace('Ġ', '').replace('##', '').strip()
            cand = re.sub(r'[^a-zA-Z0-9_]', '', cand)
            if cand and len(cand) > 1 and (cand[0].isalpha() or cand[0] == '_'):
                base_seeds.append(cand)

        # --- 2. 增强种子池 ---
        # 将模型预测的词和常用词缀合并，作为“单词工厂”的原材料
        word_factory = list(dict.fromkeys(base_seeds[:30] + self.common_affixes))

        raw_pool = []
        # 保持一部分原始种子词
        raw_pool.extend(base_seeds[:20])

        # --- 3. 核心优化：随机多单词拼接 ---
        # 尝试生成 60 个复合词，最后再通过打乱和去重筛选
        for _ in range(60):
            # 随机决定拼接 2 个还是 3 个单词
            num_words = random.choice([2, 3, 4, 5])
            # 从工厂中随机抽取单词
            selected_words = random.sample(word_factory, num_words)

            # 随机决定风格：PascalCase 或 Snake_Case
            if random.random() > 0.5:
                raw_pool.append(self._to_pascal_case(selected_words))
            else:
                raw_pool.append(self._to_snake_case(selected_words))

        # --- 4. 打乱与验证 ---
        unique_pool = list(dict.fromkeys(raw_pool))
        random.shuffle(unique_pool)

        valid_candidates = []
        for cand in unique_pool:
            if cand in keywords or cand == target_name: continue
            if not self.analyzer.can_rename_to(code_bytes, target_name, cand): continue
            try:
                _ = CodeTransformer.validate_and_apply(code_bytes, identifiers, {target_name: cand},
                                                       analyzer=self.analyzer)
                valid_candidates.append(cand)
            except:
                continue
            if len(valid_candidates) >= top_n_keep: break

        return valid_candidates

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