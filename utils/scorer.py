import json
import os
import nltk
from nltk import pos_tag


class StatisticalNamingScorer:
    """
    结合统计数据与 NLTK 词性标注的启发式打分器。
    【核心设计理念】：守门员模式。合法的命名不予加分（完全信任大模型的上下文理解），仅对严重违背命名规范的候选词施加惩罚。
    """

    def __init__(self, stats_file_path: str = 'naming_stats.json'):
        self.stats_file_path = stats_file_path
        self.stats = {}

        self._load_stats()
        self._warmup_nltk()

        # [扩充] 动词盲区补丁：涵盖 C/C++ 模块化命名中常出现的中间动作词
        self.VERB_BLIND_SPOTS = {
            'hash', 'run', 'read', 'write', 'load', 'save', 'init',
            'log', 'build', 'parse', 'bind', 'start', 'stop', 'cast',
            'add', 'fix', 'del', 'rm', 'calc', 'cmp', 'update', 'check',
            'alloc', 'free', 'pop', 'push', 'lock', 'unlock', 'clear', 'reset'
        }

        # [扩充] 名词盲区补丁：高频变量核心后缀
        self.NOUN_BLIND_SPOTS = {
            'log', 'hash', 'state', 'cache', 'count', 'size', 'len',
            'ptr', 'idx', 'buf', 'tmp', 'str', 'ret', 'val', 'msg', 'req', 'res'
        }

    def _load_stats(self):
        if os.path.exists(self.stats_file_path):
            try:
                with open(self.stats_file_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = {}
        else:
            self.stats = {}

    def _warmup_nltk(self):
        try:
            pos_tag(["warmup"])
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            pos_tag(["warmup"])

    def calculate_heuristic_score(self, cand_parts: list, entity_type: str) -> float:
        if not cand_parts:
            return 0.0

        score = 0.0
        first_word = cand_parts[0].lower()
        last_word = cand_parts[-1].lower()

        # ==========================================
        # 1. 致命规则拦截 (防抽风)
        # ==========================================
        if len(set(cand_parts)) != len(cand_parts):
            return -999.0

        # 获取当前实体的统计信息，用于判断词语是否“常见”
        entity_stats = self.stats.get(entity_type, {})
        # 兼容新的嵌套格式（prefixes/suffixes）
        known_prefixes = entity_stats.get('prefixes', {})
        known_suffixes = entity_stats.get('suffixes', {})

        # ==========================================
        # 2. 纯惩罚与极微弱辅助机制
        # ==========================================
        if entity_type == 'BOOLEAN_VAR':
            if first_word in ['is', 'has', 'can', 'should', 'will', 'was', 'did'] or \
                    last_word in ['flag', 'ok', 'status', 'success', 'enable', 'disable']:
                score += 0.005

        elif entity_type == 'FUNCTION':
            has_verb = False
            has_unknown_term = False

            tagged = pos_tag(cand_parts)

            for word, tag in tagged:
                w_lower = word.lower()
                # A. 确定是动词或在动词盲区
                if tag.startswith('VB') or w_lower in self.VERB_BLIND_SPOTS:
                    has_verb = True
                    break

                # B. 识别“未知/可疑词”：NLTK 认为是名词/形容词，但在统计库中闻所未闻
                # 这种情况通常是缩写（ioctl）或复合词（qmfb）
                # 我们认为：如果一个词在统计库前缀/后缀中出现的概率极低（< 0.001），且不是常见单词
                is_rare_in_stats = (known_prefixes.get(w_lower, 0) < 0.001 and
                                    known_suffixes.get(w_lower, 0) < 0.001)

                if is_rare_in_stats:
                    has_unknown_term = True

            if has_verb:
                # 明确合法的动词结构
                score += 0.0000
            elif has_unknown_term:
                # 包含无法确定的缩写/术语，选择“疑罪从无”，跳过惩罚
                # print(f"[NLP Debug] Function '{''.join(cand_parts)}' contains unknown terms, skipping penalty.")
                score += 0.0000
            else:
                # 只有当所有词都是 NLTK 确定的非动词（如纯名词组合），且都是常见词时，才施加惩罚
                score -= 0.03

        elif entity_type == 'VARIABLE':
            # 变量名逻辑保持原样，短变量豁免
            if len(cand_parts) == 1 and len(first_word) > 3:
                if first_word not in self.NOUN_BLIND_SPOTS:
                    tagged = pos_tag([first_word])
                    if tagged[0][1].startswith('VB'):
                        score -= 0.05

        return score