import json
import os
import nltk
from nltk import pos_tag


class StatisticalNamingScorer:
    """
    结合统计数据与 NLTK 词性标注的启发式打分器。
    引入函数语义护城河：严格把控 Getter, Setter, Bool 的转换。
    """

    def __init__(self, stats_file_path: str = 'naming_stats.json'):
        self.stats_file_path = stats_file_path
        self.stats = {}

        self._load_stats()
        self._warmup_nltk()

        # 动词/名词盲区
        self.VERB_BLIND_SPOTS = {
            'hash', 'run', 'read', 'write', 'load', 'save', 'init',
            'log', 'build', 'parse', 'bind', 'start', 'stop', 'cast',
            'add', 'fix', 'del', 'rm', 'calc', 'cmp', 'update', 'check',
            'alloc', 'free', 'pop', 'push', 'lock', 'unlock', 'clear', 'reset'
        }
        self.NOUN_BLIND_SPOTS = {
            'log', 'hash', 'state', 'cache', 'count', 'size', 'len',
            'ptr', 'idx', 'buf', 'tmp', 'str', 'ret', 'val', 'msg', 'req', 'res'
        }

        # ==========================================
        # [新增] 函数语义动作分类库
        # ==========================================
        self.GETTER_VERBS = {'get', 'fetch', 'read', 'query', 'retrieve', 'calc', 'compute', 'find', 'search'}
        self.SETTER_VERBS = {'set', 'write', 'update', 'assign', 'put', 'init', 'clear', 'reset'}
        self.BOOL_PREFIXES = {'is', 'has', 'can', 'should', 'will', 'was', 'did', 'check', 'allow'}

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

    def calculate_heuristic_score(self, cand_parts: list, entity_type: str, target_parts: list = None,
                                  return_type: str = None) -> float:
        """
        新增了 target_parts 和 return_type 参数，用于函数级语义校验。
        """
        if not cand_parts:
            return 0.0

        score = 0.0
        cand_first = cand_parts[0].lower()
        cand_last = cand_parts[-1].lower()

        target_first = target_parts[0].lower() if target_parts else ""

        # 致命规则拦截 (防抽风)
        if len(set(cand_parts)) != len(cand_parts):
            return -999.0

        entity_stats = self.stats.get(entity_type, {})
        known_prefixes = entity_stats.get('prefixes', {})
        known_suffixes = entity_stats.get('suffixes', {})

        # ==========================================
        # BOOLEAN_VAR 检查
        # ==========================================
        if entity_type == 'BOOLEAN_VAR':
            if cand_first in self.BOOL_PREFIXES or cand_last in ['flag', 'ok', 'status', 'success', 'enable',
                                                                 'disable']:
                score += 0.005
            elif len(cand_parts) > 1 and cand_first in self.VERB_BLIND_SPOTS and cand_first not in ['set', 'get']:
                score -= 0.1

        # ==========================================
        # FUNCTION 检查 (新增语义方向限制)
        # ==========================================
        elif entity_type == 'FUNCTION':
            is_target_getter = target_first in self.GETTER_VERBS
            is_target_setter = target_first in self.SETTER_VERBS
            is_target_bool = target_first in self.BOOL_PREFIXES

            is_cand_getter = cand_first in self.GETTER_VERBS
            is_cand_setter = cand_first in self.SETTER_VERBS
            is_cand_bool = cand_first in self.BOOL_PREFIXES

            # 【语义防线 1】：绝对互斥规则 (Getter 绝不能变 Setter，反之亦然)
            if is_target_getter and is_cand_setter:
                # print(f"[NLP 拦截] Getter '{'_'.join(target_parts)}' 不允许改为 Setter '{'_'.join(cand_parts)}'")
                return -999.0
            if is_target_setter and is_cand_getter:
                # print(f"[NLP 拦截] Setter '{'_'.join(target_parts)}' 不允许改为 Getter '{'_'.join(cand_parts)}'")
                return -999.0

            # 【语义防线 2】：布尔值匹配规则
            # 如果原函数是 bool 风格，候选不是 bool 风格且不是 getter，给予严重惩罚
            if is_target_bool and not (is_cand_bool or is_cand_getter):
                score -= 0.2
            # 如果候选词是 bool 风格，但原函数毫无 bool 痕迹，且明确知道返回值不是 bool/int
            if is_cand_bool and not is_target_bool:
                if return_type and return_type.lower() not in ['bool', 'boolean', 'int', '_bool']:
                    return -999.0

            # 【语义防线 3】：返回值与 Get/Set 的匹配
            if return_type:
                return_type_lower = return_type.lower()
                is_void = (return_type_lower == 'void')

                # Getter 限制：如果返回值是 void，绝不允许生造一个 getter（除非原函数就是奇葩的 void getter）
                if is_void and is_cand_getter and not is_target_getter:
                    return -999.0

                # Setter 限制：如果返回值非 void 且非 bool/int，绝不允许生造一个 setter（除非原函数本身就是 setter）
                if not is_void and return_type_lower not in ['bool', 'int'] and is_cand_setter and not is_target_setter:
                    # 返回指针或结构体的函数，突然变成 set_xxx，这是极其反常的
                    return -999.0

            # 原有的 NLTK 常规检查
            has_verb = False
            has_unknown_term = False
            tagged = pos_tag(cand_parts)

            for word, tag in tagged:
                w_lower = word.lower()
                if w_lower in self.VERB_BLIND_SPOTS or tag.startswith('VB'):
                    has_verb = True
                    break
                is_rare_in_stats = (known_prefixes.get(w_lower, 0) < 0.001 and known_suffixes.get(w_lower, 0) < 0.001)
                if is_rare_in_stats:
                    has_unknown_term = True

            if not has_verb and not has_unknown_term:
                score -= 0.08

        # ==========================================
        # VARIABLE 检查
        # ==========================================
        elif entity_type == 'VARIABLE':
            is_first_word_verb = False
            if cand_first in self.VERB_BLIND_SPOTS:
                is_first_word_verb = True
            elif cand_first not in self.NOUN_BLIND_SPOTS:
                tagged = pos_tag(cand_parts)
                if tagged and tagged[0][1].startswith('VB'):
                    is_first_word_verb = True

            safe_verb_nouns = {'request', 'reply', 'result', 'record', 'return', 'state', 'cache', 'count', 'limit'}
            if cand_first in safe_verb_nouns:
                is_first_word_verb = False

            if is_first_word_verb:
                return -999.0

        return score