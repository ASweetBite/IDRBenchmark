import re
from collections import defaultdict
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp


class IdentifierAnalyzer:
    def __init__(self, lang="cpp"):
        # ... 初始化 tree-sitter 语言 (保持你的原样) ...
        if lang == "c":
            from tree_sitter_c import language as ts_c
            self.language = Language(ts_c())
        elif lang == "cpp":
            from tree_sitter_cpp import language as ts_cpp
            self.language = Language(ts_cpp())
        else:
            raise ValueError("Unsupported language. Choose 'c' or 'cpp'.")

        self.parser = Parser()
        self.parser.language = self.language

        # 【修复 3】扩充 MLM 易生成的 C/C++ 常见类型和系统级标识符
        self.keywords = {
            # 基础关键字 (原样保留)
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen", "malloc", "free",
            "memset", "memcpy", "fopen", "fclose", "bool", "true", "false", "NULL",
            "class", "public", "private", "protected", "template", "new", "delete",
            "catch", "try", "namespace", "using", "cout", "cin", "std", "endl", "auto",
            "const", "constexpr", "virtual", "override", "final", "this", "nullptr",
            "string", "vector", "map", "set", "list", "cerr",

            # ---> 新增：修饰符与底层类型 <---
            "static", "extern", "inline", "struct", "union", "enum", "typedef",
            "short", "long", "unsigned", "signed", "register", "volatile",

            # ---> 新增：标准库极易生成的类型别名 <---
            "size_t", "ssize_t", "FILE", "DIR", "pid_t",
            "int8_t", "uint8_t", "int16_t", "uint16_t",
            "int32_t", "uint32_t", "int64_t", "uint64_t", "EOF"
        }

    def extract_identifiers(self, source_code: bytes) -> dict:
        tree = self.parser.parse(source_code)
        identifiers = defaultdict(list)

        scope_stack = [{
            "id": 0,
            "start": 0,
            "end": len(source_code)
        }]
        scope_counter = 0

        # 【修复 1】增加 function_definition 和 for_statement，防止参数和局部变量泄露到全局
        scope_nodes = {
            'compound_statement',
            'class_specifier',
            'namespace_definition',
            'struct_specifier',
            'function_definition',  # <-- 圈住整个函数(包含参数)
            'for_statement'  # <-- 圈住 for 循环内部声明
        }

        # 【修复 2】增加预处理器宏定义，防止更改宏引发编译故障
        excluded_parents = {
            'function_declarator',
            'field_identifier',
            'field_declaration',
            'namespace_identifier',
            'preproc_def',  # <-- 排除 #define 的宏名
            'preproc_function_def'  # <-- 排除带参数的宏名
        }

        def traverse(node):
            nonlocal scope_counter

            entered_scope = False
            if node.type in scope_nodes:
                scope_counter += 1
                scope_stack.append({
                    "id": scope_counter,
                    "start": node.start_byte,
                    "end": node.end_byte
                })
                entered_scope = True

            if node.type == "identifier":
                parent_type = node.parent.type if node.parent else None
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                # C/C++ 排除 call_expression 本身 (函数调用)
                is_function_call = (
                        parent_type == "call_expression" and
                        node.parent.child_by_field_name('function') == node
                )

                if (name not in self.keywords and
                        parent_type not in excluded_parents and
                        not is_function_call):
                    current_scope = scope_stack[-1]
                    identifiers[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "scope": current_scope["id"],
                        "scope_start": current_scope["start"],
                        "scope_end": current_scope["end"]
                    })

            for child in node.children:
                traverse(child)

            if entered_scope:
                scope_stack.pop()

        traverse(tree.root_node)
        return dict(identifiers)

    def get_identifier_scope_ranges(self, source_code: bytes, var_name: str):
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        identifiers = self.extract_identifiers(source_code)

        if old_name == new_name:
            return False

        if new_name not in identifiers:
            return True

        if old_name not in identifiers:
            return False

        old_scopes = {(p["scope_start"], p["scope_end"]) for p in identifiers[old_name]}
        new_scopes = {(p["scope_start"], p["scope_end"]) for p in identifiers[new_name]}

        for oscope in old_scopes:
            for nscope in new_scopes:
                if self.scopes_overlap(oscope, nscope):
                    return False

        return True


def is_valid_identifier(name: str) -> bool:
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict, analyzer=None) -> str:
        for old_name, new_name in renaming_map.items():
            if not is_valid_identifier(new_name):
                raise ValueError(f"命名不合法: '{new_name}'")

            if analyzer is not None:
                if not analyzer.can_rename_to(source_code, old_name, new_name):
                    raise ValueError(f"作用域冲突: '{old_name}' -> '{new_name}' 不可用。")
            else:
                existing_names = set(identifiers.keys())
                if new_name in existing_names and new_name != old_name:
                    raise ValueError(f"重命名冲突: '{old_name}' -> '{new_name}' 已存在。")

        code = bytearray(source_code)
        replacements = []
        for old_name, new_name in renaming_map.items():
            if old_name in identifiers:
                for pos in identifiers[old_name]:
                    replacements.append((pos['start'], pos['end'], new_name))

        # 从后往前替换，防止偏移量变化导致位置错误
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, new_name in replacements:
            code[start:end] = new_name.encode("utf-8")

        return code.decode("utf-8")