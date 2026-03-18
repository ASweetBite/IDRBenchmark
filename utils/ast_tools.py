import re
from collections import defaultdict
from tree_sitter import Language, Parser
import tree_sitter_c


class IdentifierAnalyzer:
    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.keywords = {
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen"
        }

    def parse(self, source_code: bytes):
        return self.parser.parse(source_code)

    def extract_identifiers(self, source_code: bytes) -> dict:
        """
        返回:
        {
            "var_name": [
                {
                    "start": ...,
                    "end": ...,
                    "scope": ...,
                    "scope_start": ...,
                    "scope_end": ...
                },
                ...
            ]
        }
        """
        tree = self.parser.parse(source_code)
        identifiers = defaultdict(list)

        scope_stack = [{
            "id": 0,
            "start": 0,
            "end": len(source_code)
        }]
        scope_counter = 0

        def traverse(node):
            nonlocal scope_counter

            entered_scope = False
            if node.type == 'compound_statement':
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

                if name not in self.keywords and \
                        parent_type != 'function_declarator' and \
                        parent_type != 'field_identifier':
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
        """
        返回变量 var_name 所有出现位置对应的作用域区间集合
        """
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        """
        判断两个作用域区间是否有重叠
        """
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        """
        作用域感知重命名合法性判断：
        - new_name 不存在 => 可用
        - new_name 存在，但其所有出现的作用域与 old_name 不重叠 => 可用
        - 只要存在重叠作用域 => 不可用
        """
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

        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, new_name in replacements:
            code[start:end] = new_name.encode("utf-8")

        return code.decode("utf-8")