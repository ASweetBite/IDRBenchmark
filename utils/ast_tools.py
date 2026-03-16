from tree_sitter import Language, Parser
import tree_sitter_c
import re
from collections import defaultdict

class IdentifierAnalyzer:
    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.keywords = {
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen"
        }

    def extract_identifiers(self, source_code: bytes) -> dict:
        tree = self.parser.parse(source_code)
        identifiers = defaultdict(list)
        scope_stack = [0]
        scope_counter = 0

        def traverse(node):
            nonlocal scope_counter
            if node.type == 'compound_statement':
                scope_counter += 1
                scope_stack.append(scope_counter)

            if node.type == "identifier":
                parent_type = node.parent.type if node.parent else None
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                if name not in self.keywords and \
                        parent_type != 'function_declarator' and \
                        parent_type != 'field_identifier':
                    identifiers[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "scope": scope_stack[-1]
                    })
            for child in node.children:
                traverse(child)

            if node.type == 'compound_statement':
                scope_stack.pop()

        traverse(tree.root_node)
        return dict(identifiers)


def is_valid_identifier(name: str) -> bool:
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))

class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict) -> str:
        existing_names = set(identifiers.keys())
        for old_name, new_name in renaming_map.items():
            if not is_valid_identifier(new_name):
                raise ValueError(f"命名不合法: '{new_name}'")
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

