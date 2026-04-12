import re
from collections import defaultdict
from typing import Union

from tree_sitter import Language, Parser


class IdentifierAnalyzer:
    def __init__(self, lang="cpp"):
        """Initializes the AST parser and sets up reserved keywords for the specified language."""
        if lang == "c":
            from tree_sitter_c import language as ts_c
            self.language = Language(ts_c())
        elif lang == "cpp":
            from tree_sitter_cpp import language as ts_cpp
            self.language = Language(ts_cpp())
        else:
            raise ValueError("Unsupported language. Choose 'c' or 'cpp'.")

        self.keywords = {
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen", "malloc", "free",
            "memset", "memcpy", "fopen", "fclose", "bool", "true", "false", "NULL",
            "class", "public", "private", "protected", "template", "new", "delete",
            "catch", "try", "namespace", "using", "cout", "cin", "std", "endl", "auto",
            "const", "constexpr", "virtual", "override", "final", "this", "nullptr",
            "string", "vector", "map", "set", "list", "cerr",
            "static", "extern", "inline", "struct", "union", "enum", "typedef",
            "short", "long", "unsigned", "signed", "register", "volatile",
            "size_t", "ssize_t", "FILE", "DIR", "pid_t",
            "int8_t", "uint8_t", "int16_t", "uint16_t",
            "int32_t", "uint32_t", "int64_t", "uint64_t", "EOF"
        }

    def extract_identifiers(self, source_code: bytes) -> dict:
        """Traverses the AST to extract non-keyword identifiers and their scope information."""
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)
        identifiers = defaultdict(list)

        scope_stack = [{
            "id": 0,
            "start": 0,
            "end": len(source_code),
            "type": "global",
            "name": "global"
        }]
        scope_counter = 0

        scope_nodes = {
            'compound_statement',
            'class_specifier',
            'namespace_definition',
            'struct_specifier',
            'function_definition',
            'for_statement'
        }

        excluded_parents = {
            'preproc_def',
            'preproc_function_def',
            'type_identifier',
            'template_type',
            'namespace_identifier'
        }

        forbidden_node_types = {
            'destructor_name',
            'operator_name'
        }

        def traverse(node):
            nonlocal scope_counter

            entered_scope = False
            scope_name = ""

            if node.type in scope_nodes:
                scope_counter += 1
                if node.type in ['class_specifier', 'struct_specifier', 'namespace_definition']:
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        scope_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")

                scope_stack.append({
                    "id": scope_counter,
                    "start": node.start_byte,
                    "end": node.end_byte,
                    "type": node.type,
                    "name": scope_name
                })
                entered_scope = True

            if node.type in ["identifier", "field_identifier"]:
                parent_type = node.parent.type if node.parent else None
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                if node.parent and node.parent.type in forbidden_node_types:
                    pass
                elif name not in self.keywords and parent_type not in excluded_parents:

                    is_func_decl = False
                    curr = node.parent
                    while curr and curr.type in ['qualified_identifier', 'pointer_declarator', 'reference_declarator',
                                                 'parenthesized_declarator']:
                        curr = curr.parent
                    if curr and curr.type == "function_declarator":
                        is_func_decl = True

                    is_constructor = False

                    if node.parent and node.parent.type == "qualified_identifier":
                        scope_node = node.parent.child_by_field_name('scope')
                        name_node = node.parent.child_by_field_name('name')
                        if scope_node and name_node and node == name_node:
                            scope_text = source_code[scope_node.start_byte:scope_node.end_byte].decode("utf-8")
                            scope_basename = scope_text.split("::")[-1]
                            if scope_basename == name:
                                is_constructor = True

                    if is_func_decl and not is_constructor:
                        for scope in reversed(scope_stack):
                            if scope["type"] in ['class_specifier', 'struct_specifier']:
                                if name == scope["name"]:
                                    is_constructor = True
                                break

                    is_method_call = (
                            node.type == "field_identifier" and
                            node.parent and node.parent.type == "field_expression" and
                            node.parent.parent and node.parent.parent.type == "call_expression"
                    )

                    is_func_call = (
                            parent_type == "call_expression" and
                            node.parent.child_by_field_name('function') == node
                    )

                    entity_type = "function" if (is_func_decl or is_func_call or is_method_call) else "variable"

                    is_plain_field = (node.type == "field_identifier" and not is_method_call and not is_func_decl)

                    if not is_constructor and not is_plain_field:
                        current_scope = scope_stack[-1]
                        identifiers[name].append({
                            "start": node.start_byte,
                            "end": node.end_byte,
                            "scope": current_scope["id"],
                            "scope_start": current_scope["start"],
                            "scope_end": current_scope["end"],
                            "entity_type": entity_type
                        })

            for child in node.children:
                traverse(child)

            if entered_scope:
                scope_stack.pop()

        traverse(tree.root_node)
        return dict(identifiers)

    def get_identifier_scope_ranges(self, source_code: bytes, var_name: str):
        """Returns a list of start and end byte ranges defining the scope of a given identifier."""
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        """Checks if two distinct scope ranges overlap with one another."""
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        """Determines if an identifier can be safely renamed without causing scope collisions."""
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

    def analyze_format(self, name: str) -> dict:
        """Analyzes an identifier to determine its naming convention and structure."""
        prefix_match = re.match(r'^_+', name)
        prefix = prefix_match.group(0) if prefix_match else ""
        pure_name = name[len(prefix):]

        if not pure_name:
            return {"prefix": prefix, "lengths": [], "style": "special", "count": 0}

        if '_' in pure_name:
            style = "snake_case"
            words = pure_name.split('_')
        else:
            words = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)', pure_name)
            if pure_name[0].isupper():
                style = "PascalCase"
            else:
                style = "camelCase"

        return {
            "prefix": prefix,
            "lengths": [len(w) for w in words],
            "style": style,
            "count": len(words)
        }

    def canonicalize(self, source_code: Union[str, bytes]) -> str:
        """
        [防御专属] 代码规范化兜底方法：
        将所有提取出的自定义变量和函数替换为泛化 Token (VARx, FUNCx)。
        用于在高方差告警时，物理隔离标识符级别的对抗攻击。
        """
        if isinstance(source_code, str):
            code_bytes = source_code.encode("utf-8")
        else:
            code_bytes = source_code

        identifiers = self.extract_identifiers(code_bytes)
        if not identifiers:
            return code_bytes.decode("utf-8")

        var_counter = 1
        func_counter = 1
        renaming_map = {}

        # 排序以保证相同的代码片段每次规范化的结果一致（Deterministic）
        for name in sorted(identifiers.keys()):
            # 取该标识符第一次出现的信息，判断它是函数还是变量
            entity_info = identifiers[name][0]
            entity_type = entity_info.get("entity_type", "variable")

            if entity_type == "function":
                # 防止源码中本身就有 FUNC1 导致命名冲突
                while f"FUNC{func_counter}" in identifiers:
                    func_counter += 1
                renaming_map[name] = f"FUNC{func_counter}"
                func_counter += 1
            else:
                # 防止源码中本身就有 VAR1 导致命名冲突
                while f"VAR{var_counter}" in identifiers:
                    var_counter += 1
                renaming_map[name] = f"VAR{var_counter}"
                var_counter += 1

        try:
            # 使用现有的 CodeTransformer 进行安全替换
            # 传入 analyzer=None 绕过严格的范围校验，强制执行全局替换
            canonical_code = CodeTransformer.validate_and_apply(
                code_bytes, identifiers, renaming_map, analyzer=None
            )
            return canonical_code
        except Exception as e:
            # 如果在极端情况下发生转换异常，作为最后防线，直接返回原码
            return code_bytes.decode("utf-8")


def is_valid_identifier(name: str) -> bool:
    """Validates if a string follows standard C/C++ identifier naming rules."""
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict, analyzer=None) -> str:
        """Validates renaming rules and securely applies the substitutions to the code bytearray."""
        for old_name, new_name in renaming_map.items():
            if not is_valid_identifier(new_name):
                raise ValueError(f"Invalid naming: '{new_name}'")

            if analyzer is not None:
                if not analyzer.can_rename_to(source_code, old_name, new_name):
                    raise ValueError(f"Scope conflict: '{old_name}' -> '{new_name}' is unavailable.")
            else:
                existing_names = set(identifiers.keys())
                if new_name in existing_names and new_name != old_name:
                    raise ValueError(f"Renaming conflict: '{old_name}' -> '{new_name}' already exists.")

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