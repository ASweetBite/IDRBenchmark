import re
from collections import defaultdict
from tree_sitter import Language, Parser


class IdentifierAnalyzer:
    def __init__(self, lang="cpp"):
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
        tree = self.parser.parse(source_code)
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

        # 【修正】去除了 'qualified_identifier'，以支持重命名类外定义 (MyClass::Method)
        excluded_parents = {
            'preproc_def',
            'preproc_function_def',
            'type_identifier',
            'template_type',
            'namespace_identifier'  # 排除 namespace 名字本身
        }

        forbidden_node_types = {
            'destructor_name',  # ~MyClass
            'operator_name'  # operator+
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

            # 捕捉普通标识符 和 结构体/类的成员调用或方法声明 (field_identifier)
            if node.type in ["identifier", "field_identifier"]:
                parent_type = node.parent.type if node.parent else None
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                if node.parent and node.parent.type in forbidden_node_types:
                    pass
                elif name not in self.keywords and parent_type not in excluded_parents:

                    # 1. 精准判断是否为函数/方法声明
                    # (由于指针或引用，function_declarator 可能会被包几层，我们需要向上穿透)
                    is_func_decl = False
                    curr = node.parent
                    while curr and curr.type in ['qualified_identifier', 'pointer_declarator', 'reference_declarator',
                                                 'parenthesized_declarator']:
                        curr = curr.parent
                    if curr and curr.type == "function_declarator":
                        is_func_decl = True

                    is_constructor = False

                    # 2. 检查类外定义的构造函数 (例如 DataProcessor::DataProcessor)
                    if node.parent and node.parent.type == "qualified_identifier":
                        scope_node = node.parent.child_by_field_name('scope')
                        name_node = node.parent.child_by_field_name('name')
                        if scope_node and name_node and node == name_node:
                            scope_text = source_code[scope_node.start_byte:scope_node.end_byte].decode("utf-8")
                            scope_basename = scope_text.split("::")[-1]
                            if scope_basename == name:
                                is_constructor = True

                    # 3. 检查类内定义的构造函数
                    if is_func_decl and not is_constructor:
                        for scope in reversed(scope_stack):
                            if scope["type"] in ['class_specifier', 'struct_specifier']:
                                if name == scope["name"]:
                                    is_constructor = True
                                break

                    # 4. 判断调用类型
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

                    # 5. 【关键修复】如果是 field_identifier，它必须是方法声明或方法调用，才放行。
                    # 如果不是声明也不是方法调用，说明它是纯粹的类成员变量，为了安全保持不触碰。
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
        # ... 保持不变 ...
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        # ... 保持不变 ...
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        # ... 保持不变 ...
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
        # 保持不变，代码依然适用
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