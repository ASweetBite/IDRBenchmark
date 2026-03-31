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

        # 核心新增：记录当前代码片段内真正发生过声明/定义的变量或函数名
        defined_locally = set()

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

        # 核心新增：精准判断一个标识符节点是否是“定义/声明”本身，而不是单纯的使用
        def is_definition_node(node):
            parent = node.parent
            if not parent:
                return False

            # 1. 结构体、类、枚举、命名空间的名字
            if parent.type in ['class_specifier', 'struct_specifier', 'enum_specifier', 'namespace_definition']:
                if parent.child_by_field_name('name') == node:
                    return True

            # 2. 枚举值
            if parent.type == 'enumerator':
                if parent.children and parent.children[0] == node:
                    return True

            # 对于类的外部方法实现 (如 MyClass::myFunc)，只认定最右边的函数名为定义
            if parent.type == 'qualified_identifier':
                if parent.child_by_field_name('name') != node:
                    return False

            # 向上穿透声明修饰符 (如指针、引用、数组、函数等)
            curr = node
            p = parent
            while p and p.type in [
                'pointer_declarator', 'reference_declarator',
                'array_declarator', 'function_declarator',
                'parenthesized_declarator', 'qualified_identifier'
            ]:
                curr = p
                p = p.parent

            if not p:
                return False

            # 3. 初始化声明 (如 int a = 5;)
            if p.type == 'init_declarator':
                decl_node = p.child_by_field_name('declarator')
                if decl_node == curr:
                    return True
                # 防御性回退：声明符通常在等号前
                eq_node = None
                for child in p.children:
                    if child.type == '=':
                        eq_node = child
                        break
                if eq_node and curr.start_byte < eq_node.start_byte:
                    return True

            # 4. 普通声明、字段声明、参数声明
            if p.type in ['declaration', 'field_declaration', 'parameter_declaration',
                          'optional_parameter_declaration']:
                # 排除 extern 声明，因为它的实际生存周期在外部
                if p.type == 'declaration':
                    for child in p.children:
                        if child.type == 'storage_class_specifier' and source_code[
                            child.start_byte:child.end_byte].decode("utf-8") == 'extern':
                            return False

                type_node = p.child_by_field_name('type')
                value_node = p.child_by_field_name('default_value')
                # 只要它不是类型节点，也不是默认值节点，就是被声明的标识符
                if curr != type_node and curr != value_node:
                    return True

            # 5. 函数定义 (例如 void func() {})
            if p.type == 'function_definition':
                decl_node = p.child_by_field_name('declarator')
                if decl_node == curr:
                    return True

            # 6. for 范围循环 (例如 for (auto x : vec))
            if p.type == 'for_range_loop':
                decl_node = p.child_by_field_name('declarator')
                if decl_node == curr:
                    return True

            return False

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

                    # 核心新增：如果检测到该符号在此代码片段内被“定义”，则加入局部白名单
                    if is_definition_node(node):
                        defined_locally.add(name)

                    # 1. 精准判断是否为函数/方法声明
                    is_func_decl = False
                    curr = node.parent
                    while curr and curr.type in ['qualified_identifier', 'pointer_declarator', 'reference_declarator',
                                                 'parenthesized_declarator']:
                        curr = curr.parent
                    if curr and curr.type == "function_declarator":
                        is_func_decl = True

                    is_constructor = False

                    # 2. 检查类外定义的构造函数
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

                    # 5. 过滤掉纯类成员变量的安全保护
                    is_plain_field = (node.type == "field_identifier" and not is_method_call and not is_func_decl)

                    if not is_constructor and not is_plain_field:
                        current_scope = scope_stack[-1]
                        # 这里继续收集，待到最后一步进行过滤
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

        # 开始遍历 AST
        traverse(tree.root_node)

        # 核心新增：最后过滤阶段！
        # 仅保留存在于局部名单 (`defined_locally`) 中的标识符，抛弃 `printf`、外接库等纯引用标识符
        filtered_identifiers = {
            name: occurences
            for name, occurences in identifiers.items()
            if name in defined_locally
        }

        return filtered_identifiers

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