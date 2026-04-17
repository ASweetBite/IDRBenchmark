import re
from collections import defaultdict
from typing import Union

from tree_sitter import Language
from tree_sitter import Parser


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

        # [性能优化] 预先将 keywords 转换为 bytes，避免在遍历时对每个标识符进行耗时的 decode 操作
        self.keywords_bytes = {k.encode("utf-8") for k in self.keywords}

        # [性能优化] 预编译 Tree-sitter S-expression Query
        # 将原本需要在 Python 中递归的成千上万个节点，下放给底层的 C 引擎直接过滤出我们关心的节点
        query_str = """
        [
            (compound_statement)
            (class_specifier)
            (namespace_definition)
            (struct_specifier)
            (function_definition)
            (for_statement)
        ] @scope

        [
            (identifier)
            (field_identifier)
        ] @ident
        """
        # 兼容新老版本的 tree-sitter python binding
        try:
            from tree_sitter import Query
            self.ast_query = Query(self.language, query_str)
        except ImportError:
            self.ast_query = self.language.query(query_str)

    def extract_identifiers(self, source_code: bytes) -> dict:
        """Traverses the AST to extract non-keyword identifiers and their scope information."""
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)
        identifiers = defaultdict(list)

        defined_names = set()

        scope_stack = [{
            "id": 0,
            "start": 0,
            "end": len(source_code),
            "type": "global",
            "name": "global"
        }]
        scope_counter = 0

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

        # 1. 核心提速：使用 C 引擎一次性找出所有相关的 scope 和 标识符节点
        if hasattr(self.ast_query, "captures"):
            captures = self.ast_query.captures(tree.root_node)
        else:
            from tree_sitter import QueryCursor
            cursor = QueryCursor(self.ast_query)
            captures = cursor.captures(tree.root_node)

        if isinstance(captures, dict):
            flat_captures = [(node, tag) for tag, nodes in captures.items() for node in
                             (nodes if isinstance(nodes, list) else [nodes])]
        else:
            flat_captures = list(captures)

        flat_captures.sort(key=lambda x: (x[0].start_byte, -x[0].end_byte))

        for node, tag in flat_captures:

            while len(scope_stack) > 1 and scope_stack[-1]["end"] <= node.start_byte:
                scope_stack.pop()

            if tag == "scope":
                scope_counter += 1
                scope_name = ""
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

            elif tag == "ident":
                # 字节级短路验证：极大减少不必要的 UTF-8 解码开销
                name_bytes = source_code[node.start_byte:node.end_byte]
                if name_bytes in self.keywords_bytes:
                    continue

                parent_type = node.parent.type if node.parent else None

                if node.parent and node.parent.type in forbidden_node_types:
                    continue
                if parent_type in excluded_parents:
                    continue

                # 确认是目标变量后，再进行解码
                name = name_bytes.decode("utf-8")

                is_def = False
                curr_node = node
                while curr_node:
                    parent = curr_node.parent
                    if not parent:
                        break

                    # 1. 明确的“使用”场景：作为初始化右值、或数组长度参数
                    if parent.type == 'init_declarator' and parent.child_by_field_name('value') == curr_node:
                        break
                    if parent.type == 'array_declarator' and parent.child_by_field_name('size') == curr_node:
                        break

                    # 2. 如果遇到任何表达式或执行语句，说明它是被调用/使用的变量或函数（例如宏替换、外部全局变量运算）
                    if parent.type in {
                        'binary_expression', 'unary_expression', 'update_expression',
                        'assignment_expression', 'call_expression', 'subscript_expression',
                        'conditional_expression', 'initializer_list', 'initializer',
                        'argument_list', 'return_statement', 'expression_statement',
                        'if_statement', 'while_statement', 'do_statement', 'switch_statement',
                        'case_statement', 'parenthesized_expression', 'cast_expression',
                        'comma_expression', 'sizeof_expression', 'type_descriptor'
                    }:
                        break

                    # 3. 成功到达声明/定义层级，确认此标识符在当前片段中有被定义
                    if parent.type in {
                        'declaration', 'parameter_declaration', 'function_definition',
                        'field_declaration', 'catch_declaration', 'optional_parameter_declaration',
                        'struct_specifier', 'class_specifier', 'enum_specifier'
                    }:
                        is_def = True
                        break

                    curr_node = parent

                if is_def:
                    defined_names.add(name)
                # =====================================================================

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

                # =====================================================================
                # [新增逻辑] 提取函数返回值类型 / 变量声明类型
                # =====================================================================
                extracted_type = None
                # 只有在它是定义/声明节点时，才能在 AST 中稳定找到类型
                if is_def or is_func_decl:
                    decl_node = node.parent
                    while decl_node and decl_node.type not in {
                        'function_definition', 'declaration', 'field_declaration',
                        'parameter_declaration', 'optional_parameter_declaration'
                    }:
                        decl_node = decl_node.parent

                    if decl_node:
                        type_node = decl_node.child_by_field_name('type')
                        if type_node:
                            extracted_type = source_code[type_node.start_byte:type_node.end_byte].decode("utf-8")
                # =====================================================================

                if not is_constructor and not is_plain_field:
                    current_scope = scope_stack[-1]
                    identifiers[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "scope": current_scope["id"],
                        "scope_start": current_scope["start"],
                        "scope_end": current_scope["end"],
                        "entity_type": entity_type,
                        "return_type": extracted_type  # 统一存入 return_type 字段
                    })

        filtered_identifiers = {
            name: usages
            for name, usages in identifiers.items()
            if name in defined_names
        }

        return filtered_identifiers

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

        for name in sorted(identifiers.keys()):
            entity_info = identifiers[name][0]
            entity_type = entity_info.get("entity_type", "variable")

            if entity_type == "function":
                while f"FUNC{func_counter}" in identifiers:
                    func_counter += 1
                renaming_map[name] = f"FUNC{func_counter}"
                func_counter += 1
            else:
                while f"VAR{var_counter}" in identifiers:
                    var_counter += 1
                renaming_map[name] = f"VAR{var_counter}"
                var_counter += 1

        try:
            canonical_code = CodeTransformer.validate_and_apply(
                code_bytes, identifiers, renaming_map, analyzer=None
            )
            return canonical_code
        except Exception as e:
            return code_bytes.decode("utf-8")

    def _get_enclosing_statement(self, node):
        """
        向上遍历 AST，寻找包含当前节点的完整语句（Statement）。
        当父节点是作用域块或控制流语句头部时停止，当前节点即为最小完整语句级切片。
        """
        curr = node
        stop_parent_types = {
            'compound_statement',
            'translation_unit',
            'for_statement',
            'while_statement',
            'if_statement',
            'switch_statement',
            'function_definition'
        }

        while curr.parent:
            if curr.parent.type in stop_parent_types:
                break
            curr = curr.parent
        return curr

    def get_folded_code(self, source_code: bytes, target_var: str) -> str:
        """提取数据流切片，强制闭合所有分支，并完美保留无括号单行语句和所有函数调用"""
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)

        target_nodes = []
        call_nodes = []

        def find_nodes(node):
            if node.type in ["identifier", "field_identifier"]:
                name = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                if name == target_var:
                    target_nodes.append(node)
            elif node.type == "call_expression":
                call_nodes.append(node)
            for child in node.children:
                find_nodes(child)

        find_nodes(tree.root_node)
        if not target_nodes:
            return source_code.decode("utf-8", errors="replace")

        ranges_to_keep = []
        insertions = []

        def handle_control_body(body_node):
            if not body_node:
                return
            if body_node.type == 'compound_statement':
                # 已有括号：仅保留 '{' 和 '}'
                ranges_to_keep.append((body_node.start_byte, body_node.start_byte + 1))
                ranges_to_keep.append((body_node.end_byte - 1, body_node.end_byte))
            elif body_node.type != 'if_statement':
                # 无括号的单行语句：强制注入 '{' 和 '}'，并将单行语句本身保留
                insertions.append((body_node.start_byte, b" { "))
                insertions.append((body_node.end_byte, b" } "))
                ranges_to_keep.append((body_node.start_byte, body_node.end_byte))

        # 使用基于祖先路径的追溯法，确保所有嵌套的控制流都能精准还原头部和闭合
        for node in target_nodes + call_nodes:
            curr = node
            while curr:
                if curr.type in ['expression_statement', 'declaration', 'return_statement',
                                 'break_statement', 'continue_statement', 'goto_statement']:
                    ranges_to_keep.append((curr.start_byte, curr.end_byte))

                elif curr.type == 'if_statement':
                    # 提取 if 头部
                    cond_node = curr.child_by_field_name('condition')
                    if cond_node:
                        ranges_to_keep.append((curr.start_byte, cond_node.end_byte))
                    else:
                        ranges_to_keep.append((curr.start_byte, curr.start_byte + 2))

                    handle_control_body(curr.child_by_field_name('consequence'))

                    alt = curr.child_by_field_name('alternative')
                    if alt:
                        # 只有当前追踪的变量在 else 内部时，才保留 else 关键字
                        if node.start_byte >= alt.start_byte and node.end_byte <= alt.end_byte:
                            for child in curr.children:
                                if child.type == 'else':
                                    ranges_to_keep.append((child.start_byte, child.end_byte))
                                    break
                            handle_control_body(alt)

                elif curr.type in ['while_statement', 'for_statement', 'switch_statement']:
                    # 提取控制流头部（精准找寻右括号）
                    cond_node = curr.child_by_field_name('condition')
                    if cond_node:
                        for child in curr.children:
                            if child.type == ')' and child.start_byte >= cond_node.end_byte:
                                ranges_to_keep.append((curr.start_byte, child.end_byte))
                                break
                        else:
                            ranges_to_keep.append((curr.start_byte, cond_node.end_byte))
                    else:
                        body = curr.child_by_field_name('body')
                        if body:
                            for child in reversed(curr.children):
                                if child.type == ')' and child.end_byte <= body.start_byte:
                                    ranges_to_keep.append((curr.start_byte, child.end_byte))
                                    break
                            else:
                                ranges_to_keep.append((curr.start_byte, body.start_byte))
                        else:
                            ranges_to_keep.append((curr.start_byte, curr.end_byte))

                    handle_control_body(curr.child_by_field_name('body'))

                elif curr.type == 'function_definition':
                    body = curr.child_by_field_name('body')
                    if body and body.type == 'compound_statement':
                        ranges_to_keep.append((curr.start_byte, body.start_byte + 1))
                        ranges_to_keep.append((body.end_byte - 1, body.end_byte))
                    else:
                        ranges_to_keep.append((curr.start_byte, curr.end_byte))

                curr = curr.parent

        # 区间去重与合并
        ranges_to_keep.sort(key=lambda x: x[0])
        merged_ranges = []
        for current in ranges_to_keep:
            if not merged_ranges:
                merged_ranges.append(current)
            else:
                last = merged_ranges[-1]
                if current[0] <= last[1] + 15:  # 容忍 15 字节以内的空白符吞并
                    merged_ranges[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged_ranges.append(current)

        # 注入点整理
        unique_insertions = list(set(insertions))
        ins_dict = {}
        for off, txt in unique_insertions:
            if off not in ins_dict:
                ins_dict[off] = []
            ins_dict[off].append(txt)

        output = bytearray()
        last_end = 0

        # 重构文本，精准应用保留区间与大括号注入
        for start, end in merged_ranges:
            gap_insertions = [(off, txts) for off, txts in ins_dict.items() if last_end <= off <= start]
            gap_insertions.sort(key=lambda x: x[0])

            gap_cursor = last_end
            for off, txts in gap_insertions:
                if off - gap_cursor > 15:
                    if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                        output.extend(b"\n    /* ... */\n")
                else:
                    output.extend(source_code[gap_cursor:off])
                for txt in txts:
                    output.extend(txt)
                gap_cursor = off

            if start - gap_cursor > 15:
                if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                    output.extend(b"\n    /* ... */\n")
            else:
                output.extend(source_code[gap_cursor:start])

            keep_cursor = start
            inside_insertions = [(off, txts) for off, txts in ins_dict.items() if start < off < end]
            inside_insertions.sort(key=lambda x: x[0])

            for off, txts in inside_insertions:
                output.extend(source_code[keep_cursor:off])
                for txt in txts:
                    output.extend(txt)
                keep_cursor = off

            output.extend(source_code[keep_cursor:end])
            last_end = end

        final_insertions = [(off, txts) for off, txts in ins_dict.items() if off >= last_end]
        final_insertions.sort(key=lambda x: x[0])

        gap_cursor = last_end
        for off, txts in final_insertions:
            if off - gap_cursor > 15:
                if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                    output.extend(b"\n    /* ... */\n")
            else:
                output.extend(source_code[gap_cursor:off])
            for txt in txts:
                output.extend(txt)
            gap_cursor = off

        if len(source_code) - gap_cursor > 15:
            if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                output.extend(b"\n    /* ... */\n")
        else:
            output.extend(source_code[gap_cursor:len(source_code)])

        return output.decode("utf-8", errors="replace")

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