from collections import defaultdict
from tree_sitter import Language, Parser
import tree_sitter_c


class IdentifierAnalyzer:
    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.keywords = {"int", "char", "float", "double", "void", "if", "else", "for", "while", "return", "printf",
                         "sizeof"}

    def extract_identifiers(self, source_code: bytes) -> dict:
        tree = self.parser.parse(source_code)
        # 存储格式: name -> list of {start, end, scope_id}
        identifiers = defaultdict(list)

        # 使用栈来模拟作用域进入与退出
        # 每进入一个 compound_statement ({) 就增加一层 scope
        scope_stack = [0]
        scope_counter = 0

        def traverse(node):
            nonlocal scope_counter

            if node.type == 'compound_statement':
                scope_counter += 1
                scope_stack.append(scope_counter)

            if node.type == "identifier":
                # 排除结构体成员访问: node->value, 这里的 value 不应被重命名
                # field_identifier 是 tree-sitter 中表示成员访问的类型
                parent_type = node.parent.type

                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                # 过滤规则：
                # 1. 不是关键字
                # 2. 不是函数定义
                # 3. 不是结构体成员访问 (field_identifier)
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