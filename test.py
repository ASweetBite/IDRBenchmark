import pprint
from collections import defaultdict
from tree_sitter import Language, Parser
import tree_sitter_cpp


class IdentifierAnalyzer:
    def __init__(self):
        self.language = Language(tree_sitter_cpp.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.keywords = {
            "int", "char", "float", "double", "void", "if", "else", "for", "while", "return",
            "printf", "sizeof", "include", "main", "strcpy", "strlen", "malloc", "free",
            "memset", "memcpy", "fopen", "fclose", "bool", "true", "false", "NULL",
            "class", "public", "private", "protected", "template", "new", "delete",
            "catch", "try", "namespace", "using", "cout", "cin", "std", "endl", "auto",
            "const", "constexpr", "virtual", "override", "final", "this", "nullptr",
            "string", "vector", "map", "set", "list", "cerr", "static", "extern", "inline",
            "struct", "union", "enum", "typedef", "short", "long", "unsigned", "signed",
            "size_t", "EOF"
        }

    def is_definition_node(self, node, source_code):
        """判断标识符是否在当前代码片段中被定义/声明"""
        parent = node.parent
        if not parent: return False

        # 1. 类/结构体/命名空间定义名
        if parent.type in ['class_specifier', 'struct_specifier', 'namespace_definition']:
            if parent.child_by_field_name('name') == node: return True

        # 2. 作用域限定符处理 (如 WebPluginDelegateProxy::PluginDestroyed)
        # 只把最右边的名字(PluginDestroyed)视为定义主体
        if parent.type == 'qualified_identifier':
            if parent.child_by_field_name('name') != node: return False

        # 3. 向上穿透各类声明修饰符
        curr = node
        p = parent
        while p and p.type in ['pointer_declarator', 'reference_declarator', 'array_declarator',
                               'function_declarator', 'parenthesized_declarator', 'qualified_identifier']:
            curr = p
            p = p.parent
        if not p: return False

        # 4. 变量/函数声明
        if p.type in ['init_declarator', 'declaration', 'field_declaration', 'parameter_declaration',
                      'function_definition']:
            # 排除 extern (外部生存周期)
            if p.type == 'declaration':
                for child in p.children:
                    if child.type == 'storage_class_specifier' and source_code[child.start_byte:child.end_byte].decode(
                            "utf-8") == 'extern':
                        return False

            # 只要不是类型名节点，就是声明出的标识符
            type_node = p.child_by_field_name('type')
            if curr != type_node:
                return True

        return False

    def extract_identifiers(self, source_code: bytes) -> dict:
        tree = self.parser.parse(source_code)
        all_occurrences = defaultdict(list)
        defined_locally = set()

        def traverse(node):
            if node.type in ["identifier", "field_identifier"]:
                name = source_code[node.start_byte:node.end_byte].decode("utf-8")

                if name not in self.keywords:
                    # 检查是否为定义
                    if self.is_definition_node(node, source_code):
                        defined_locally.add(name)

                    # 记录所有位置（包括引用位置）
                    all_occurrences[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "type": node.type
                    })

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        # 核心逻辑：只输出那些在本片段中有“定义”的标识符
        return {name: locs for name, locs in all_occurrences.items() if name in defined_locally}


# --- 测试脚本运行 ---
if __name__ == "__main__":
    source_code = b"""
void WebPluginDelegateProxy::PluginDestroyed() {
  if (window_)
    WillDestroyWindow();

  if (channel_host_) {
    Send(new PluginMsg_DestroyInstance(instance_id_));

    channel_host_->RemoveRoute(instance_id_);

    channel_host_ = NULL;
  }

  if (window_script_object_) {
    window_script_object_->OnPluginDestroyed();
  }

  plugin_ = NULL;

  MessageLoop::current()->DeleteSoon(FROM_HERE, this);
}
"""

    analyzer = IdentifierAnalyzer()
    extracted = analyzer.extract_identifiers(source_code)

    print("=== 识别到的本地定义标识符 ===")
    if not extracted:
        print("未发现本地定义的标识符（所有标识符均为外部引用或类成员）。")
    else:
        for name, info in extracted.items():
            print(f"标识符: {name} | 出现次数: {len(info)}")
            for i in info:
                print(f"  - [{i['start']}:{i['end']}] 类型: {i['type']}")

    print("\n=== 逻辑说明 ===")
    print("1. PluginDestroyed 被提取：因为它在片段中被定义。")
    print("2. window_, channel_host_, Send, MessageLoop 等被忽略：因为它们只有引用，定义在外部或类定义中。")