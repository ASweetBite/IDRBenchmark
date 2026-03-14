import re

def is_valid_identifier(name: str) -> bool:
    """
    检查标识符是否合法：
    1. 不能以数字开头
    2. 只能包含字母、数字、下划线
    3. 长度不能为 0
    4. 不能是 C 语言关键字（建议提前在 analyzer 里维护一份）
    """
    # 匹配标识符正则：字母或下划线开头，后接字母数字下划线
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))

class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict) -> str:
        """
        renaming_map 格式: {'old_name': 'new_name'}
        """
        existing_names = set(identifiers.keys())

        for old_name, new_name in renaming_map.items():
            # 1. 语法规范性校验 (防止 '='、'*' 等非法字符)
            if not is_valid_identifier(new_name):
                raise ValueError(f"命名不合法: '{new_name}' 不是一个合法的标识符 (包含非法字符或格式错误)。")

            # 2. 预留关键词校验 (可从 Analyzer 获取)
            # if new_name in analyzer.keywords: ...

            # 3. 冲突检查
            if new_name in existing_names and new_name != old_name:
                raise ValueError(f"重命名冲突: 无法将 '{old_name}' 改为 '{new_name}'，名称已存在。")

        # 4. 执行替换逻辑 (保持原有代码不变)
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