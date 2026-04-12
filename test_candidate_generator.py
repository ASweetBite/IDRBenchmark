import os
import sys
from collections import defaultdict

# 假设你的文件结构如下，确保路径能正确导入
from utils.ast_tools import IdentifierAnalyzer
from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator


def test_visualization():
    # 1. 初始化工具
    # 注意：这里的 model_zoo 可以传入 None 或 Mock，因为我们主要测试结构逻辑
    # 如果 generate_structural_candidates 内部需要 MLM 词汇，请确保已经有基础词库
    analyzer = IdentifierAnalyzer(lang="cpp")
    generator = HeavyWeightCandidateGenerator(mlm_engine=None, analyzer=analyzer)

    # 2. 准备测试用例：覆盖各种复杂的命名场景
    test_cases = [
        "ResReturnBuf",  # PascalCase [3, 6, 3]
        "user_login_count",  # snake_case [4, 5, 5]
        "_ptrIdx",  # camelCase [3, 3] 带下划线前缀
        "__init_data",  # snake_case [4, 4] 带双下划线
        "mixed_StyleName",  # 混合风格 (根据你的要求，应走 snake_case)
        "data2_ptr",  # 带数字的下划线
        "i",  # 单字母
    ]

    # 3. 模拟一个词库 (优化 A 的体现)
    # 在实际运行中，这些词应该来自 MLM 或本地代码
    mock_mlm_seeds = [
        "get", "set", "val", "ptr", "data", "info", "buffer", "process",
        "result", "index", "item", "handle", "status", "output", "input"
    ]
    # 手动填充一下 generator 的词库，防止生成的列表为空
    generator.local_pool = generator._build_length_pool(mock_mlm_seeds, [])

    # 4. 打印表头
    header = f"{'Original Name':<20} | {'Generated Candidate':<20} | {'Style':<12} | {'Lengths':<12} | {'Status'}"
    print("\n" + "=" * 85)
    print(header)
    print("-" * 85)

    dummy_code = "void dummy() { int " + ", ".join(test_cases) + "; }"

    for original in test_cases:
        # 获取原词格式信息
        orig_info = analyzer.analyze_format(original)

        # 生成候选词
        # 注意：确保你的 generator 类中有这个方法
        candidates = generator.generate_structural_candidates(dummy_code, original, top_n_keep=10)

        if not candidates:
            print(
                f"{original:<20} | {'[No Candidates]':<20} | {orig_info['style']:<12} | {str(orig_info['lengths']):<12} | ⚠️")
            continue

        for i, cand in enumerate(candidates):
            cand_info = analyzer.analyze_format(cand)

            # 验证逻辑
            style_ok = orig_info['style'] == cand_info['style']
            len_ok = orig_info['lengths'] == cand_info['lengths']
            prefix_ok = orig_info['prefix'] == cand_info['prefix']

            status = "✅" if (style_ok and len_ok and prefix_ok) else "❌"

            # 第一行显示原名，后续行留空以便阅读
            display_name = original if i == 0 else ""
            print(
                f"{display_name:<20} | {cand:<20} | {cand_info['style']:<12} | {str(cand_info['lengths']):<12} | {status}")

        print("-" * 85)


if __name__ == "__main__":
    # 模拟环境设置
    try:
        test_visualization()
    except Exception as e:
        print(f"运行失败: {e}")
        print("请检查 ast_tools.py 是否已添加 analyze_format 方法，以及 CodeBasedCandidateGenerator.py 是否已添加相应逻辑。")