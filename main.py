import random
import torch
import numpy as np
import argparse
import os

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.model_zoo import ModelZoo
from utils.dataset import DatasetLoader
from attacks.generators import CodeBasedCandidateGenerator
from attacks.attacker import VRTGAttacker


def main(args):
    # 1. 初始化核心组件
    model_configs = {
        "CodeBERT": "./models/binary_diversevul_codebert" if args.mode == "binary" else "./models/multi_diversevul_codebert",
        "GraphCodeBERT": "./models/binary_diversevul_graphcodebert",
        "UniXcoder": "./models/binary_diversevul_unixcoder",
    }

    # 确保模型路径存在
    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"[!] 模型路径不存在: {path}")
            return

    model_zoo = ModelZoo(model_configs)
    analyzer = IdentifierAnalyzer(lang="cpp")  # 假设初始化支持语言指定
    transformer = CodeTransformer()
    generator = CodeBasedCandidateGenerator(model_zoo, analyzer)

    # 2. 定义回调函数
    def get_all_identifiers_fn(code_str: str) -> list:
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def get_subs_pool_fn(code_str: str, variables: list) -> dict:
        pool = {}
        code_bytes = code_str.encode("utf-8")
        identifiers = analyzer.extract_identifiers(code_bytes)
        for var in variables:
            pool[var] = generator.generate_candidates(code_str, var, identifiers=identifiers)
        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    # 3. 初始化 Attacker (传入 mode 和 iterations)
    # 确保你的VRTGAttacker 类初始化支持这两个新参数
    evaluator = VRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode,  # 告知攻击者当前是二分类还是多分类，用于定义攻击成功逻辑
        iterations=args.iterations  # 告知遗传算法迭代次数
    )

    # 4. 加载数据 (使用修改后的 loader)
    loader = DatasetLoader()
    print(f"[*] Loading dataset in {args.mode} mode...")
    dataset = loader.load_parquet_dataset(filepath=args.dataset, mode=args.mode, max_samples=args.samples)

    # 5. 执行攻击
    evaluator.attack(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对抗性样本生成攻击工具")

    # 参数定义
    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="选择运行模式: binary (二分类) 或 multi (细分)")
    parser.add_argument("--samples", type=int, default=100,
                        help="参与攻击的样本数量")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径 (parquet文件)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="遗传算法迭代次数")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main(args)