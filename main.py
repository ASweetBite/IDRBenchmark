import random
import torch
import numpy as np

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer, is_valid_identifier
from utils.model_zoo import ModelZoo
from utils.dataset import DatasetLoader
from attacks.generators import CodeBasedCandidateGenerator
from attacks.attacker import VRTGAttacker, TransferabilityEvaluator


def main():
    # 1. 初始化核心组件
    model_configs = {
        "CodeBERT": "./models/codebert_finetuned",
        "GraphCodeBERT": "./models/graphcodebert_finetuned",
        "UniXcoder": "./models/unixcoder_finetuned"
    }
    model_zoo = ModelZoo(model_configs)
    analyzer = IdentifierAnalyzer()
    transformer = CodeTransformer()
    generator = CodeBasedCandidateGenerator(model_zoo, analyzer)

    # 2. 定义回调函数
    def get_all_vars_fn(code_str: str) -> list:
        return list(analyzer.extract_identifiers(code_str.encode("utf-8")).keys())

    def get_subs_pool_fn(code_str: str, variables: list) -> dict:
        pool = {}
        for var in variables:
            pool[var] = generator.generate_candidates(code_str, var)
        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map)

        # --- 3. 执行攻击与评估 (关键修改) ---

        # 删掉手动创建 attacker 的代码， evaluator 内部会循环创建
        # attacker = VRTGAttacker(model_zoo, TARGET_MODEL, get_all_vars_fn, get_subs_pool_fn, rename_fn, top_k=3)

        # 使用新的初始化方式

    evaluator = TransferabilityEvaluator(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_vars_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn
    )

    dataset = DatasetLoader.load_json("./data/megavul_test.json", max_samples=10)
    evaluator.evaluate(dataset)


if __name__ == "__main__":
    random.seed(42);
    np.random.seed(42);
    torch.manual_seed(42)
    main()