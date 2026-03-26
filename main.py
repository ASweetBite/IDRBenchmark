import random
import torch
import numpy as np
import argparse
import os

from attacks.NormalizationAttacker import NormalizationAttacker
from attacks.RandomAttacker import RandomAttacker
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.model_zoo import ModelZoo
from utils.dataset import DatasetLoader
from attacks.generators import CodeBasedCandidateGenerator
from attacks.IRTGAttacker import IRTGAttacker


def main(args):
    # 1. 初始化核心组件
    model_configs = {
        "CodeBERT": "./models/binary_diversevul_codebert" if args.mode == "binary" else "./models/multi_diversevul_codebert",
        # "GraphCodeBERT": "./models/binary_diversevul_graphcodebert",
        # "UniXcoder": "./models/binary_diversevul_unixcoder",
    }

    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"[!] 模型路径不存在: {path}")
            return

    model_zoo = ModelZoo(model_configs)
    analyzer = IdentifierAnalyzer(lang="cpp")
    transformer = CodeTransformer()
    generator = CodeBasedCandidateGenerator(model_zoo, analyzer)

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

    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode,
        iterations=args.iterations,
        run_mode=args.run_mode
    )

    loader = DatasetLoader()
    print(f"[*] Loading dataset in {args.mode} mode | Run mode: {args.run_mode}...")
    dataset = loader.load_parquet_dataset(filepath=args.dataset, mode=args.mode, max_samples=args.samples)

    asr_matrix_vrtg = evaluator.attack(dataset)

    if args.run_mode == "dataset":
        return
    normalizer = NormalizationAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        rename_fn=rename_fn,
        mode=args.mode
    )
    asr_matrix_norm = normalizer.attack(dataset)

    print("\n" + "=" * 80)
    print("🚀  RUNNING RANDOM RENAMING ATTACK (随机改名攻击)")
    print("=" * 80)

    random_attacker = RandomAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode
    )
    random_attacker.set_analyzer(analyzer)  # 绑定AST分析器
    asr_matrix_random= random_attacker.attack(dataset)

    print("\n" + "=" * 80)
    print("🏆  MODEL DEFENSE SCORES (模型防御性综合评分)")
    print("权重分配：VRTG(70%) + Normalization(20%) + Random(10%)")
    print("=" * 80)

    model_names = model_zoo.model_names
    defense_scores = {}

    # 权重定义
    W_VRTG = 0.7
    W_NORM = 0.2
    W_RAND = 0.1

    for m in model_names:
        # 1. 提取该模型作为目标时的 Self-ASR (对角线数据)
        # get(m, {}).get(m, 0.0) 确保即便某个模型没数据也不会报错
        vrtg_self_asr = asr_matrix_vrtg.get(m, {}).get(m, 0.0)
        norm_self_asr = asr_matrix_norm.get(m, {}).get(m, 0.0)
        rand_self_asr = asr_matrix_random.get(m, {}).get(m, 0.0)

        # 2. 计算防御率 (Defense Success Rate = 100 - ASR)
        vrtg_def = 100 - vrtg_self_asr
        norm_def = 100 - norm_self_asr
        rand_def = 100 - rand_self_asr

        # 3. 加权求和得到综合鲁棒性得分
        total_score = (vrtg_def * W_VRTG) + (norm_def * W_NORM) + (rand_def * W_RAND)

        defense_scores[m] = {
            "total": round(total_score, 2),
            "vrtg_asr": round(vrtg_self_asr, 2),
            "norm_asr": round(norm_self_asr, 2),
            "rand_asr": round(rand_self_asr, 2),
            "vrtg_def": round(vrtg_def, 2)
        }

    # 打印评分结果表
    header = f"{'Target Model':<20} | {'VRTG ASR':<12} | {'Norm ASR':<12} | {'Rand ASR':<12} | {'OVERALL SCORE'}"
    print(header)
    print("-" * len(header))

    # 按照综合得分从高到低排序 (最难被攻破的模型排在第一)
    ranked_models = sorted(defense_scores.items(), key=lambda x: x[1]['total'], reverse=True)

    for model_name, data in ranked_models:
        print(
            f"{model_name:<20} | {data['vrtg_asr']:>10.2f}% | {data['norm_asr']:>10.2f}% | {data['rand_asr']:>10.2f}% | {data['total']:>13} / 100")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对抗性样本生成攻击工具")

    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="选择运行模式: binary (二分类) 或 multi (细分)")
    parser.add_argument("--samples", type=int, default=100,
                        help="参与攻击的样本数量")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径 (parquet文件)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="遗传算法迭代次数")
    parser.add_argument("--run_mode", type=str, choices=["attack", "dataset"], default="attack",
                        help="运行模式：attack(成功即停止) 或 dataset(固定跑满世代，保存所有改名前后样本用于迁移攻击)")

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main(args)