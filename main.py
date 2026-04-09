import argparse
import random

import numpy as np
import torch

from attacks.IRTGAttacker import IRTGAttacker
from attacks.LightweightCandidateGenerator import LightweightCandidateGenerator
from attacks.NormalizationAttacker import NormalizationAttacker
from attacks.RandomAttacker import RandomAttacker
from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.dataset import DatasetLoader
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo, CodeSmoother


def main(args):
    # 1. 初始化核心组件
    analyzer = IdentifierAnalyzer(lang="cpp")
    mlm_engine = MLMEngine("microsoft/codebert-base-mlm")
    light_cand_config = {
        "candidate": {
            "top_m": 15,
            "fasttext_model_path": "./models/fasttext_cpp.bin",
            "faiss_index_path": "./models/faiss_cpp.index",
            "faiss_vocab_path": "./models/vocab_cpp.json"
        }
    }
    lightweight_generator = LightweightCandidateGenerator(light_cand_config)
    # 2. 中间层：初始化候选词生成器 (依赖 MLM)
    generator = HeavyWeightCandidateGenerator(mlm_engine, analyzer)

    # 3. 防御层：初始化平滑器 (依赖生成器)
    smoother_cfg = {
        "num_samples": 50,
        "variance_threshold": 0.05,
        "replace_prob": 0.5,
        "batch_size": 32
    }
    smoother = CodeSmoother(smoother_cfg, candidate_generator=lightweight_generator)
    # 4. 顶层：初始化 ModelZoo (依赖平滑器)，传入被攻击的靶标模型
    model_configs = {
        "CodeBERT": "./models/finetuned_model_codebert"
    }
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        smoother=smoother  # 完美注入，没有循环依赖
    )
    transformer = CodeTransformer()

    # 2. 定义回调函数
    def get_all_identifiers_fn(code_str: str) -> list:
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def get_subs_pool_fn(code_str: str, variables: list) -> dict:
        pool = {}
        code_bytes = code_str.encode("utf-8")
        identifiers = analyzer.extract_identifiers(code_bytes)
        for var in variables:
            pool[var] = generator.generate_structural_candidates(code_str, var, identifiers=identifiers)
        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    # 3. 初始化 Attacker (传入 mode 和 iterations)
    # 确保你的VRTGAttacker 类初始化支持这两个新参数
    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode,  # 告知攻击者当前是二分类还是多分类，用于定义攻击成功逻辑
        iterations=args.iterations,  # 告知遗传算法迭代次数
        optimizer_type=args.algorithm
    )

    # 4. 加载数据 (使用修改后的 loader)
    loader = DatasetLoader()
    print(f"[*] Loading dataset in {args.mode} mode...")
    dataset = loader.load_parquet_dataset(filepath=args.dataset, mode=args.mode, max_samples=args.samples)

    # 5. 执行攻击
    asr_matrix_vrtg = evaluator.attack(dataset)
    normalier = NormalizationAttacker(
        model_zoo=model_zoo,
        rename_fn=rename_fn,
        mode=args.mode
    )
    asr_matrix_norm = normalier.attack(dataset)

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

    # 参数定义
    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="选择运行模式: binary (二分类) 或 multi (细分)")
    parser.add_argument("--samples", type=int, default=100,
                        help="参与攻击的样本数量")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径 (parquet文件)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="遗传算法迭代次数")
    parser.add_argument("--algorithm", type=str, choices=["ga","greedy"], default="ga",
                        help="使用什么优化算法")
    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main(args)