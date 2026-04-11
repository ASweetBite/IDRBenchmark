import argparse
import concurrent.futures
import random
import yaml

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


def main(args, config):
    # 1. 初始化核心组件 (从 config 中读取配置)
    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    light_cand_config = {
        "candidate": config['lightweight_candidate']
    }
    lightweight_generator = LightweightCandidateGenerator(light_cand_config)

    # 2. 中间层：初始化候选词生成器 (依赖 MLM)
    generator = HeavyWeightCandidateGenerator(
        mlm_engine,
        analyzer,
        config=config['heavyweight_candidate']
    )
    # 3. 防御层：初始化平滑器 (依赖生成器)
    smoother_cfg = config['smoother']
    smoother = CodeSmoother(smoother_cfg, candidate_generator=lightweight_generator)

    # 4. 顶层：初始化 ModelZoo (依赖平滑器)，传入被攻击的靶标模型
    model_configs = config['model_zoo']
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        config=config,
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

        # 提取 AST 标识符只需做一次
        identifiers = analyzer.extract_identifiers(code_bytes)

        for var in variables:
            try:
                pool[var] = generator.generate_structural_candidates(
                    code_str, var, identifiers=identifiers
                )
            except Exception as e:
                print(f"[Warning] Failed to generate for {var}: {e}")
                pool[var] = []

        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    run_params = config['run_params']

    # 3. 初始化 Attacker (从 config 读取 iterations 和 algorithm)
    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode,
        config=config
    )

    # 4. 加载数据 (从 config 读取 dataset, samples, label_map)
    loader = DatasetLoader()
    print(f"[*] Loading dataset in {args.mode} mode...")
    dataset = loader.load_parquet_dataset(
        filepath=run_params['dataset'],
        mode=args.mode,
        max_samples=run_params['samples'],
        label_map_path=run_params.get('label_map')
    )

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
    asr_matrix_random = random_attacker.attack(dataset)

    print("\n" + "=" * 80)
    print("🏆  MODEL DEFENSE SCORES (模型防御性综合评分)")

    # 权重定义从配置文件读取
    W_VRTG = config['scoring_weights']['W_VRTG']
    W_NORM = config['scoring_weights']['W_NORM']
    W_RAND = config['scoring_weights']['W_RAND']

    print(f"权重分配：VRTG({int(W_VRTG * 100)}%) + Normalization({int(W_NORM * 100)}%) + Random({int(W_RAND * 100)}%)")
    print("=" * 80)

    model_names = model_zoo.model_names
    defense_scores = {}

    for m in model_names:
        # 1. 提取该模型作为目标时的 Self-ASR (对角线数据)
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

    # 仅保留 mode 和 config 参数
    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="选择运行模式: binary (二分类) 或 multi (细分)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="系统配置文件路径 (YAML格式)")

    args = parser.parse_args()

    # 读取 YAML 配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        parser.error(f"❌ 找不到配置文件: {args.config}。请确保文件存在！")

    # 校验 multi 模式下的 label_map 参数
    run_params = config.get('run_params', {})
    if args.mode == "multi" and run_params.get('label_map') is None:
        parser.error("❌ 当 --mode=multi 时，必须在配置文件中提供 run_params.label_map 参数")

    # 设置随机种子
    seed = config.get('global', {}).get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(args, config)