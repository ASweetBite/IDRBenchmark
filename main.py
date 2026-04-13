import argparse
import random
import time

import numpy as np
import torch
import yaml

from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from attacks.IRTGAttacker import IRTGAttacker
from attacks.LightweightCandidateGenerator import LightweightCandidateGenerator
from attacks.NormalizationAttacker import NormalizationAttacker
from attacks.RandomAttacker import RandomAttacker
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.dataset import DatasetLoader
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo, CodeSmoother


def main(args, config):
    """Orchestrates the evaluation of model robustness against various renaming attacks."""
    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    light_cand_config = {
        "candidate": config['lightweight_candidate']
    }
    lightweight_generator = LightweightCandidateGenerator(light_cand_config)

    generator = HeavyWeightCandidateGenerator(
        mlm_engine,
        analyzer,
        config=config['heavyweight_candidate']
    )

    smoother_cfg = config['smoother']
    smoother = CodeSmoother(smoother_cfg, candidate_generator=lightweight_generator)

    model_configs = config['model_zoo']
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        config=config,
        smoother=smoother
    )
    transformer = CodeTransformer()

    def get_all_identifiers_fn(code_str: str) -> list:
        """Extracts all identifiers from the code except 'main'."""
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def get_subs_pool_fn(code_str: str, variables: list) -> dict:
        """Generates a pool of structural candidates for the specified variables."""
        pool = {}
        code_bytes = code_str.encode("utf-8")
        identifiers = analyzer.extract_identifiers(code_bytes)

        for var in variables:
            try:
                # 记录开始时间
                start_time = time.perf_counter()

                pool[var] = generator.generate_candidates(
                    code_str, var, identifiers=identifiers
                )

                # 记录结束时间并计算差值
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # 打印耗时和生成的候选词数量
                print(
                    f"[Info] Successfully generated {len(pool[var])} candidates for '{var}' in {elapsed_time:.4f} seconds.")

            except Exception as e:
                print(f"[Warning] Failed to generate for {var}: {e}")
                pool[var] = []

        return pool

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        """Applies variable renaming using the code transformer."""
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    run_params = config['run_params']

    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode,
        config=config
    )

    loader = DatasetLoader()
    print(f"[*] Loading dataset in {args.mode} mode...")
    dataset = loader.load_parquet_dataset(
        filepath=run_params['dataset'],
        mode=args.mode,
        max_samples=run_params['samples'],
        label_map_path=run_params.get('label_map')
    )

    asr_matrix_vrtg = evaluator.attack(dataset)

    normalier = NormalizationAttacker(
        model_zoo=model_zoo,
        rename_fn=rename_fn,
        mode=args.mode
    )
    asr_matrix_norm = normalier.attack(dataset)

    print("\n" + "=" * 80)
    print("🚀  RUNNING RANDOM RENAMING ATTACK")
    print("=" * 80)

    random_attacker = RandomAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        get_subs_pool_fn=get_subs_pool_fn,
        rename_fn=rename_fn,
        mode=args.mode
    )
    random_attacker.set_analyzer(analyzer)
    asr_matrix_random = random_attacker.attack(dataset)

    print("\n" + "=" * 80)
    print("🏆  MODEL DEFENSE SCORES")

    W_VRTG = config['scoring_weights']['W_VRTG']
    W_NORM = config['scoring_weights']['W_NORM']
    W_RAND = config['scoring_weights']['W_RAND']

    print(
        f"Weight Distribution: VRTG({int(W_VRTG * 100)}%) + Normalization({int(W_NORM * 100)}%) + Random({int(W_RAND * 100)}%)")
    print("=" * 80)

    model_names = model_zoo.model_names
    defense_scores = {}

    for m in model_names:
        vrtg_self_asr = asr_matrix_vrtg.get(m, {}).get(m, 0.0)
        norm_self_asr = asr_matrix_norm.get(m, {}).get(m, 0.0)
        rand_self_asr = asr_matrix_random.get(m, {}).get(m, 0.0)

        vrtg_def = 100 - vrtg_self_asr
        norm_def = 100 - norm_self_asr
        rand_def = 100 - rand_self_asr

        total_score = (vrtg_def * W_VRTG) + (norm_def * W_NORM) + (rand_def * W_RAND)

        defense_scores[m] = {
            "total": round(total_score, 2),
            "vrtg_asr": round(vrtg_self_asr, 2),
            "norm_asr": round(norm_self_asr, 2),
            "rand_asr": round(rand_self_asr, 2),
            "vrtg_def": round(vrtg_def, 2)
        }

    header = f"{'Target Model':<20} | {'VRTG ASR':<12} | {'Norm ASR':<12} | {'Rand ASR':<12} | {'OVERALL SCORE'}"
    print(header)
    print("-" * len(header))

    ranked_models = sorted(defense_scores.items(), key=lambda x: x[1]['total'], reverse=True)

    for model_name, data in ranked_models:
        print(
            f"{model_name:<20} | {data['vrtg_asr']:>10.2f}% | {data['norm_asr']:>10.2f}% | {data['rand_asr']:>10.2f}% | {data['total']:>13} / 100")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial sample generation attack tool")

    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="Select run mode: binary or multi")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="System configuration file path (YAML format)")

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        parser.error(f"❌ Configuration file not found: {args.config}. Please ensure the file exists!")

    run_params = config.get('run_params', {})
    if args.mode == "multi" and run_params.get('label_map') is None:
        parser.error(
            "❌ When --mode=multi, the run_params.label_map parameter must be provided in the configuration file")

    seed = config.get('global', {}).get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(args, config)