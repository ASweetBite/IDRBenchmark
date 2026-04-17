import argparse
import math
import os
import random

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
from utils.llm_loder import LocalLLMClient
from utils.miner import NamingDataMiner
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo, CodeSmoother


def main(args, config):
    """Orchestrates the evaluation of model robustness against various renaming attacks."""

    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])
    if 'heavyweight_candidate' not in config:
        config['heavyweight_candidate'] = {}
    stats_path = config['heavyweight_candidate'].get('naming_stats_path', 'naming_stats.json')
    dataset_path = config['run_params']['dataset']

    if not os.path.exists(stats_path):
        print(f"\n[!] 启发式命名统计字典 '{stats_path}' 不存在。")
        print(f"[*] 正在启动离线数据挖掘程序 (基于数据集: {dataset_path})...")

        miner = NamingDataMiner(analyzer)
        miner.mine_parquet(dataset_path)
        miner.export_json(stats_path)
    else:
        print(f"\n[*] 发现已存在的命名统计字典: {stats_path}，跳过挖掘阶段。")

    # 显式注入路径，确保 Generator 能够读取到
    config['heavyweight_candidate']['naming_stats_path'] = stats_path
    # =========================================================================

    # 2. 字典准备就绪，正常加载极其耗时的深度学习引擎
    print("\n[*] Loading MLM Engine and Model Zoo...")
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    # ==================== [新增] 初始化 LLM 客户端 ====================
    llm_name = config.get('llm_generator_name', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)
    # ================================================================

    light_cand_config = {
        "candidate": config['lightweight_candidate']
    }
    lightweight_generator = LightweightCandidateGenerator(light_cand_config)

    generator = HeavyWeightCandidateGenerator(
        mlm_engine=mlm_engine,
        llm_client=llm_client,  # <--- 注入轻量级 LLM
        analyzer=analyzer,
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

    import time
    import gc
    import torch

    def get_subs_pool_fn(code_str: str, variables: list) -> dict:
        """Generates a pool of structural candidates using chunked batch processing to prevent OOM."""
        code_bytes = code_str.encode("utf-8")

        # 🌟 新增：安全批处理大小，针对 8GB 显存建议设为 3 或 4
        # 你可以根据实际显存占用率动态调高或调低
        MAX_BATCH_SIZE = 4

        # 1. 提取完整代码的标识符，寻找最外部的函数名
        identifiers = analyzer.extract_identifiers(code_bytes)
        outermost_func_name = None

        for name, occurrences in identifiers.items():
            for occ in occurrences:
                if occ["entity_type"] == "function" and occ["scope"] == 0:
                    outermost_func_name = name
                    break
            if outermost_func_name:
                break

        # 2. Phase 1: 收集所有变量的上下文切片
        batch_tasks = []
        print(f"[*] Slicing context for {len(variables)} variables...")

        for var in variables:
            try:
                if var == outermost_func_name:
                    target_code_str = code_str
                else:
                    target_code_str = analyzer.get_folded_code(code_bytes, var)

                batch_tasks.append({
                    "target_name": var,
                    "code_str": target_code_str
                })
            except Exception as e:
                print(f"[Warning] Failed to slice code for {var}: {e}")

        # 兜底初始化所有变量，确保不会报 KeyError
        final_pool = {var: [] for var in variables}

        if not batch_tasks:
            return final_pool

        # 3. Phase 2: 分块进行批量模型推理 (Chunked Batch Generation)
        total_tasks = len(batch_tasks)
        num_chunks = math.ceil(total_tasks / MAX_BATCH_SIZE)

        print(
            f"[*] Starting GPU generation for {total_tasks} tasks across {num_chunks} chunks (Max {MAX_BATCH_SIZE}/chunk)...")
        start_time = time.perf_counter()

        for i in range(0, total_tasks, MAX_BATCH_SIZE):
            chunk = batch_tasks[i:i + MAX_BATCH_SIZE]
            current_chunk_idx = (i // MAX_BATCH_SIZE) + 1
            print(f"  -> Processing chunk {current_chunk_idx}/{num_chunks} (size: {len(chunk)})...")

            try:
                # 将切分好的 chunk 送入生成器
                chunk_pool = generator.generate_candidates(
                    batch_tasks=chunk,
                    top_k_mlm=40,
                    top_n_keep=50
                )
                # 将当前块的结果合并到最终的池子中
                final_pool.update(chunk_pool)

            except Exception as e:
                print(f"[Error] Chunk {current_chunk_idx} failed catastrophically: {e}")
                # 如果当前 chunk 爆显存了，程序不会崩溃，只会损失这几个变量的候选词

            finally:
                if 'chunk_pool' in locals():
                    del chunk_pool

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # 4. 统计打印
        for var in variables:
            cands = final_pool.get(var, [])
            print(f"    - Generated {len(cands)} candidates for '{var}'")

        print(
            f"[*] All {num_chunks} chunks completed in {elapsed_time:.4f} seconds (Avg: {elapsed_time / max(1, total_tasks):.4f}s per variable).")

        return final_pool

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