import os
import re
import yaml
import torch
import torch.nn.functional as F
import gc
import time
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 保持原有导入不变
from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer
from utils.llm_loader import LocalLLMClient
from utils.mlm_engine import MLMEngine
from utils.miner import NamingDataMiner


def evaluate_and_print_candidates(generator, full_code_bytes: bytes, target_name: str, candidates: list):
    """
    重构后的评估函数：直接接收 candidates 列表，计算相似度并打印。
    """
    print(f"\n[*] 正在评估变量 '{target_name}' 的候选词 (共 {len(candidates)} 个)...")

    if not candidates:
        print(f"[-] 未能为 '{target_name}' 生成任何有效且合法的候选词。")
        return

    # 1. 提取数据计算排版分数
    identifiers = generator.analyzer.extract_identifiers(full_code_bytes)
    if target_name not in identifiers:
        print(f"[-] 错误: '{target_name}' 在 AST 解析中丢失。")
        return

    best_occ_idx = generator._find_best_context_occurrence(full_code_bytes, identifiers[target_name])
    target_info = identifiers[target_name][best_occ_idx]

    raw_entity_type = target_info.get('entity_type', 'variable')
    entity_type = 'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE'
    if entity_type == 'VARIABLE' and target_name.startswith(('is_', 'has_', 'can_', 'should_')):
        entity_type = 'BOOLEAN_VAR'

    local_prefix, local_suffix = generator._extract_local_context_ast(
        full_code_bytes, target_info['start'], target_info['end']
    )

    orig_emb = generator._get_variable_token_embeddings(
        [local_prefix], [target_name], [local_suffix]
    ).to(generator.mlm_engine.device)

    prefixes = [local_prefix] * len(candidates)
    suffixes = [local_suffix] * len(candidates)
    cand_embs = generator._get_variable_token_embeddings(
        prefixes, candidates, suffixes
    ).to(generator.mlm_engine.device)

    sims = F.cosine_similarity(orig_emb, cand_embs)

    results = []
    has_scorer = hasattr(generator, 'scorer')

    for cand, sim in zip(candidates, sims):
        base_sim = sim.item()
        heuristic_bonus = 0.0

        if has_scorer:
            cand_parts, _ = generator._split_identifier(cand)
            heuristic_bonus = generator.scorer.calculate_heuristic_score(cand_parts, entity_type)

        final_score = base_sim + heuristic_bonus
        results.append((cand, base_sim, heuristic_bonus, final_score))

    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n[+] '{target_name}' 的最终输出候选词及得分排名 (共 {len(results)} 个):")
    print(f"    上下文提取策略: {entity_type} Context")
    print("-" * 85)
    print(f"{'候选词 (Candidate)':<30} | {'Token 相似度':<15} | {'NLP 修正分':<12} | {'最终得分':<15}")
    print("-" * 85)
    for cand, base_sim, bonus, final_score in results:
        bonus_str = f"{bonus:>+.4f}" if has_scorer else "N/A"
        print(f"{cand:<30} | {base_sim:>.4f}{' ':>5} | {bonus_str:<12} | {final_score:>.4f}")
    print("-" * 85)


def main():
    try:
        with open("config/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[!] 未找到 config/config.yaml")
        return

    # 1. 在循环外部初始化基础组件（只加载一次模型）
    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])

    if 'heavyweight_candidate' not in config:
        config['heavyweight_candidate'] = {}

    stats_path = config['heavyweight_candidate'].get('naming_stats_path', 'naming_stats.json')
    if not os.path.exists(stats_path):
        print(f"\n[!] 提示: {stats_path} 不存在，将以无统计修正分模式运行。")

    print("\n[*] Loading Engines (This may take a minute)...")
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    llm_name = config.get('llm_generator_name', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)

    generator = HeavyWeightCandidateGenerator(
        mlm_engine=mlm_engine,
        llm_client=llm_client,
        analyzer=analyzer,
        config=config['heavyweight_candidate']
    )

    # 🌟 防 OOM 分块批处理大小设置
    MAX_BATCH_SIZE = 4

    # 2. 进入交互主循环
    while True:
        print("\n" + "=" * 60)
        print("📥 请输入/粘贴 C/C++ 函数代码")
        print("💡 结束代码输入：新起一行输入 'END'")
        print("🚪 退出程序：输入 'OUT'")
        print("=" * 60)

        lines = []
        is_exit = False
        while True:
            try:
                line = input()
                if line.strip().upper() == "OUT":
                    is_exit = True
                    break
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            except EOFError:
                break

        if is_exit:
            print("[*] 正在安全退出程序...")
            break

        source_code_str = "\n".join(lines)
        if not source_code_str.strip():
            print("[!] 未检测到有效代码输入，请重新输入。")
            continue

        # 3. 提取标识符
        code_bytes = source_code_str.encode("utf-8")
        identifiers = analyzer.extract_identifiers(code_bytes)
        if not identifiers:
            print("[!] 未能从代码中提取到有效标识符。")
            continue

        print(f"\n[*] 当前代码共提取到 {len(identifiers)} 个标识符: ")
        print(", ".join(identifiers.keys()))

        # 4. 选择目标变量
        user_input = input(
            "\n请输入要测试的变量名 (多个变量用空格分隔, 输入 'ALL' 测试全部, 输入 'BACK' 返回): ").strip()

        if user_input.upper() == "BACK":
            continue

        target_vars = []
        if user_input.upper() == "ALL":
            target_vars = list(identifiers.keys())
        else:
            selected = user_input.split()
            for v in selected:
                if v in identifiers:
                    target_vars.append(v)
                else:
                    print(f"[-] 警告：变量 '{v}' 不在提取列表中，已忽略。")

        if not target_vars:
            print("[!] 未选择任何有效变量。")
            continue

        # 5. 组装批量任务 (代码切片)
        batch_tasks = []
        print(f"\n[*] 正在为 {len(target_vars)} 个变量执行代码切片 (Slicing)...")
        for var in target_vars:
            sliced_code = source_code_str
            if hasattr(analyzer, 'get_folded_code'):
                sliced_code = analyzer.get_folded_code(code_bytes, var)
            batch_tasks.append({
                "target_name": var,
                "code_str": sliced_code
            })

        # 6. 分块批量生成 (防 OOM 核心逻辑)
        total_tasks = len(batch_tasks)
        num_chunks = math.ceil(total_tasks / MAX_BATCH_SIZE)

        print(f"[*] 开始 GPU 并行生成，分为 {num_chunks} 个批次 (最大 {MAX_BATCH_SIZE} 任务/批次)...")
        start_time = time.perf_counter()

        all_results = {}
        for i in range(0, total_tasks, MAX_BATCH_SIZE):
            chunk = batch_tasks[i:i + MAX_BATCH_SIZE]
            current_chunk_idx = (i // MAX_BATCH_SIZE) + 1
            print(f"  -> 正在处理第 {current_chunk_idx}/{num_chunks} 批次...")

            try:
                chunk_pool = generator.generate_candidates(
                    batch_tasks=chunk,
                    top_k_mlm=40,
                    top_n_keep=50
                )
                all_results.update(chunk_pool)
            except Exception as e:
                print(f"[!] 第 {current_chunk_idx} 批次处理失败: {e}")
            finally:
                if 'chunk_pool' in locals():
                    del chunk_pool
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed_time = time.perf_counter() - start_time
        print(f"[*] 批量生成完成！总耗时: {elapsed_time:.2f} 秒 (平均 {elapsed_time / total_tasks:.2f} 秒/变量)")

        # 7. 评估并打印每一个变量的得分
        for var_name, cands in all_results.items():
            evaluate_and_print_candidates(generator, code_bytes, var_name, cands)

        # 循环结束后的清理
        gc.collect()
        torch.cuda.empty_cache()

    print("[*] 程序已结束。")


if __name__ == "__main__":
    main()