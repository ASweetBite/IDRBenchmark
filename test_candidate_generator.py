import os
import re
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer
from utils.llm_loder import LocalLLMClient
from utils.mlm_engine import MLMEngine
from utils.miner import NamingDataMiner


def evaluate_and_print_candidates(generator, code_str: str, target_name: str):
    """
    调用最新版生成器获取候选词，并重构计算过程，打印 Token 级相似度与 NLP 修正分。
    """
    print(f"\n[*] 开始为变量 '{target_name}' 生成候选词 (LLM + MLM Filter)...")

    # 1. 调用最新的生成方法
    candidates = generator.generate_candidates(code_str, target_name)

    if not candidates:
        print(f"[-] 未能为 '{target_name}' 生成任何有效且合法的候选词。")
        return

    # 2. 提取数据计算排版分数
    code_bytes = code_str.encode('utf-8')
    identifiers = generator.analyzer.extract_identifiers(code_bytes)

    best_occ_idx = generator._find_best_context_occurrence(code_bytes, identifiers[target_name])
    target_info = identifiers[target_name][best_occ_idx]

    raw_entity_type = target_info.get('entity_type', 'variable')
    entity_type = 'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE'
    if entity_type == 'VARIABLE' and target_name.startswith(('is_', 'has_', 'can_', 'should_')):
        entity_type = 'BOOLEAN_VAR'

    local_prefix, local_suffix = generator._extract_local_context_ast(
        code_bytes, target_info['start'], target_info['end']
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


import os
import re
import yaml
import torch
import torch.nn.functional as F
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 保持原有导入不变
from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer
from utils.mlm_engine import MLMEngine
from utils.miner import NamingDataMiner


# ... [LocalLLMClient 类保持不变] ...

# ... [evaluate_and_print_candidates 函数保持不变] ...

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

    # 2. 进入交互主循环
    while True:
        print("\n" + "=" * 50)
        print("📥 请输入/粘贴 C/C++ 函数代码")
        print("💡 结束代码输入：新起一行输入 'END'")
        print("🚪 退出程序：输入 'OUT'")
        print("=" * 50)

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

        # 4. 选择目标变量
        target_var = max(identifiers.keys(), key=lambda k: len(identifiers[k]))
        print(f"\n当前代码提取到 {len(identifiers)} 个标识符。")
        print(f"自动选择高频变量: '{target_var}'")

        user_input = input(f"请输入要测试的变量名 (直接回车使用 '{target_var}', 输入 'BACK' 返回代码输入): ").strip()

        if user_input.upper() == "BACK":
            continue
        if user_input:
            if user_input not in identifiers:
                print(f"[-] 错误：变量 '{user_input}' 不在提取列表中。")
                continue
            target_var = user_input

        # 5. 执行评估
        sliced_code = source_code_str
        if hasattr(analyzer, 'get_folded_code'):
            sliced_code = analyzer.get_folded_code(code_bytes, target_var)

        try:
            evaluate_and_print_candidates(generator, sliced_code, target_var)
        except Exception as e:
            print(f"[!] 处理过程中发生错误: {e}")

        # 6. 显存与内存清理（循环运行必备，防止 OOM）
        gc.collect()
        torch.cuda.empty_cache()

    print("[*] 程序已结束。")


if __name__ == "__main__":
    main()