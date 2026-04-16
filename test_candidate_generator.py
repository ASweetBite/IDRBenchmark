import os
import torch
import torch.nn.functional as F
import yaml

# 请确保这些路径与你本地的项目结构匹配
from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer
from utils.mlm_engine import MLMEngine
from utils.miner import NamingDataMiner  # 引入离线挖掘器


def evaluate_and_print_candidates(generator, code_str: str, target_name: str):
    """
    调用最新版生成器获取候选词，并重构计算过程，打印 Token 级相似度与 NLP 修正分。
    """
    print(f"\n[*] 开始为变量 '{target_name}' 生成候选词...")

    # 1. 调用最新的生成方法获取最终存活的候选词列表
    # (内部 _verify_and_filter 时会打印一次过滤日志，这里我们将提取最终结果再次拆解计分)
    candidates = generator.generate_candidates(code_str, target_name)

    if not candidates:
        print(f"[-] 未能为 '{target_name}' 生成任何有效候选词。")
        return

    # 2. 获取语法树标识符，精准定位最佳上下文位置
    code_bytes = code_str.encode('utf-8')
    identifiers = generator.analyzer.extract_identifiers(code_bytes)

    if target_name not in identifiers:
        print(f"[-] AST 提取失败：语法树中未找到变量 {target_name}")
        return

    # 使用与生成器内部完全一致的上下文挑选逻辑
    best_occ_idx = generator._find_best_context_occurrence(code_bytes, identifiers[target_name])
    target_info = identifiers[target_name][best_occ_idx]

    # 获取该变量的实体类型并转为大写，适配 Scorer (默认 fallback 为 VARIABLE)
    raw_entity_type = target_info.get('entity_type', 'variable')
    entity_type = 'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE'

    # 粗略判断是否为布尔类型
    if entity_type == 'VARIABLE' and target_name.startswith(('is_', 'has_', 'can_', 'should_')):
        entity_type = 'BOOLEAN_VAR'

    # 3. 复用 AST 局部切片方法，获取最小逻辑语句的 prefix 和 suffix
    local_prefix, local_suffix = generator._extract_local_context_ast(
        code_bytes,
        target_info['start'],
        target_info['end']
    )

    # 4. 获取 原变量 的 Token Embedding (✨ 最新核心科技)
    orig_emb = generator._get_variable_token_embeddings(
        [local_prefix], [target_name], [local_suffix]
    ).to(generator.mlm_engine.device)

    # 5. 获取 所有候选词 的 Token Embedding
    prefixes = [local_prefix] * len(candidates)
    suffixes = [local_suffix] * len(candidates)
    cand_embs = generator._get_variable_token_embeddings(
        prefixes, candidates, suffixes
    ).to(generator.mlm_engine.device)

    # 6. 计算精准的 Token 级 Cosine Similarity
    sims = F.cosine_similarity(orig_emb, cand_embs)

    # 7. 整合数据，融入 NLP 统计打分器 (如果已集成)
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

    # 8. 按最终得分降序排序
    results.sort(key=lambda x: x[3], reverse=True)

    # 9. 格式化输出炫酷的分析表格
    print(f"\n[+] '{target_name}' 的最终输出候选词及得分排名 (共 {len(results)} 个):")
    print(f"    上下文提取策略: {entity_type} Context")
    print("-" * 85)
    print(f"{'候选词 (Candidate)':<30} | {'Token 相似度':<15} | {'NLP 修正分':<12} | {'最终得分':<15}")
    print("-" * 85)
    for cand, base_sim, bonus, final_score in results:
        # 高亮原变量的基准分
        if cand == target_name:
            print(f"{cand:<30} | {base_sim:>.4f} (Original)")
        else:
            bonus_str = f"{bonus:>+.4f}" if has_scorer else "N/A"
            print(f"{cand:<30} | {base_sim:>.4f}{' ':>5} | {bonus_str:<12} | {final_score:>.4f}")
    print("-" * 85)


def main():
    try:
        with open("config/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[!] 未找到 config/config.yaml，请确保在根目录下执行或修改路径。")
        return

    # 初始化基础组件
    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])

    # ==================== [新增] 动态数据挖掘与字典对齐 ====================
    if 'heavyweight_candidate' not in config:
        config['heavyweight_candidate'] = {}

    stats_path = config['heavyweight_candidate'].get('naming_stats_path', 'naming_stats.json')
    dataset_path = config.get('run_params', {}).get('dataset', '')

    if not os.path.exists(stats_path):
        print(f"\n[!] 启发式命名统计字典 '{stats_path}' 不存在。")
        if dataset_path and os.path.exists(dataset_path):
            print(f"[*] 正在启动离线数据挖掘程序 (基于数据集: {dataset_path})...")
            miner = NamingDataMiner(analyzer)
            miner.mine_parquet(dataset_path)
            miner.export_json(stats_path)
        else:
            print("[!] 未找到有效数据集路径，将以纯 NLP 规则模式启动测试。")
    else:
        print(f"\n[*] 发现已存在的命名统计字典: {stats_path}，已加载。")

    # 显式将字典路径写入 config，供 Generator 的 __init__ 读取
    config['heavyweight_candidate']['naming_stats_path'] = stats_path
    # =========================================================================

    print("\n[*] Loading MLM Engine...")
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    # 初始化生成器 (内部会自动实例化 self.scorer)
    generator = HeavyWeightCandidateGenerator(
        mlm_engine,
        analyzer,
        config=config['heavyweight_candidate']
    )

    print("\n--- 📥 请输入/粘贴 C/C++ 函数代码 ---")
    print("💡 输入完成后，请在新的一行输入 'END' 并回车结束读取：")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    source_code_str = "\n".join(lines)

    if not source_code_str.strip():
        print("未检测到输入，退出。")
        return

    code_bytes = source_code_str.encode("utf-8")
    identifiers = analyzer.extract_identifiers(code_bytes)

    if not identifiers:
        print("未能提取到有效标识符。")
        return

    # 自动选择出现频率最高的变量作为默认目标
    target_var = max(identifiers.keys(), key=lambda k: len(identifiers[k]))

    print(f"\n自动选择高频变量: '{target_var}'")
    user_input = input("输入你想测试的新变量名 (或按回车继续使用默认): ").strip()
    if user_input:
        target_var = user_input

    if target_var not in identifiers:
        print(f"[-] 错误：你输入的变量 '{target_var}' 不在提取的标识符列表中。")
        return

    # 如果有代码折叠功能，调用折叠切片
    if hasattr(analyzer, 'get_folded_code'):
        sliced_code = analyzer.get_folded_code(code_bytes, target_var)
    else:
        sliced_code = source_code_str

    # 执行测试评估
    evaluate_and_print_candidates(generator, sliced_code, target_var)


if __name__ == "__main__":
    main()