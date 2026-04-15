import torch
import torch.nn.functional as F
import yaml
from yaml import parser

from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer
from utils.mlm_engine import MLMEngine


def evaluate_and_print_candidates(generator, code_str: str, target_name: str):
    """
    调用生成器获取候选词，并计算打印它们与原变量的代码级上下文相似度。
    """
    print(f"[*] 开始为变量 '{target_name}' 生成候选词...")

    # 1. 调用你封装好的候选词生成方法
    candidates = generator.generate_candidates(code_str, target_name)

    if not candidates:
        print(f"[-] 未能为 '{target_name}' 生成任何有效候选词。")
        return

    # 2. 获取语法树标识符，以提取目标变量的字节起始位置
    code_bytes = code_str.encode('utf-8')
    identifiers = generator.analyzer.extract_identifiers(code_bytes)

    if target_name not in identifiers:
        print(f"[-] AST 提取失败：语法树中未找到变量 {target_name}")
        return

    target_info = identifiers[target_name][0]

    # 3. 复用你的 AST 局部切片方法，获取原句的 prefix 和 suffix
    local_prefix, local_suffix = generator._extract_local_context_ast(
        code_bytes,
        target_info['start'],
        target_info['end']
    )

    # 4. 组装原代码片段，并获取 Original Embedding
    original_local_str = local_prefix + target_name + local_suffix
    orig_emb = generator._get_code_embedding_batched([original_local_str])
    orig_emb = orig_emb.to(generator.mlm_engine.device)

    # 5. 组装所有候选词替换后的代码片段
    cand_codes = [local_prefix + cand + local_suffix for cand in candidates]

    # 6. 批量获取所有候选代码的 Embedding
    cand_embs = generator._get_code_embedding_batched(cand_codes)
    cand_embs = cand_embs.to(generator.mlm_engine.device)

    # 7. 计算 Cosine Similarity
    sims = F.cosine_similarity(orig_emb, cand_embs)

    # 8. 整合数据并按相似度降序排序
    results = []
    for cand, sim in zip(candidates, sims):
        results.append((cand, sim.item()))

    results.sort(key=lambda x: x[1], reverse=True)

    # 9. 格式化输出结果
    print(f"\n[+] '{target_name}' 的 Top-{len(results)} 候选词及语义相似度:")
    print("-" * 65)
    print(f"{'候选词 (Candidate)':<35} | {'代码级相似度 (Cosine)':<20}")
    print("-" * 65)
    for cand, sim in results:
        # 高亮原变量供对比参考
        if cand == target_name:
            print(f"{cand:<35} | {sim:.4f} (Original)")
        else:
            print(f"{cand:<35} | {sim:.4f}")
    print("-" * 65)


def main():
    try:
        with open("config/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        pass

    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    light_cand_config = {
        "candidate": config['lightweight_candidate']
    }

    generator = HeavyWeightCandidateGenerator(
        mlm_engine,
        analyzer,
        config=config['heavyweight_candidate']
    )

    print("--- 📥 请输入/粘贴 C/C++ 函数代码 ---")
    print("💡 输入完成后，请在新的一行输入 'END' 并回车结束读取：")

    # 1. 使用哨兵值循环读取，不触发 EOF
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

    # 选择目标变量
    target_var = max(identifiers.keys(), key=lambda k: len(identifiers[k]))

    # 由于没有触发 EOF，这里的 input() 会正常工作
    print(f"\n自动选择高频变量: '{target_var}'")
    user_input = input("输入新变量名或按回车继续: ").strip()
    if user_input:
        target_var = user_input
    scliced_code = analyzer.get_folded_code(code_bytes, target_var)
    evaluate_and_print_candidates(generator, scliced_code, target_var)

# ==========================================
# 使用示例 (Mock 实例化流程)
# ==========================================
if __name__ == "__main__":
    """Orchestrates the evaluation of model robustness against various renaming attacks."""

    main()
