import os
from tabulate import tabulate
from attacks.generators import CodeBasedCandidateGenerator
from utils.model_zoo import ModelZoo
from utils.ast_tools import IdentifierAnalyzer


def run_kernel_code_test(generator, analyzer):
    # 1. 准备内核源码样例
    kernel_code = (
        "static int ipgre_close(struct net_device *dev)\n"
        "{\n"
        "    struct ip_tunnel *t = netdev_priv(dev);\n"
        "\n"
        "    if (ipv4_is_multicast(t->parms.iph.daddr) && t->mlink) {\n"
        "        struct in_device *in_dev;\n"
        "        in_dev = inetdev_by_index(dev_net(dev), t->mlink);\n"
        "        if (in_dev) {\n"
        "            ip_mc_dec_group(in_dev, t->parms.iph.daddr);\n"
        "            in_dev_put(in_dev);\n"
        "        }\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )

    # 2. 定义我们要测试的标识符（涵盖函数名、指针、局部变量）
    targets = ["ipgre_close", "in_dev", "dev", "t", "mlink","ipv4_is_multicast"]

    print("\n" + " 🧪 Linux 内核代码重命名测试 ".center(80, "="))
    print(f"原始函数名: ipgre_close")

    summary = []
    ids = analyzer.extract_identifiers(kernel_code.encode("utf-8"))

    for target in targets:
        print(f"正在分析标识符: {target}...")

        # 生成候选词
        candidates = generator.generate_candidates(
            code=kernel_code,
            target_name=target,
            identifiers=ids,
            top_k_mlm=50,
            top_n_keep=10
        )

        summary.append([
            target,
            "Function" if "close" in target else "Variable/Member",
            len(candidates),
            ", ".join(candidates[:6])
        ])

    # 3. 展示结果
    headers = ["标识符 (Target)", "类型", "生成数量", "候选词预览 (Top 6)"]
    print("\n" + " 生成结果报告 ".center(100, "-"))
    print(tabulate(summary, headers=headers, tablefmt="grid"))
    print("-" * 100)


if __name__ == "__main__":
    # 初始化环境 (请确保路径与你本地一致)
    model_configs = {"CodeBERT": "./models/binary_diversevul_codebert"}

    if not os.path.exists(model_configs["CodeBERT"]):
        print("错误：请检查模型路径！")
        exit()

    print("[*] 正在加载 CodeBERT MLM 模型...")
    zoo = ModelZoo(model_configs)
    # 内核代码通常是 C 语言，这里指定 C
    analyzer = IdentifierAnalyzer(lang="c")
    generator = CodeBasedCandidateGenerator(zoo, analyzer)

    run_kernel_code_test(generator, analyzer)