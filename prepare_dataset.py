import pandas as pd

# 请将这里的路径替换为你本地某个 parquet 文件的实际路径
file_path = 'data/cleaned_dataset.parquet'

try:
    # 1. 读取数据 (只读取需要的两列以节省内存和加快速度)
    df = pd.read_parquet(file_path, columns=['func', 'vul'])

    # 2. 过滤出 vul = 1 的样本
    vul_df = df[df['vul'] == 1]

    if vul_df.empty:
        print("在这个文件中没有找到 vul = 1 的样本。")
    else:
        # 3. 随机抽取 3 个样本（如果总数不足 3 个则全部取出）
        sample_size = min(5, len(vul_df))
        samples = vul_df.sample(n=sample_size)

        print(f"该文件共有 {len(vul_df)} 个漏洞样本。以下是随机抽取的 {sample_size} 个：\n")
        print("=" * 60)

        # 4. 打印代码内容
        for index, row in samples.iterrows():
            print(f"【数据行索引 (Index): {index}】")
            print("【原始代码 (func)】:")
            print(row['func'])
            print("=" * 60)

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请检查路径是否正确。")
except Exception as e:
    print(f"读取或处理文件时发生错误: {e}")