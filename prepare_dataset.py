from datasets import load_dataset

# 加载全量数据
dataset = load_dataset("parquet", data_files="./data/full_dataset.parquet", split="train")

# 1. 直接切分出 2,000 条作为测试集
# seed 保证每次运行切分的结果一致
dataset = dataset.train_test_split(test_size=2000, seed=42)

# 2. 保存测试集
dataset['test'].to_parquet("./data/test_dataset.parquet")

# 3. 剩下的作为训练集候选池
dataset['train'].to_parquet("./data/train_pool.parquet")

print("[+] 测试集 (2000条) 已保存至 ./data/test_dataset.parquet")
print("[+] 训练候选池已保存至 ./data/train_pool.parquet")