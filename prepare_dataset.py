import polars as pl
import time


def process_bigvul_polars(input_path, output_path):
    print(f"开始处理数据集: {input_path}")
    start_time = time.time()

    # 1. 扫描本地文件（惰性读取，不消耗大量内存）
    # 如果你的原文件是 jsonl 格式，请将 scan_csv 换成 scan_ndjson
    lazy_df = pl.scan_csv(input_path, ignore_errors=True)

    # 2. 构建查询计算图
    query = (
        lazy_df
        # 过滤语言，为了鲁棒性，把小写也加上
        .filter(pl.col("lang").is_in(['C', 'C++', 'CPP', 'c', 'c++', 'cpp']))
        # 仅选择需要的列，并对 CWE ID 进行重命名
        .select([
            pl.col("func_before"),
            pl.col("func_after"),
            pl.col("CWE ID").alias("cwe")
        ])
        # (可选) 过滤掉 func_before 和 func_after 都为空的无效数据
        # .drop_nulls(subset=["func_before", "func_after"])
    )

    # 3. 触发流式计算并直接写入 Parquet（流式写入）
    query.sink_parquet(output_path, compression="snappy")

    end_time = time.time()
    print(f"处理完成！已保存至 {output_path}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 替换为你的本地文件路径
    INPUT_FILE = "data/MSR_data_cleaned.csv"
    OUTPUT_FILE = "data/bigvul_polars.parquet"

    process_bigvul_polars(INPUT_FILE, OUTPUT_FILE)