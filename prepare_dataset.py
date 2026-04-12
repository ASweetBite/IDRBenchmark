import polars as pl
import time


def process_bigvul_polars(input_path, output_path):
    """Processes the BigVul dataset using Polars lazy execution to filter specific C/C++ columns and saves it as a Parquet file."""
    print(f"开始处理数据集: {input_path}")
    start_time = time.time()

    lazy_df = pl.scan_csv(input_path, ignore_errors=True)

    query = (
        lazy_df
        .filter(pl.col("lang").is_in(['C', 'C++', 'CPP', 'c', 'c++', 'cpp']))
        .select([
            pl.col("func_before"),
            pl.col("func_after"),
            pl.col("CWE ID").alias("cwe")
        ])
    )

    query.sink_parquet(output_path, compression="snappy")

    end_time = time.time()
    print(f"处理完成！已保存至 {output_path}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    INPUT_FILE = "data/MSR_data_cleaned.csv"
    OUTPUT_FILE = "data/bigvul_polars.parquet"

    process_bigvul_polars(INPUT_FILE, OUTPUT_FILE)