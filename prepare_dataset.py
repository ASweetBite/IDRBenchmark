import pandas as pd
import csv
import pyarrow as pa
import pyarrow.parquet as pq
import os

# 修复报错：设置 500MB 限制
csv.field_size_limit(500 * 1024 * 1024)


def process_bigvul_chunked(input_csv, output_parquet, chunk_size=10000):
    print(f"[*] 开始处理: {input_csv}")

    valid_langs = {'C', 'C++', 'CPP'}
    writer = None  # Parquet 写句柄

    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)

        chunk = []
        count = 0

        for row in reader:
            try:
                lang = str(row.get('lang', '')).upper()
                if lang not in valid_langs:
                    continue

                func = row.get('func_before')
                cwe = row.get('CWE ID', '')

                if func:
                    chunk.append({'func': func, 'cwe': cwe if cwe else ""})
            except:
                continue

            # 当 chunk 达到大小时写入一次
            if len(chunk) >= chunk_size:
                df_chunk = pd.DataFrame(chunk)
                table = pa.Table.from_pandas(df_chunk)

                # 初始化写入器
                if writer is None:
                    writer = pq.ParquetWriter(output_parquet, table.schema)

                writer.write_table(table)
                count += len(chunk)
                print(f"[*] 已写入 {count} 条记录...")
                chunk = []  # 清空内存

        # 处理剩余的最后一块
        if chunk:
            df_chunk = pd.DataFrame(chunk)
            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)
            writer.write_table(table)
            count += len(chunk)

    if writer:
        writer.close()
        print(f"[+] 处理完成! 总记录数: {count}")
    else:
        print("[!] 没有提取到数据。")


if __name__ == "__main__":
    INPUT_CSV = "data/MSR_data_cleaned.csv"
    OUTPUT_PARQUET = "data/big_vul.parquet"

    if os.path.exists(INPUT_CSV):
        process_bigvul_chunked(INPUT_CSV, OUTPUT_PARQUET)
    else:
        print(f"[!] 找不到文件: {INPUT_CSV}")