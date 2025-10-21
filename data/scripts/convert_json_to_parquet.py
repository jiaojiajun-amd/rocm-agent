import pandas as pd

# 读取 JSON list 文件
df = pd.read_json('../rocmprim.json')

print(f"len df is {len(df)}")

# 转换为 Parquet
df.to_parquet('../train.parquet', engine='pyarrow', compression='snappy')

df.to_parquet('../test.parquet', engine='pyarrow', compression='snappy')

print("转换完成!")
