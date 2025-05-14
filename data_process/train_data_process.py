import pandas as pd
import re
import numpy as np

file_path = "xxxxx"
df = pd.read_excel(file_path)
expected_columns = ['咨询问题', '问题类别', '内容', '回复']
if not all(col in df.columns for col in expected_columns):
    print("缺少以下列：", [col for col in expected_columns if col not in df.columns])
    raise ValueError("Excel 文件缺少必要列")
df = df[expected_columns]
df = df.replace("", np.nan).infer_objects(copy=False)
df = df.dropna()
if len(df) == 0:
    raise ValueError("DataFrame 在删除 NaN 后为空，请检查输入数据！")
def contains_valid_characters(value):
    if isinstance(value, str):
        return bool(re.search(r'[a-zA-Z\u4e00-\u9fa5]', value))
    return False
for column in df.columns:
    df = df[df[column].apply(contains_valid_characters)]
    if len(df) == 0:
        raise ValueError(f"过滤列 '{column}' 后 DataFrame 为空，请检查该列数据！")

df = df[df['问题类别'] != '其他']
df = df[df['问题类别'] != '婚姻关系']
df = df[df['问题类别'] != '交通事故成因、车速、痕迹鉴定']
df.drop_duplicates(subset=['咨询问题', '问题类别', '内容', '回复'], inplace=True)
df['问题类别'] = df['问题类别'].replace('邻里关系', '邻里纠纷')
df.drop_duplicates(subset=['咨询问题'], inplace=True)
df.dropna(subset=['咨询问题', '问题类别', '内容', '回复'], how='any', inplace=True)
df['回复'] = df['回复'].apply(lambda x: re.sub(r'^根据您所表达的需求，我们为您提供如下信息：', "", x)).str.strip()
valid_categories = df['问题类别'].unique()
print(len(valid_categories))
df.to_excel('xxxxx', index=False)
