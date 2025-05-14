
import pandas as pd
import re
import numpy as np


file_path = "xxxxx"
df = pd.read_excel(file_path)
df = df[['咨询问题', '问题类别', '内容', '回复']]
def clean_content(content):
    if not isinstance(content, str):
        return ""
    match = re.search(
        r'您好！欢迎关注中国法律服务网。\s*根据您所表达的需求，我们为您提供如下信息：(.*?)\n中国法律服务网平台为您提供以上信息，仅供您参考。如有疑问，欢迎进一步咨询。感谢您对中国法律服务网的关注和支持！',
        content, re.DOTALL)
    return match.group(1).strip() if match else ""

df['回复'] = df['回复'].apply(clean_content)
df = df.replace("", np.nan)
df = df.dropna()
def contains_valid_characters(value):
    if isinstance(value, str):
        return bool(re.search(r'[a-zA-Z\u4e00-\u9fa5]', value))
    return False

for column in df.columns:
    df = df[df[column].apply(contains_valid_characters)]

df = df[df['问题类别'] != '其他']
df.drop_duplicates(subset=['咨询问题', '问题类别', '内容', '回复'], inplace=True)
df.dropna(subset=['咨询问题', '问题类别', '内容', '回复'], how='any', inplace=True)
df = df[df['问题类别'] != '婚姻关系']
df['问题类别'] = df['问题类别'].replace('邻里关系', '邻里纠纷')
data = pd.read_excel('train.xlsx')
valid_categories = data['问题类别'].unique()
df = df[df['问题类别'].isin(valid_categories)]
df.to_excel('xxxxx', index=False)
print('保存成功')



