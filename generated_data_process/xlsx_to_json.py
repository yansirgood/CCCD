import os
import json
import re
import pandas as pd

INSTRUCT_CONTENT = """你是一个法律咨询助手，负责对用户提出的法律问题进行多分类，并生成相应的法律建议。
问题的类别有：劳动人事，合同纠纷，民事，债权债务，消费维权，刑事犯罪，交通事故，侵权纠纷，房产纠纷，工伤事故，国内仲裁，刑事，子女抚养，公司相关，社会保障，土地纠纷，遗产继承，物业管理，行政，征收补偿，拖欠农民工工资纠纷，建筑工程，治安消防，其他纠纷，医患纠纷，金融保险，老人赡养，劳动争议纠纷，知识产权，国内公证，损害赔偿纠纷，其他消费纠纷，邻里纠纷，房产宅基地纠纷，婚姻家庭纠纷，道路交通事故纠纷，法医病理司法鉴定，电子商务纠纷，法医临床司法鉴定，文书司法鉴定，物业纠纷，山林土地纠纷，生产经营纠纷，征地拆迁纠纷，证券期货，医疗纠纷，法医精神病司法鉴定。"""
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"- \*\*|\*\*", "", text)
    text = re.sub(r"\d+\.", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def convert_xlsx_to_json(input_path, output_json_path):

    json_data = []
    total_rows = 0
    converted_rows = 0
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
        required_columns = ['咨询问题', '问题类别', '内容', '回复']
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Excel文件缺少必要列：{missing}")
        total_rows = len(df)
        for _, row in df.iterrows():
            try:
                cleaned_data = {
                    "咨询问题": clean_text(row['咨询问题']),
                    "问题类别": clean_text(row['问题类别']),
                    "内容": clean_text(row['内容']),
                    "回复": clean_text(row['回复'])
                }
                if all(value != "" for value in cleaned_data.values()):
                    json_item = {
                        "instruct": INSTRUCT_CONTENT,
                        "input": {
                            "咨询问题": cleaned_data["咨询问题"],
                            "内容": cleaned_data["内容"],
                            "回复": cleaned_data["回复"]
                        },
                        "label": cleaned_data["问题类别"]
                    }
                    json_data.append(json_item)
                    converted_rows += 1
                else:
                    print(f"跳过第{_ + 1}行，存在空字段")

            except Exception as row_error:
                print(f"处理第{_ + 1}行时出错: {str(row_error)}")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成：共处理 {total_rows} 行，成功转换 {converted_rows} 行")
        print(f"失败行数：{total_rows - converted_rows}")
        print(f"结果已保存至：{os.path.abspath(output_json_path)}")

    except Exception as e:
        print(f"发生严重错误: {str(e)}")

if __name__ == "__main__":
    input_excel = "xxxxx"
    output_json = "xxxxx"
    convert_xlsx_to_json(input_excel, output_json)
