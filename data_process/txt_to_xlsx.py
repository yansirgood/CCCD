import pandas as pd
import os
import logging
from datetime import datetime


def setup_logger(log_file: str = "logs/txt_to_excel.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_txt_file(file_path: str, logger: logging.Logger) -> dict:

    try:
        filename = os.path.basename(file_path)
        category = filename.split("_", 1)[0]

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sample = {"问题类别": category}
        current_field = None

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if "- **咨询问题**：" in line:
                sample["咨询问题"] = line.split("**咨询问题**：", 1)[1].strip()
                current_field = None
            elif "- **问题类别**：" in line:
                continue
            elif "- **内容**：" in line:
                sample["内容"] = line.split("**内容**：", 1)[1].strip()
                current_field = "内容"
            elif "- **回复**：" in line:
                reply = line.split("**回复**：", 1)[1].strip()
                sample["回复"] = reply.replace("**", "")
                current_field = "回复"
            elif current_field and line:
                if current_field == "回复":
                    sample[current_field] += "\n" + line.replace("**", "")
                else:
                    sample[current_field] += "\n" + line.strip()

        required_fields = ["咨询问题", "问题类别", "内容", "回复"]
        missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
        if not missing_fields:
            logger.info(f"成功解析文件：{file_path}，类别：{category}")
            return sample
        else:
            logger.error(f"解析失败，缺少字段 {missing_fields}：{file_path}")
            return None
    except Exception as e:
        logger.error(f"解析文件 {file_path} 时出错：{str(e)}")
        return None
def txt_to_excel(input_dir: str = "generated_data", output_dir: str = "output"):

    logger = setup_logger()
    logger.info(f"开始将 {input_dir} 中的 txt 文件提取为 xlsx")
    if not os.path.exists(input_dir):
        logger.error(f"目录 {input_dir} 不存在")
        print(f"错误：目录 {input_dir} 不存在")
        return
    txt_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir)
                 for file in files if file.endswith(".txt")]
    if not txt_files:
        logger.error(f"在 {input_dir} 中未找到任何 .txt 文件")
        print(f"错误：在 {input_dir} 中未找到任何 .txt 文件")
        return
    logger.info(f"找到 {len(txt_files)} 个 txt 文件")
    samples = []
    for txt_file in txt_files:
        sample = parse_txt_file(txt_file, logger)
        if sample:
            samples.append(sample)
    if not samples:
        logger.error("未成功解析任何 txt 文件，无法生成 xlsx")
        print("错误：未成功解析任何 txt 文件，无法生成 xlsx")
        return
    df = pd.DataFrame(samples, columns=["咨询问题", "问题类别", "内容", "回复"])
    total_samples = len(df)
    logger.info(f"成功解析 {total_samples} 个样本")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"generated_samples_{timestamp}.xlsx")
    df.to_excel(output_path, index=False)
    logger.info(f"结果已保存至：{output_path}")
    print(f"结果已保存至：{output_path}")
    print(f"总样本数：{total_samples}")
if __name__ == "__main__":
    txt_to_excel()
