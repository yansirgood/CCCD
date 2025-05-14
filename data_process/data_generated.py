import requests
import json
import yaml
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

class DeepSeekClient:
    def __init__(self, api_url: str, api_key: str, model: str, max_tokens: int):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    def generate_data(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens
        }
        response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API 请求失败: {response.status_code}, {response.text}")

class PromptGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.client = DeepSeekClient(
            api_url=self.config["api"]["url"],
            api_key=self.config["api"]["api_key"],
            model=self.config["api"]["model"],
            max_tokens=self.config["api"]["max_tokens"]
        )
        self.logger = setup_logger(os.path.join(self.config["paths"]["log_dir"], "app.log"))
        self.data = pd.read_excel(r"data_cleaned.xlsx")
        self.generated_samples = []

    def get_base_prompt_template(self) -> str:
        return """你是一个法律领域的专家，擅长生成高质量的法律纠纷 QA 样本。请基于以下结构和要求，生成一个新的法律纠纷 QA 样本。

#### 样本结构
每个样本包含以下四个字段：
- **咨询问题**：用户提出的法律纠纷问题的简短概述，通常为一句话，概括问题的核心。
- **问题类别**：问题的法律纠纷类别，例如“合同纠纷”、“债权债务”、“租赁纠纷”等，需准确反映问题的法律性质。
- **内容**：用户提出的法律纠纷问题的详细描述，包含具体的事实（如时间、地点、行为等）、争议点和相关细节，避免模糊或不完整。
- **回复**：专业法律工作人员的回复，需基于中国法律法规，引用具体条文（引用法条时要严格引用完整的法条），并提供明确的法律建议或解决方案。

#### 生成要求
1. **咨询问题**：简洁明了，直接点明用户困惑的核心问题。
2. **问题类别**：与“内容”中的纠纷性质一致，类别名称需规范（如“合同纠纷”而非“合同问题”）。
3. **内容**：详细描述问题的背景和争议点，包含关键信息：
   - **主体**：涉及的主要当事人（如“用户”、“朋友”、“装修公司”）。
   - **客体**：纠纷涉及的对象（如“房屋”、“款项”、“钥匙”）。
   - **行为**：导致纠纷的具体行为或事件（如“未付款”、“不归还钥匙”）。
   - **争议点**：双方分歧的具体内容。
4. **回复**：逻辑清晰，包含：
   - 法律依据：引用《中华人民共和国民法典》等具体条款。
   - 分析：结合“内容”中的事实，说明法律适用。
   - 建议：提供可行的解决方案或后续行动建议。

#### 示例
以下是两个示例，帮助你理解样本的结构和内容风格：
{examples}

#### 生成指令
请生成一个新的法律纠纷 QA 样本，问题类别为“{category}”，确保包含“咨询问题”、“问题类别”、“内容”和“回复”四个字段。情境需与示例中的具体情境和细节有所不同。回复必须引用《中华人民共和国民法典》等具体条文，并保持法律专业性和逻辑性:"""

    def format_example(self, sample: pd.Series) -> str:
        return f"""**示例**
- **咨询问题**：{sample['咨询问题']}
- **问题类别**：{sample['问题类别']}
- **内容**：{sample['内容']}
- **回复**：{sample['回复']}"""

    def generate_sample_for_category(self, category: str, group: pd.DataFrame, used_indices: set) -> None:
        self.logger.info(f"正在处理类别: {category} (样本数: {len(group)})")
        if len(group) < 2:
            self.logger.warning(f"类别 {category} 样本不足2条，跳过")
            return
        available_indices = set(group.index) - used_indices
        if len(available_indices) < 2:
            self.logger.info(f"类别 {category} 剩余样本不足2条，停止生成")
            return
        samples = group.loc[list(available_indices)].sample(n=2, replace=False, random_state=42)
        used_indices.update(samples.index)
        with open("used_samples.json", "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in used_samples.items()}, f)

        examples = "\n\n".join(self.format_example(row) for _, row in samples.iterrows())
        prompt = self.get_base_prompt_template().format(examples=examples, category=category)

        try:
            response = self.client.generate_data(prompt)
            generated_data = response["choices"][0]["message"]["content"]
            new_sample = self.parse_generated_sample(generated_data, category)
            if new_sample:
                self.generated_samples.append(new_sample)
                self.logger.info(f"类别 {category} 新样本生成成功")

            txt_path = os.path.join(self.config["paths"]["data_dir"],
                                    f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            save_to_file(generated_data, txt_path)
            self.logger.info(f"中间结果保存至: {txt_path}")

        except Exception as e:
            self.logger.error(f"类别 {category} 生成失败: {str(e)}")

    def generate_samples(self):
        grouped = self.data.groupby('问题类别')
        categories = list(grouped.groups.keys())
        self.logger.info(f"处理的类别: {categories}")
        global used_samples
        if os.path.exists("used_samples.json"):
            with open("used_samples.json", "r", encoding="utf-8") as f:
                used_samples = {k: set(v) for k, v in json.load(f).items()}
            self.logger.info("已加载已使用样本记录")
        else:
            used_samples = {category: set() for category in categories}
            self.logger.info("未找到已使用样本记录，初始化为空")
        def process_category(category):
            group = grouped.get_group(category)
            while len(set(group.index) - used_samples[category]) >= 2:
                self.generate_sample_for_category(category, group, used_samples[category])
        def save_results():
            if self.generated_samples:
                output_df = pd.DataFrame(self.generated_samples)
                output_path = os.path.join(self.config["paths"]["data_dir"], "generated_samples.xlsx")
                output_df.to_excel(output_path, index=False)
                self.logger.info(f"生成样本已保存至 {output_path}")
            else:
                self.logger.info("没有生成任何样本，未保存 xlsx 文件")
        def signal_handler(sig, frame):
            self.logger.info("检测到中断信号，正在保存已有结果...")
            save_results()
            with open("used_samples.json", "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in used_samples.items()}, f)
            self.logger.info("已使用样本索引已保存至 used_samples.json")
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        try:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(process_category, category): category for category in categories}
                for future in as_completed(futures):
                    category = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"线程处理类别 {category} 时出错: {str(e)}")
            save_results()
        except KeyboardInterrupt:
            self.logger.info("程序被手动中断，正在保存已有结果...")
            save_results()
            with open("used_samples.json", "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in used_samples.items()}, f)
            self.logger.info("已使用样本索引已保存至 used_samples.json")

    def parse_generated_sample(self, text: str, category: str) -> Dict[str, str]:
        try:
            lines = text.splitlines()
            sample = {}
            current_field = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "- **咨询问题**：" in line:
                    sample["咨询问题"] = line.split("**咨询问题**：", 1)[1].strip()
                    current_field = None
                elif "- **问题类别**：" in line:
                    sample["问题类别"] = line.split("**问题类别**：", 1)[1].strip()
                    current_field = None
                elif "- **内容**：" in line:
                    sample["内容"] = line.split("**内容**：", 1)[1].strip()
                    current_field = "内容"
                elif "- **回复**：" in line:
                    sample["回复"] = line.split("**回复**：", 1)[1].strip()
                    current_field = "回复"
                elif current_field and line:
                    sample["当前字段"] += "\n" + line.strip()

            required_fields = ["咨询问题", "问题类别", "内容", "回复"]
            missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
            if not missing_fields:
                return sample
            else:
                self.logger.error(f"解析失败，缺少字段 {missing_fields}: {text}")
                return None
        except Exception as e:
            self.logger.error(f"解析样本时出错: {str(e)}")
            return None

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
def setup_logger(log_file: str) -> logging.Logger:
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
def save_to_file(content: str, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

if __name__ == "__main__":
    config_path = "xxxxx"
    config = load_config(config_path)
    generator = PromptGenerator(config)
    generator.generate_samples()
