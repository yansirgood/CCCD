# CCCD
欢迎使用中国法律案例分类数据集（CCCD），这是一个全面、高质量的分类数据集，包含 47 个类别的 55,988 个中国法律案例。
## 介绍
我们推出了中国法律案例分类数据集（CCCD）。这是一个大规模、高质量的分类数据集，包含超过55,988个中国法律案例，涵盖“劳动人事”、“消费者权益保护”等47个类别。其中，“劳动人事”类别最为常见，占比36.94%（20,684例）；其次是“合同纠纷”类别，占比8.98%（5,026例）；其余45个类别共占比54.08%（35,278例）。该数据集按8:1:1的比例划分为训练集（44,738例）、验证集（5,625例）和测试集（5,625例），以支持标准的机器学习工作流程。
## 任务结构
该任务分为两部分：零样本评估和微调实验。
我们提供了本次实验所需要安装的依赖。
  ```bash
  pip install -r requirements.txt
  ```
CCCD数据集从Hugging Face上获取：https://huggingface.co/datasets/gehits/Chinese-Legal-Case-Classification-Dataset
# 评测代码使用方法
模型评测主要分为“模型结果生成”与“模型结果评估”两步。
## 模型结果生成
* 准备数据文件。
* 直接运行`./code/main.py`文件。
    ```bash
    cd code
    MODEL_PATH='xxx'
    MODEL_NAME='xxx'
    DATA_DIR='xxx'
    models='xxx'
    # Zero-shot
    python3 main.py \
        --f_path $test.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../../model_test_result/zero_shot/$MODEL_NAME \
    # If you want to generate multiple files
    python3 main.py \
        --data_dir $DATA_DIR \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/zero_shot/$MODEL_NAME \
    # If you want to generate with multiple models
    python3 main.py \
        --data_dir $DATA_DIR \
        --model_config_file $models \
    ```
    * `--f_path`: 原始数据的路径。
    * `--data_dir`: 原始数据文件的路径。
    * `--model_path`:模型权重的路径。
    * `--model_name`: 模型的名称。你可以在 models.txt 中找到所有可用的模型名称。
    * `--output_dir`:模型生成结果的输出目录名称。
    * `--log_name`: 日志的路径。
    * `--api_key`: 模型的 API 密钥名称，仅在使用 API 时有效。
    * `--api_base`: 模型的 API 基础地址名称，仅在使用 API 时有效。
    * `--device`: cuda 的设备 ID。
    * `--is_vllm`: 是否使用 vLLM 进行更快的推理，目前并非所有模型都支持。
    * `--batch_size`: vLLM 每次推理处理的提问数量，仅在使用 vLLM 时有效。
* 以zero_shot为例，运行完毕后可以进入`./zero_shot_output/$MODEL_NAME`文件夹查看结果`.json`文件，每一行格式如下：
    ```python
    {"input": xxx, "output": xxx, "answer": xxx}
    ```
## 模型结果评估
* 直接运行`./evaluation/evaluate.py`文件
    ```bash
    cd evaluation
    python evaluate.py \
        --input_dir ../../model_output/zero_shot \
        --output_dir ../../model_output/zero_shot \
    ```
    * `--input_dir`: 模型生成的目录路径。
    * `--output_dir`: 评估结果的输出目录。
    * `--metrics_choice`: 多选任务的评估指标，目前支持“Accuracy”。
    * `--model_path`: bert 和 bart 模型的路径，仅在评估指标为“Bertscore”或“Bartscore”时有效。
    * `--device`: cuda 的设备 ID。
* 进入`./evaluation_output/evaluation_result.csv`查看运行完毕的结果，结果如下 

  | Model         | Zero-shot Accuracy |
    |---------------|--------------------|
    | Llama3.2 1B   | 0.324              |
    | Llama3.2 3B   | 0.741              |
    | Llama3.1 8B   | 0.823              |
    | Qwen2.5 1.5B  | 0.815              |
    | Qwen2.5 3B    | 0.828              |
    | Qwen2.5 7B    | 0.844              |
    | Baichuan 7B   | 0.239              |
    | Baichuan 13B   | 0.504              |
    | ChatGLM3 6B   | 0.736              |
    | ChatGLM4 9B   | 0.810              |
    | Gemma2 2B     | 0.524              |
    | Gemma2 9B     | 0.832              |
    | InternLM2 1.8B | 0.799              |
    | InternLM2 7B  | 0.859              |
    | InternLM2 20B | 0.851              |
    | **AVERAGE**   | **0.702**          |

# 微调代码使用说明
微调过程主要包括两个步骤：“LoRA模型微调”和“LoRA微调模型结果生成”。

## LoRA模型微调
* 准备数据文件，
  * 直接运行 `./code/lora_tuning.py`。以下是一个示例：
      ```bash
      cd code
      base_path = "xxxxx"
      model_name = "xxxxx"
      model_name_path = "xxxxx"
      train_json_path = "xxxxx"
      val_json_path = "xxxxx"
      # 微调
      python3 lora_tuning.py \
      ```
      * `--base_path`: 微调后lora权重保存的地址
      * `--model_name`: 微调模型的名称
      * `--model_name_path`: 初始模型的地址
      * `--train_json_path`: 训练集的地址
      * `--val_json_path`: 验证集的地址
## LoRA微调模型结果生成
* 准备数据文件，
  * 直接运行 `./code/lora_main.py`，以下是一个示例：
      ```bash
      cd code
      MODEL_PATH='xxx'
      MODEL_NAME='xxx'
      DATA_DIR='xxx'
      MODEL_PATH_BASE='xxx'
      python3 lora_main.py \
        --data_dir $DATA_DIR \
        --model_path $MODEL_PATH \
        --model_path_base $MODEL_PATH_BASE \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/zero_shot/$MODEL_NAME \
      ```
      * `--model_path`: 微调模型的lora权重地址.
      * `--model_path_base`: 该微调模型的初始模型的地址
      * `--data_dir`: 测试的数据文件地址
      * `--model_name`: 微调模型的名称.


