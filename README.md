# CCCD
Welcome to the Chinese Case Characterization Dataset(CCCD),a comprehensive,high-quality classification dataset containing 55,988 Chinese legal cases across 47 categories.
## Instruction
we launched the Chinese Case Characterization Dataset (CCCD), a large-scale, high-quality classification dataset containing more than 55,988 Chinese legal cases covering 47 categories, including "Labor and Personnel", "Consumer Rights Protection",among them, the “Labor and Personnel” category was the most common, accounting for 36.94% (20,684 cases), followed by the “Contract Dispute” category, accounting for 8.98% (5,026 cases),the remaining 45 categories account for 54.08%(35278 cases). The dataset is divided into a training set (44,738 cases), a validation set (5,625 cases), and a test set (5,625 cases) in a ratio of 8:1:1 to support standard machine learning workflows.

## Task Structure
The task is divided into two parts: Zero-shot Evaluation and Fine-tuning Experiments. 
We provide the requirements for this experiment.
  ```bash
  pip install -r requirements.txt
  ```
The dataset(CCCD) is obtained from hugging face:https://huggingface.co/datasets/gehits/Chinese-Case-Characterization-Dataset/tree/main
# Zero-shot Evaluation:Usage of evaluation code
The evaluation process mainly consists of two steps: "model result generation" and "model result evaluation".
## Model Result Generation
* Prepare the data files,
* Directly run the `./code/main.py`. Here is an example :
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
    * `--f_path`: The path for original data
    * `--data_dir`: The path for original data_files
    * `--model_path`: The path for model's weights.
    * `--model_name`: The name of the model. You can find all available model names in `models.txt`.
    * `--output_dir`: The name of the output directory for model's generation.
    * `--log_name`: The path for the log.
    * `--api_key`: The name of model's api key, only valid when using api.
    * `--api_base`: The name of model's api base, only valid when using api.
    * `--device`: The device id for `cuda`.
    * `--is_vllm`: Whether to use vllm for faster inference, and currently it is not available for all models.
    * `--batch_size`: The number of questions processed per inference time by vllm, only effective when using vllm.
* Take zero-shot setting as an example. You can check the results in `./zero_shot_output/$MODEL_NAME`. The `.json` file format for each line is as follows:
    ```python
    {"input": xxx, "output": xxx, "answer": xxx}
    ```
## Model Result Evaluation
* Directly run `./evaluation/evaluate.py`.
    ```bash
    cd evaluation
    python3 evaluate.py \
        --input_dir ../../model_test_result/zero_shot \
        --output_dir ../../model_test_result/zero_shot \
    ```
    * `--input_dir`: The directory path for the model's generation.
    * `--output_dir`: The output directory for evaluation result.
    * `--metrics_choice`: Evaluation metric for multiple-choice tasks, 'Accuracy' is currently supported.
    * `--model_path`: The path for bert and bart model, only valid when the evaluation metric is 'Bertscore' or 'Bartscore'.
    * `--device`: The device id for `cuda`.
  * Go to `./model_test_result/zero_shot/evaluation_result.csv` to check the full results. The example format is as follows:
    
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

# Fine-tuning Experiments:Usage of fine_tuning code
The fine_tuning process mainly consists of two steps: "lora tuning model" and "lora_tuning_model result generation".
## Lora Tuning Model
* Prepare the data files,
  * Directly run the `./code/lora_tuning.py`. Here is an example :
      ```bash
      cd code
      #Define your own variables
      base_path = "xxxxx"
      model_name = "xxxxx"
      model_name_path = "xxxxx"
      train_json_path = "xxxxx"
      val_json_path = "xxxxx"
      # fine_tuning
      python3 lora_tuning.py \
      ```
       * `--base_path`: The path where the fine-tuned model (LoRA weights) is saved.
       * `--model_name`: The name of the Fine-tuned model.
       * `--model_name_path`: The address of the model to be fine-tuned.
       * `--train_json_path`: The address of the training set
       * `--val_json_path`: The address of the validation set
## Lora_tuning_model Result Generation
* Prepare the data files,
  * Directly run the `./code/lora_main.py`. Here is an example :
      ```bash
      cd code
      MODEL_PATH='xxx'
      MODEL_NAME='xxx'
      DATA_DIR='xxx'
      MODEL_PATH_Base='xxx'
      python3 lora_main.py \
        --data_dir $DATA_DIR \
        --model_path $MODEL_PATH \
        --model_path_base  $MODEL_PATH_Base \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/zero_shot/$MODEL_NAME \
      ```
      * `--model_path`: The path for lora_model's lora weights.
      * `--model_path_base`: The path for model's weights.(original model)
      * `--data_dir`: The path for original data_files
      * `--model_name`: The name of the Fine-tuned model.