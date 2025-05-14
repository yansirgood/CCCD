import argparse
import os
import json
import logging
import time
from generation.lora_Llama3_2_3B_gen import lora_Llama3_2_3B_generator


def main(args):
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    task_name = args.f_path.split("/")[-1].split(".")[0]
    log_filename = f"{args.model_name}_{task_name}_running.log"
    log_path = os.path.join(args.output_dir, log_filename)
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # 追加模式
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.model_name == 'lora_Llama3.2_3B':
        llm_generator = lora_Llama3_2_3B_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            few_shot_path=args.few_shot_path,
            model_path=args.model_path,
            model_path_base=args.model_path_base,  
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'mix_all_train_Llama3.2_3B':
        llm_generator = lora_Llama3_2_3B_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            few_shot_path=args.few_shot_path,
            model_path=args.model_path,
            model_path_base=args.model_path_base,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    print("Model name:", args.model_name)


    if args.model_name != 'gpt-4' and args.model_name != 'gpt-3.5-turbo':
        model, tokenizer = llm_generator.model_init()
    else:
        model, tokenizer = None, None
    task_num = args.f_path.split("/")[-1].split(".")[0].split("_")[0]
    q_type = "multiple_choice" if task_num != "5" else "generation"
    logging.info(f"Start running {args.model_name} on task {task_num}_{args.f_path.split('/')[-1].split('.')[0].split('_')[1]}")
    output_ls, num_unsuccess = llm_generator.generate_output(q_type=q_type, tokenizer=tokenizer, model=model, batch_size=args.batch_size)


    outfile_name = os.path.join(args.output_dir, f"{args.model_name}_{args.f_path.split('/')[-1]}")
    with open(outfile_name, 'w', encoding='utf8') as outfile:
        for save_dict in output_ls:
            outline = json.dumps(save_dict, ensure_ascii=False) + '\n'
            outfile.write(outline)
    if num_unsuccess != 0:
        logging.info(f"Number of failure generations for {args.f_path}: {num_unsuccess}")
    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60
    logging.info(f"Experiment completed in {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

    return runtime_seconds
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data", help="Directory containing JSON data files")
    parser.add_argument('--f_path', type=str, default=None, help='Path to a single JSON file')
    parser.add_argument("--is_few_shot", action="store_true")
    parser.add_argument("--is_vllm", action="store_true")
    parser.add_argument("--few_shot_dir", type=str, default=None, help="Directory containing few-shot JSON files")
    parser.add_argument("--few_shot_path", type=str, default=None, help="Directory containing few-shot JSON")
    parser.add_argument("--model_path", type=str, default="xxx", help="Path to the LoRA model")
    parser.add_argument("--model_path_base", type=str, default=None, help="Path to the base LLaMA model")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="lora_Llama3_2_3B")
    parser.add_argument("--output_dir", type=str, default="../../model_output/zero_shot/")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.f_path:
        if args.is_few_shot and not hasattr(args, 'few_shot_path'):
            # 自动生成 few_shot_path
            args.few_shot_path = args.f_path.replace('.json', '_few_shot.json')
            if not os.path.exists(args.few_shot_path):
                raise FileNotFoundError(f"Few-shot file not found: {args.few_shot_path}")
        print(f"Running single task: {args.f_path}")
        try:
            runtime = main(args)
            runtime_minutes = runtime / 60
            print(f"Task completed in {runtime:.2f} seconds ({runtime_minutes:.2f} minutes)")
        except Exception as e:
            print(f"Error in task: {e}")
            logging.error(f"Error in task: {e}")
    else:
        if not args.data_dir:
            raise ValueError("Must specify either --f_path or --data_dir")
        data_files = sorted(
            [f for f in os.listdir(args.data_dir) if f.endswith('.json') and not f.endswith('_few_shot.json')],
            key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0]))
        )

        if args.is_few_shot:
            if args.few_shot_dir is None:
                args.few_shot_dir = args.data_dir  # 默认与数据文件在同一目录
            few_shot_files = {
                f.replace('.json', ''): f.replace('.json', '_few_shot.json') for f in data_files
            }
        else:
            few_shot_files = {f: None for f in data_files}
        total_runtime = 0
        for data_file in data_files:
            task_name = data_file.replace('.json', '')
            args.f_path = os.path.join(args.data_dir, data_file)
            args.few_shot_path = os.path.join(args.few_shot_dir, few_shot_files[data_file]) if args.is_few_shot else None
            print(f"Running task: {task_name}")
            try:
                runtime = main(args)
                total_runtime += runtime
                print(f"Task {task_name} completed in {runtime:.2f} seconds")
            except Exception as e:
                print(f"Error in task {task_name}: {e}")
                logging.error(f"Error in task {task_name}: {e}")
                continue
        total_minutes = total_runtime / 60
        print(f"All tasks completed in {total_runtime:.2f} seconds ({total_minutes:.2f} minutes)")