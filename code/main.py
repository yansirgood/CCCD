import argparse
import os
import json
import logging
import time
from generation.Llama_base_gen import Llama_base_generator
from generation.gemma2_gen import gemma2_generator
from generation.Chatglm_gen import Chatglm_generator
from generation.Chatglm4_gen import Chatglm4_generator
from generation.Baichuan_base_gen import Baichuan_base_generator
from generation.Qwen_gen import Qwen_generator
from generation.Internlm_base_gen import Internlm_base_generator

def main(args):

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    task_name = args.f_path.split("/")[-1].split(".")[0]
    log_filename = f"{args.model_name}_running.log"
    log_path = os.path.join(args.output_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


    if args.model_name == 'Llama3_2_1B':
        llm_generator = Llama_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Llama3_2_3B':
        llm_generator = Llama_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Llama3_1_8B':
        llm_generator = Llama_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Baichuan_7B':
        llm_generator = Baichuan_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Baichuan_13B':
        llm_generator = Baichuan_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'chatglm3_6B':
        llm_generator = Chatglm_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'chatglm4_9B':
        llm_generator = Chatglm4_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'gemma2_2B':
        llm_generator = gemma2_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'gemma2_9B':
        llm_generator = gemma2_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Qwen2_5_1_5B':
        llm_generator = Qwen_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Qwen2_5_3B':
        llm_generator = Qwen_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Qwen2_5_7B':
        llm_generator = Qwen_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Internlm2_1_8b':
        llm_generator = Internlm_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Internlm2_20b':
        llm_generator = Internlm_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.model_name == 'Internlm2_7b':
        llm_generator = Internlm_base_generator(
            f_path=args.f_path,
            is_few_shot=args.is_few_shot,
            device=args.device,
            is_vllm=args.is_vllm,
            model_path=args.model_path,
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
    q_type = "multiple_choice"
    logging.info(f"Start running {args.model_name} on task {task_name}")
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
    parser.add_argument("--model_path", type=str, default="xxx")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Llama3_2_1B")
    parser.add_argument("--model_config_file", type=str, default="models.txt", help="Path to the model configuration file")
    parser.add_argument("--output_dir", type=str, default="../../model_output/zero_shot/Llama3_2_1B")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    if (args.model_name and args.model_path and args.output_dir):
        model_configs = [{
            'model_name': args.model_name,
            'model_path': args.model_path,
            'output_dir': args.output_dir
    }]
    else:
        model_configs = []
        try:
            with open(args.model_config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        model_name, model_path, output_dir = line.split(',')
                        model_configs.append({
                            'model_name': model_name.strip(),
                            'model_path': model_path.strip(),
                            'output_dir': output_dir.strip()
                        })
        except FileNotFoundError:
            raise FileNotFoundError(f"Model configuration file not found: {args.model_config_file}")
        except ValueError:
            raise ValueError("Each line in the model configuration file must contain model_name,model_path,output_dir")

    if args.f_path:
        print(f"Running single task: {args.f_path}")
        total_runtime = 0
        for config in model_configs:
            print(f"\nRunning model: {config['model_name']}")
            args.model_name = config['model_name']
            args.model_path = config['model_path']
            args.output_dir = config['output_dir']
            try:
                runtime = main(args)
                total_runtime += runtime
                runtime_minutes = runtime / 60
                print(f"Model {args.model_name} completed in {runtime:.2f} seconds ({runtime_minutes:.2f} minutes)")
            except Exception as e:
                print(f"Error in model {args.model_name}: {e}")
                logging.error(f"Error in model {args.model_name}: {e}")
        total_minutes = total_runtime / 60
        print(f"All models completed in {total_runtime:.2f} seconds ({total_minutes:.2f} minutes)")
    else:

        if not args.data_dir:
            raise ValueError("Must specify either --f_path or --data_dir")
        data_files = sorted(
            [f for f in os.listdir(args.data_dir) if f.endswith('.json')],
            key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0]))
        )

        total_runtime = 0
        for data_file in data_files:
            task_name = data_file.replace('.json', '')
            args.f_path = os.path.join(args.data_dir, data_file)

            print(f"Running task: {task_name}")
            task_total_runtime = 0
            for config in model_configs:
                print(f"\nRunning model: {config['model_name']}")
                args.model_name = config['model_name']
                args.model_path = config['model_path']
                args.output_dir = config['output_dir']
                try:
                    runtime = main(args)
                    task_total_runtime += runtime
                    total_runtime += runtime
                    print(f"Model {args.model_name} completed in {runtime:.2f} seconds")
                except Exception as e:
                    print(f"Error in model {args.model_name} for task {task_name}: {e}")
                    logging.error(f"Error in model {args.model_name} for task {task_name}: {e}")
            task_total_minutes = task_total_runtime / 60
            print(f"Task {task_name} completed in {task_total_runtime:.2f} seconds ({task_total_minutes:.2f} minutes)")

        total_minutes = total_runtime / 60
        print(f"All tasks completed in {total_runtime:.2f} seconds ({total_minutes:.2f} minutes)")
    