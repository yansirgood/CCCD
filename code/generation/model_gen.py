import json
from vllm import LLM, SamplingParams

class model_generator:
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, tensor_parallel_size=None, gpu_memory_utilization=None):
        '''
        Args:
            f_path: The path for original data file in json format
            is_few_shot: Whether to utilize few-shot setting
            device: Device number of cuda
            is_vllm: Whether to use vLLM for inference
            model_path: The path of the model
            tensor_parallel_size: Numbers of GPUs to use
            gpu_memory_utilization: Proportion of memory to use
        '''
        self.f_path = f_path
        self.model_path = model_path
        self.is_few_shot = is_few_shot
        self.device = device
        if is_vllm and (tensor_parallel_size is None or gpu_memory_utilization is None):
            raise ValueError("Please provide the tensor_parallel_size and gpu_memory_utilization, current value is none")
        self.is_vllm = is_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

    def model_init_vllm(self):
        '''
        Init the model using vLLM
        '''
        model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer

    @staticmethod
    def truncate_long(prompt, context_length, tokenizer, q_type):
        '''
        For question with long context, truncate it.
        Args:
            prompt: Original prompt for the model
            context_length: Maximum context length for the model
            tokenizer: Tokenizer for the model
            q_type: Must be 'generation' or 'multiple_choice'
        '''
        ori_prompt = tokenizer.encode(prompt)
        if q_type == 'generation':
            if len(ori_prompt) > context_length - 512:
                print(f"Input tokens too long, cut to {context_length-512} tokens!")
                half = int((context_length-512) / 2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
        elif q_type == 'multiple_choice':
            if len(ori_prompt) > context_length - 20:
                print(f"Input tokens too long, cut to {context_length-20} tokens!")
                half = int((context_length-20) / 2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True) + tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
        else:
            raise ValueError(f"Wrong question type, q_type must be 'generation' or 'multiple_choice' but get {q_type}")
        return prompt

    def process_prompt(self, task_name):
        '''
        Load the test data and prepare the prompt
        Args:
            task_name: Name of the task (e.g., "1_4")
        Returns:
            instruction_ls: List of instructions
            input_text_ls: List of input texts
            answer_ls: List of answers
        '''
        print("Loading test data...")
        with open(self.f_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        instruction_ls, input_text_ls, answer_ls = [], [], []
        for idx, line in enumerate(lines):
            qa_dict = json.loads(line)
            instruction = qa_dict['instruction']
            answer = qa_dict['answer']
            input_text = qa_dict['input'] + '\n' + '答案:'
            instruction_ls.append(instruction)
            input_text_ls.append(input_text)
            answer_ls.append(answer)
        print(f"Loaded {len(instruction_ls)} test samples.")
        return instruction_ls, input_text_ls, answer_ls