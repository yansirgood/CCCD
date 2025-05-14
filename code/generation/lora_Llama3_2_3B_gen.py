import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from generation.lora_model_gen import lora_model_generator

class lora_Llama3_2_3B_generator(lora_model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_path_base, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        super(lora_Llama3_2_3B_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_path_base = model_path_base
        self.model_name = model_name
        
    def model_init(self):
        if self.is_vllm:
            raise ValueError("vLLM is not supported for classification tasks in this implementation.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path_base, use_fast=False, bos_token='<s>', add_bos_token=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            base_model = AutoModelForSequenceClassification.from_pretrained(self.model_path_base, num_labels=4).half().eval().cuda()
            model = PeftModel.from_pretrained(base_model, self.model_path)
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
            prompt = instruction + input_text
            try:
                inputs = tokenizer(prompt, return_tensors='pt', padding=True).to('cuda')
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                predicted_label_idx = torch.argmax(logits, dim=-1).item()
                predicted_label = label_map[predicted_label_idx]
                output_ls.append({"input": input_text,
                                  "output": predicted_label,
                                  "answer": answer})
            except:
                num_unsuccess += 1
                print("Fail to classify")
                output_ls.append({"input": input_text,
                                  "output": "未成功分类",
                                  "answer": answer})
        return output_ls, num_unsuccess